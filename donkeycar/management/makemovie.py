import moviepy.editor as mpy
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import cv2
from matplotlib import pyplot as plt

try:
    from tf_keras_vis import utils
    from tf_keras_vis.activation_maximization import ActivationMaximization
    from tf_keras_vis.utils.scores import CategoricalScore
    from tf_keras_vis.saliency import Saliency
    #from tf_keras_vis.gradcam import GradcamPlusPlus
    from tf_keras_vis.scorecam import ScoreCAM
    from tf_keras_vis.utils import normalize
    from tf_keras_vis.utils.callbacks import Print
except:
    raise Exception("Please install tf-keras-vis: pip install tf-keras-vis")

import donkeycar as dk
from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import *
import numpy as np

DEG_TO_RAD = math.pi / 180.0

cfg = None

class MakeMovie(object):

    def run(self, args, parser):
        '''
        Load the images from a tub and create a movie from them.
        Movie
        '''
        global cfg

        if args.tub is None:
            print("ERR>> --tub argument missing.")
            parser.print_help()
            return

        conf = os.path.expanduser(args.config)
        if not os.path.exists(conf):
            print("No config file at location: %s. Add --config to specify\
                 location or run from dir containing config.py." % conf)
            return

        cfg = dk.load_config(conf)

        if args.type is None and args.model is not None:
            args.type = cfg.DEFAULT_MODEL_TYPE
            print("Model type not provided. Using default model type from config file")

        if args.salient:
            if args.model is None:
                print("ERR>> salient visualization requires a model. Pass with the --model arg.")
                parser.print_help()

            if args.type not in ['linear', 'categorical']:
                print("Model type {} is not supported. Only linear or categorical is supported for salient visualization".format(args.type))
                parser.print_help()
                return

        self.tub = Tub(args.tub)

        start = args.start
        self.end_index = args.end if args.end != -1 else len(self.tub)
        num_frames = self.end_index - start

        # Move to the correct offset
        self.current = 0
        self.iterator = self.tub.__iter__()
        while self.current < start:
            self.iterator.next()
            self.current += 1

        self.scale = args.scale
        self.keras_part = None
        self.do_salient = False
        self.user = args.draw_user_input
        if args.model is not None:
            self.keras_part = get_model_by_type(args.type, cfg=cfg)
            self.keras_part.load(args.model)
            if args.salient:
                self.do_salient = self.init_salient(self.keras_part.model)

        print('making movie', args.out, 'from', num_frames, 'images')
        clip = mpy.VideoClip(self.make_frame, duration=((num_frames - 1) / cfg.DRIVE_LOOP_HZ))
        clip.write_videofile(args.out, fps=cfg.DRIVE_LOOP_HZ)

    @staticmethod
    def draw_line_into_image(angle, throttle, is_left, img, color):

        height, width, _ = img.shape
        length = height
        a1 = angle * 45.0
        l1 = throttle * length
        mid = width // 2 + (- 1 if is_left else +1)

        p1 = tuple((mid - 2, height - 1))
        p11 = tuple((int(p1[0] + l1 * math.cos((a1 + 270.0) * DEG_TO_RAD)),
                     int(p1[1] + l1 * math.sin((a1 + 270.0) * DEG_TO_RAD))))

        cv2.line(img, p1, p11, color, 2)

    def draw_user_input(self, record, img):
        """
        Draw the user input as a green line on the image
        """
        user_angle = float(record["user/angle"])
        user_throttle = float(record["user/throttle"])
        green = (0, 255, 0)
        self.draw_line_into_image(user_angle, user_throttle, False, img, green)
        
    def draw_model_prediction(self, img, salient_image):
        """
        query the model for it's prediction, draw the predictions
        as a red line on the image
        """
        if self.keras_part is None:
            return

        expected = tuple(self.keras_part.get_input_shape()[1:])
        actual = img.shape

        # if model expects grey-scale but got rgb, covert
        if expected[2] == 1 and actual[2] == 3:
            # normalize image before grey conversion
            grey_img = rgb2gray(img)
            actual = grey_img.shape
            img = grey_img.reshape(grey_img.shape + (1,))

        if expected != actual:
            print(f"expected input dim {expected} didn't match actual dim "
                  f"{actual}")
            return

        red = (255, 0, 0)
        pilot_angle, pilot_throttle = self.keras_part.run(img)
        self.draw_line_into_image(pilot_angle, pilot_throttle, True, salient_image, red)

    def draw_steering_distribution(self, img, salient_image):
        """
        query the model for it's prediction, draw the distribution of
        steering choices
        """
        from donkeycar.parts.keras import KerasCategorical

        if self.keras_part is None or type(self.keras_part) is not KerasCategorical:
            return

        pred_img = normalize_image(img)
        pred_img = pred_img.reshape((1,) + pred_img.shape)
        angle_binned, _ = self.keras_part.model.predict(pred_img)

        x = 4
        dx = 4
        y = cfg.IMAGE_H - 4
        iArgMax = np.argmax(angle_binned)
        for i in range(15):
            p1 = (x, y)
            p2 = (x, y - int(angle_binned[0][i] * 100.0))
            if i == iArgMax:
                cv2.line(salient_image, p1, p2, (255, 0, 0), 2)
            else:
                cv2.line(salient_image, p1, p2, (200, 200, 200), 2)
            x += dx

    def init_salient(self, model):
        # Utility to search for layer index by name. 
        # Alternatively we can specify this as -1 since it corresponds to the last layer.
        model.summary()
        self.output_names = []

        for i, layer in enumerate(model.layers):
            if "dropout" not in layer.name.lower() and "out" in layer.name.lower():
                self.output_names += [layer.name]

        if len(self.output_names) == 0:
            print("Failed to find the model layer named with 'out'. Skipping salient.")
            return False

        print("####################")
        print("Visualizing activations on layer:", *self.output_names)
        print("####################")
        
        # Create Saliency object.
        # If `clone` is True(default), the `model` will be cloned,
        # so the `model` instance will be NOT modified, but it takes a machine resources.
        self.saliency = Saliency(model,
                                 model_modifier=self.model_modifier,
                                 clone=False)
        """
        # Create GradCAM++ object, Just only repalce class name to "GradcamPlusPlus"
        self.gradcampp = GradcamPlusPlus(model,
                                         model_modifier=self.model_modifier,
                                         clone=False)
        """
 
        return True

    def draw_gradcam_plus_plus(self, img):

        x = preprocess_input(img, mode='tf')

        # Generate heatmap with GradCAM++
        salient_map = self.gradcampp(self.loss,
                             x,
                             penultimate_layer=-1, # model.layers number
                             )

        return self.draw_mask(img, salient_map)


    def draw_salient(self, img):
        # https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
        x = preprocess_input(img, mode='tf')

        # Generate saliency map with smoothing that reduce noise by adding noise
        salient_map = self.saliency(self.loss, x)
        return self.draw_mask(img, salient_map)


    def draw_mask(self, img, salient_map):
        if salient_map[0].size != cfg.IMAGE_W * cfg.IMAGE_H:
            print("salient size failed.")
            return

        salient_map = salient_map[0]
        salient_map = cv2.cvtColor(salient_map, cv2.COLOR_GRAY2RGB)

        salient = img * salient_map

        return salient
        

    def make_frame(self, t):
        '''
        Callback to return an image from from our tub records.
        This is called from the VideoClip as it references a time.
        We don't use t to reference the frame, but instead increment
        a frame counter. This assumes sequential access.
        '''

        if self.current >= self.end_index:
            return None

        rec = self.iterator.next()
        img_path = os.path.join(self.tub.images_base_path, rec['cam/image_array'])
        camera_image = img_to_arr(Image.open(img_path))

        salient_image = None
        if self.do_salient:
            salient_image = self.draw_salient(camera_image)
            #salient_image = self.draw_gradcam_plus_plus(camera_image)
            salient_image = salient_image.astype('uint8')
        if salient_image is None:
            salient_image = camera_image

        if self.user: self.draw_user_input(rec, salient_image)
        if self.keras_part is not None:
            self.draw_model_prediction(camera_image, salient_image)
            self.draw_steering_distribution(camera_image, salient_image)

        if self.scale != 1:
            h, w, d = salient_image.shape
            dsize = (w * self.scale, h * self.scale)
            salient_image = cv2.resize(salient_image, dsize=dsize, interpolation=cv2.INTER_CUBIC)

        self.current += 1
        # returns a 8-bit RGB array
        return salient_image

    def model_modifier(self, m):
        m.layers[-1].activation = tf.keras.activations.linear

    def loss(self, output):
        return (output[0])

