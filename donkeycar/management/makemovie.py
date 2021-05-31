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
    from tf_keras_vis.gradcam import GradcamPlusPlus
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
        self.pilot_angle = 0.0
        self.pilot_throttle = 0.0
        self.pilot_score = 1.0 # used for color intensity
        self.user_angle = 0.0
        self.user_throttle = 0.0
        self.control_score = 0.25 # used for control size
        self.flag_test_pilot_angle = 1
        self.flag_test_pilot_throttle = 1
        self.flag_test_user_angle = 1
        self.flag_test_user_throttle = 1
        self.throttle_circle_pilot_angle = 0
        self.throttle_circle_user_angle = 0
        self.is_test = False
        self.last_pilot_throttle = 0.0 # used for color transparency
        self.last_user_throttle = 0.0 # used for color transparency
        self.pilot_throttle_trans = 1.0 # used for color transparency
        self.user_throttle_trans = 1.0 # used for color transparency
        self.pilot_throttle_trans_rate = 0.25 # used for color transparency
        self.user_throttle_trans_rate = 0.25 # used for color transparency


        if args.model is not None:
            self.keras_part = get_model_by_type(args.type, cfg=cfg)
            self.keras_part.load(args.model)
            if args.salient:
                self.do_salient = self.init_salient(self.keras_part.model)

        print('making movie', args.out, 'from', num_frames, 'images')
        clip = mpy.VideoClip(self.make_frame, duration=((num_frames - 1) / cfg.DRIVE_LOOP_HZ))
        clip.write_videofile(args.out, fps=cfg.DRIVE_LOOP_HZ)

    @staticmethod
    def draw_line_into_image(angle, throttle, is_pred, img, color):
        """
        is_pred:
            True: from draw_model_prediction()
            False: from draw_user_input()
        """

        height, width, _ = img.shape
        mid_h = height//2
        length = height//4
        a1 = angle * 45.0
        l1 = throttle * length
        mid_w = width // 2 + (- 1 if is_pred else +1)

        p1 = tuple((mid_w - 2, mid_h - 1))
        p11 = tuple((int(p1[0] + l1 * math.cos((a1 + 270.0) * DEG_TO_RAD)),
                     int(p1[1] + l1 * math.sin((a1 + 270.0) * DEG_TO_RAD))))

        cv2.line(img, p1, p11, color, 2)

    def get_user_input(self, record, img):
        """
        Get the user input from record
        """
        if self.is_test:
            self.user_angle_a = float(record["user/angle"])
            self.user_throttle_a = float(record["user/throttle"])
        else:
            self.user_angle = float(record["user/angle"])
            self.user_throttle = float(record["user/throttle"])
        
    def get_model_prediction(self, img, salient_image):
        """
        Get the pilot input from model prediction
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

        if self.is_test:
            self.pilot_angle_a, self.pilot_throttle_a = self.keras_part.run(img)
        else:
            self.pilot_angle, self.pilot_throttle = self.keras_part.run(img)


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
        # Create GradCAM++ object, Just only repalce class name to "GradcamPlusPlus"
        self.gradcampp = GradcamPlusPlus(model,
                                         model_modifier=self.model_modifier,
                                         clone=False)
 
        return True

    def draw_gradcam_pp(self, img):

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
        salient_map = salient_map * 255
        salient_mask = cv2.cvtColor(salient_map, cv2.COLOR_GRAY2RGB)
        salient = cv2.applyColorMap(salient_map.astype('uint8'), cv2.COLORMAP_JET)
        salient = cv2.applyColorMap(salient, cv2.COLOR_BGR2RGB)
        salient = cv2.bitwise_and(salient, salient_mask.astype('uint8'))

        blend = cv2.addWeighted(img, 1.0, salient, 1.0, 0.0)

        return blend
        

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
            #salient_image = self.draw_gradcam_pp(camera_image)
            salient_image = salient_image.astype('uint8')
        if salient_image is None:
            salient_image = camera_image
        # resize
        if self.scale != 1:
            h, w, d = salient_image.shape
            dsize = (w * self.scale, h * self.scale)
            salient_image = cv2.resize(salient_image, dsize=dsize, interpolation=cv2.INTER_CUBIC)

        # draw control
        if self.keras_part is not None:
            self.get_model_prediction(camera_image, salient_image)
            self.draw_steering_distribution(camera_image, salient_image)
        if self.user: self.get_user_input(rec, salient_image)
        salient_image = self.draw_control_into_image(salient_image)
        
        """
        # left upper text
        display_str = []
        display_str.append(f"pilot_angle: {self.pilot_angle:.2f}")
        display_str.append(f"pilot_throttle: {self.pilot_throttle:.2f}")
        self.draw_text(salient_image, display_str)
        """

        self.current += 1
        # returns a 8-bit RGB array
        return salient_image

    def model_modifier(self, m):
        m.layers[-1].activation = tf.keras.activations.linear

    def loss(self, output):
        return (output[0])

    def draw_control_into_image(self, img):
        """
        test
        self.pilot_score = np.random.default_rng().uniform(low=0.2, high=1.0)
        self.pilot_throttle = np.random.default_rng().uniform(low=-1.0, high=1.0)
        self.pilot_angle = np.random.default_rng().uniform(low=-1.0, high=1.0)
        """
        if self.is_test:
            if self.pilot_angle >= 1.0:
                self.flag_test_pilot_angle = -0.1
            elif self.pilot_angle <= -1.0:
                self.flag_test_pilot_angle = +0.1
            self.pilot_angle += self.flag_test_pilot_angle

            if self.pilot_throttle >= 1.0:
                self.flag_test_pilot_throttle = -0.01
            elif self.pilot_throttle <= -1.0:
                self.flag_test_pilot_throttle = +0.01
            self.pilot_throttle += self.flag_test_pilot_throttle

            if self.user_angle >= 1.0:
                self.flag_test_user_angle = -0.05
            elif self.user_angle <= -1.0:
                self.flag_test_user_angle = +0.05
            self.user_angle += self.flag_test_user_angle

            if self.user_throttle >= 1.0:
                self.flag_test_user_throttle = -0.02
            elif self.user_throttle <= -1.0:
                self.flag_test_user_throttle = +0.02
            self.user_throttle += self.flag_test_user_throttle

            #self.control_score = abs(self.pilot_throttle/4*3) + 0.25

        height, width, _ = img.shape
        y = height//2
        x = width//2

        # prepare ellipse mask
        r_base = 6.0
        r_mask = int(height//r_base)
        white = np.ones_like(img)*255
        ellipse = np.zeros_like(img)
        

        r_pilot = int(height//r_base+(height//3.1)*self.control_score)
        r_user = int(height//r_base+(height//6.2)*self.control_score)

        pilot_trans, user_trans = self.trans_make()

        # draw pilot control
        green = (0,int(255*pilot_trans),0) # green for reverse
        blue = (0,0,int(255*pilot_trans)) # blue for reverse
        self.draw_ellipse(self.pilot_angle, self.pilot_throttle, x,y,r_pilot, ellipse, green, blue)

        pilot_circle_mask = cv2.circle(white, (int(x),int(y)), int(r_user+(height//10)*self.control_score), (0,0,0), -1).astype('uint8')
        # pilot mask
        ellipse = cv2.bitwise_and(ellipse, pilot_circle_mask)

        # draw user control
        green = (0,int(255*user_trans),0) # green for reverse
        blue = (0,0,int(255*user_trans)) # blue for reverse
        orange = (255,69,0) # orange for reverse
        self.draw_ellipse(self.user_angle, self.user_throttle, x,y,r_user, ellipse, green, blue)
        white = np.ones_like(img)*255

        user_circle_mask = cv2.circle(white, (int(x),int(y)), int(r_mask+(height//10)*self.control_score), (0,0,0), -1).astype('uint8')
        # user mask
        ellipse = cv2.bitwise_and(ellipse, user_circle_mask)
        
        # draw circle
        color1= (0,0,215)
        color2= (0,25,78)
        color3= (0,255,215)
        cv2.circle(ellipse, (int(x),int(y)), int(r_pilot), color1, 1)
        cv2.circle(ellipse, (int(x),int(y)), int(r_user), color2, 1)
        cv2.circle(ellipse, (int(x),int(y)), int(r_mask), color3, 1)

        # draw dot circle
        ellipse = self.draw_dot_circle2(ellipse, x,y,r_pilot+1+self.scale//4, 1+self.scale//4, (0,255,218), -1, 4*self.scale, self.pilot_throttle, True)
        ellipse = self.draw_dot_circle2(ellipse, x,y,r_user+1+self.scale//4, 1+self.scale//4, (0,255,218), -1, 4*self.scale, self.user_throttle, False)


        # draw speed meter
        #self.draw_analog_meter(ellipse, self.pilot_speed)
        self.draw_analog_direction_meter(ellipse, self.pilot_angle, self.pilot_throttle)
        ellipse = self.draw_digital_meter(ellipse, r_mask-(2+self.scale//4), self.pilot_throttle, 18*self.scale, pilot_trans)
        ellipse = self.draw_digital_meter(ellipse, r_mask-(2+self.scale//4)*2-(width//10)*self.control_score, self.user_throttle, 12*self.scale, user_trans)

        #self.draw_analog_meter(ellipse, -0.75)
        #print(f"r_pilot: {r_pilot}")
        # draw stripe circle
        if self.pilot_throttle >= 0:
            flag_pilot_throttle = 1
            add_pilot_deg = 180
        else:
            flag_pilot_throttle = -1
            add_pilot_deg = 0
        if self.user_throttle >= 0:
            flag_user_throttle = 1
            add_user_deg = 180
        else:
            flag_user_throttle = -1
            add_user_deg = 0
        """
        p1 = (x,y-flag_pilot_throttle*r_pilot)
        p2 = (x,y+flag_pilot_throttle*y//10)
        red = (255,0,0)
        pts = self.points_rotation([p1,p2], center=(x,y), degrees=flag_pilot_throttle*self.pilot_angle*90)
        cv2.line(ellipse, pts[0], pts[1], red, 2)
        cv2.line(ellipse, p1, p2, red, 2)
        """
        red=(255,0,0)
        self.draw_arc_center_line(ellipse, x,y,r_pilot+1+(height//40),r_user+(height//20)*self.control_score, red, 2, degrees=flag_pilot_throttle*self.pilot_angle*90+add_pilot_deg)
        self.draw_arc_center_line(ellipse, x,y,r_user+1+(height//40),r_mask+(height//20)*self.control_score, red, 2, degrees=flag_user_throttle*self.user_angle*90+add_user_deg)

        

        # left upper text
        """
        display_str = []
        display_str.append(f"x: {x}")
        display_str.append(f"y: {y}")
        display_str.append(f"r_pilot: {r_pilot}")
        display_str.append(f"angle: {self.pilot_angle:.2f}")
        display_str.append(f"throttle: {self.pilot_throttle:.2f}")
        self.draw_text(ellipse, display_str)
        """

        # blur
        if self.scale <= 3:
            ellipse = cv2.GaussianBlur(ellipse,(3,3),0)
        else:
            ellipse = cv2.GaussianBlur(ellipse,(5,5),0)
            

        self.last_pilot_throttle = self.pilot_throttle
        self.last_user_throttle = self.user_throttle

        return cv2.addWeighted(img, 1.0, ellipse, 1.0, 0.0)

    def trans_make(self):
        if self.pilot_throttle != self.last_pilot_throttle:
            if self.pilot_throttle_trans >= 0.8:
                self.pilot_throttle_trans_rate = -0.25
            elif self.pilot_throttle_trans <= 0.3:
                self.pilot_throttle_trans_rate = +0.25
            self.pilot_throttle_trans += self.pilot_throttle_trans_rate
        else:
            self.pilot_throttle_trans = 1.0
        if self.user_throttle != self.last_user_throttle:
            if self.user_throttle_trans >= 0.8:
                self.user_throttle_trans_rate = -0.25
            elif self.user_throttle_trans <= 0.3:
                self.user_throttle_trans_rate = +0.25
            self.user_throttle_trans += self.user_throttle_trans_rate
        else:
            self.user_throttle_trans = 1.0
        return self.pilot_throttle_trans, self.user_throttle_trans

    def draw_analog_meter(self, img, value):
        height, width, _ = img.shape
        y = height//2
        x = width//2

        r_base = 6.0
        r_pilot = int(height//r_base+(height//3.1)*self.control_score)
        red = (255,0,0)

        p1 = (x,y-r_pilot)
        p2 = (x,y+y//10)
        pts = self.points_rotation([p1,p2], center=(x,y), degrees=value*180-90)
        cv2.line(img, tuple(pts[0]), tuple(pts[1]), red, 2)


    def draw_analog_direction_meter(self, img, angle, throttle):
        height, width, _ = img.shape
        y = height//2
        x = width//2

        r_base = 6.0
        r_pilot = int(height//r_base+(height//3.1)*self.control_score)
        red = (255,0,0)

        p1 = (x,y-r_pilot)
        p2 = (x,y+y//10)
        if throttle >= 0:
            flag_throttle = 1
            add_deg = 0
        else:
            flag_throttle = -1
            #add_deg = 180
            add_deg = 0

        pts = self.points_rotation([p1,p2], center=(x,y), degrees=flag_throttle*angle*90+add_deg)
        cv2.line(img, tuple(pts[0]), tuple(pts[1]), red, 2)
        

    def draw_arc_center_line(self, img, x,y,r1,r2, color, thickness, degrees):
        base_point1 = np.array([0,r1])
        base_point2 = np.array([0,r2])
        rot = self.rot(degrees)
        p1 = np.dot(rot, base_point1) # NEVER DO .astype('uint8'). x,y is more than 255.
        p2 = np.dot(rot, base_point2)
        cv2.line(img, (int(p1[0]+x),int(p1[1]+y)), (int(p2[0]+x),int(p2[1]+y)), color, thickness)


    def color_make(self,value):
        """
        Rainbow color maker.
        value: -1.0 to 1.0

        abs(value) 0.0: blue
        abs(value) 0.5: green
        abs(value) 1.0: red
        """
        value = abs(value)
        if value > 1:
            value = 1

        c = int(255*value)
        c1 = int(255*(value*2-0.5))
        c05 = int(255*value*2)
        if c > 255:
            c = 255
        elif c < 0:
            c = 0
        if c1 > 255:
            c1 = 255
        elif c1 < 0:
            c1 = 0
        if c05 > 255:
            c05 = 255
        elif c05 < 0:
            c05 = 0

        if 0 <= value and value < 0.5:
            color = (0,c05,255-c05) # blue -> green
        elif 0.5 <= value and value <= 1.0:
            color = (c1,c05-c1,0) # green -> red
        elif 1.0 < value:
            color = (255,0,0) # red

        return color


    def draw_digital_meter(self, img, r_mask, value, num, trans):
        """
        Rainbow digital throttle meter.
        img: image to draw
        r_mask: circumferential radius
        value: -1.0 to 1.0
        num: number of boxes
        trans: when the value is changed, boxes transparency is changed with this value
        """
        if num > 36:
            num = 36
        height, width, _ = img.shape
        y = height//2
        x = width//2
        h = (height//20)*self.control_score
        w = (width//10)*self.control_score
        base_points = np.array([[r_mask,h/2],[r_mask,-h/2],[-w+r_mask,-h/2],[-w+r_mask,h/2]])
        dot_angle = 360.0/num
        center = np.atleast_2d((x,y))
        start_angle = 145

        mask_img = np.zeros_like(img)
        meter_img = np.zeros_like(img)
        self.draw_digital_mask(value,x,y,r_mask,mask_img)
        
        for i in range(0, num):
            deg = i * dot_angle + start_angle
            box = self.points_rotation(base_points,center=(0,0),degrees=deg)
            box = ((box.T + center.T).T)
            if i == 0:
                color = (0,0,255*trans)
            else:
                if value >= 0:
                    color = self.color_make(i*dot_angle/270)
                else:
                    color = self.color_make((360-i*dot_angle)/270)
                color=(color[0]*trans,color[1]*trans,color[2]*trans)
            cv2.fillPoly(meter_img, np.array([box], dtype=np.int32), color)

        meter_img = cv2.bitwise_and(meter_img.astype('uint8'), mask_img.astype('uint8'))
        img = cv2.bitwise_or(img, meter_img.astype('uint8'))
        return img


    def draw_dot_circle(self, img, x,y,r, cr, color, thickness, num):
        """
        Draw outer dot circle.
        """
        if num > 36:
            num = 36
        base_point = np.array([r,0])
        dot_angle = 360.0/num
        for i in range(0, num):
            deg = i * dot_angle
            rot = self.rot(deg)
            rotated = np.dot(rot, base_point)
            img = cv2.circle(img, (int(rotated[0]+x),int(rotated[1]+y)), cr, color, thickness).astype('uint8')
        return img

    def draw_dot_circle2(self, img, x,y,r, cr, color, thickness, num, throttle, is_pilot):
        """
        Draw outer dot circle.
        This circle rotates according to the throttle.
        """
        if num > 36:
            num = 36
        dot_angle = 360.0/num
        rot_spd = throttle*dot_angle
        if rot_spd > dot_angle/2.0: # rotation speed limit
            rot_spd = dot_angle/2.0
        elif rot_spd < -dot_angle/2.0:
            rot_spd = -dot_angle/2.0
        if is_pilot:
            self.throttle_circle_pilot_angle += rot_spd
            start_angle = self.throttle_circle_pilot_angle
        else:
            self.throttle_circle_user_angle += rot_spd
            start_angle = self.throttle_circle_user_angle
        rot = self.rot(start_angle)
        base_point = np.dot(rot,np.array([r,0]))
        throttle = abs(throttle)
        for i in range(0, num):
            deg = i * dot_angle
            rot = self.rot(deg)
            rotated = np.dot(rot, base_point)
            
            color = self.color_make(throttle)
            img = cv2.circle(img, (int(rotated[0]+x),int(rotated[1]+y)), cr, color, thickness).astype('uint8')
            #print(f'i: {i}, deg: {deg}, (x,y): ({int(rotated[0]+x),int(rotated[1]+y)})')
        return img

    def rot(self, degrees):
        rad = np.deg2rad(degrees)
        return np.array([[np.cos(rad), -np.sin(rad)],
                         [np.sin(rad), np.cos(rad)]])

    def points_rotation_test(self, pts, center=(0,0), degrees=0):
        """
        the same as points_rotation()
        """
        res = None
        for pt in pts:
            base_point = np.array([pt[0]-center[0], pt[1]-center[1]])
            rot = self.rot(degrees)
            p = np.dot(rot, base_point)
            p = (int(p[0]+center[0]), int(p[1]+center[1]))
            if res is None:
                res = p
            else:
                res = np.vstack((res,p))
        return res


    def points_rotation(self, pts, center=(0,0), degrees=0):
        """
        https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
        This function rotates the polygon coordinates.
        pts: 2D polygon coordinates 
        center: Center coordinates of rotation
        degrees: angle
        """
        R = self.rot(degrees)
        o = np.atleast_2d(center)
        pts = np.atleast_2d(pts)
        return np.squeeze((R @ (pts.T-o.T) + o.T).T).astype('int32') # 'int16' or more. x,y is larger than 255. cv2.fillPoly required int32


    def draw_ellipse(self, angle, throttle, x,y,r, img, color, reverse_color):
        """
        Draw the outer arc.
        """
        angle_deg = int(throttle * 270)
        if throttle < 0:
            center_deg = +90 + abs(angle_deg/2)
            color = reverse_color
        else:
            center_deg = -90 - abs(angle_deg/2)

        if throttle < 0:
            angle = angle * -1.0
        rotate_deg = int(angle * 90) + center_deg

        cv2.ellipse(img,(int(x),int(y)),(int(r),int(r)),rotate_deg,0,angle_deg,color,-1)
        

    def draw_digital_mask(self, throttle, x,y,r, img):
        """
        throttle: -1.0 to 1.0
        x: x coordinate in the center of the image
        y: y coordinate in the center of the image
        r: radius of arc
        img: mask image
        """
        angle_deg = int(throttle * 270) # ellipse angle
        rotate_deg = 145 # ellipse start position
        color=(255,255,255)
        cv2.ellipse(img,(int(x),int(y)),(int(r),int(r)),rotate_deg,0,angle_deg,color,-1)
        

    def draw_text(self, img, display_str=[]):
        """ STATISTICS FONT """
        fontScale = img.shape[0]/1000.0
        if fontScale < 0.4:
            fontScale = 0.4
        fontThickness = 1 + int(fontScale)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX

        max_text_width = 0
        max_text_height = 0

        """ DRAW BLACK BOX AND TEXT """
        [(text_width, text_height), baseLine] = cv2.getTextSize(text=display_str[0], fontFace=fontFace, fontScale=fontScale, thickness=fontThickness)
        x_left = int(baseLine)
        y_top = int(baseLine)
        for i in range(len(display_str)):
            [(text_width, text_height), baseLine] = cv2.getTextSize(text=display_str[i], fontFace=fontFace, fontScale=fontScale, thickness=fontThickness)
            if max_text_width < text_width:
                max_text_width = text_width
            if max_text_height < text_height:
                max_text_height = text_height
        """ DRAW BLACK BOX """
        cv2.rectangle(img, (x_left - 2, int(y_top)), (int(x_left + max_text_width + 2), int(y_top + len(display_str)*max_text_height*1.2+baseLine)), color=(0, 0, 0), thickness=-1)
        """ DRAW FPS, TEXT """
        for i in range(len(display_str)):
            cv2.putText(img, display_str[i], org=(x_left, y_top + int(max_text_height*1.2 + (max_text_height*1.2 * i))), fontFace=fontFace, fontScale=fontScale, thickness=fontThickness, color=(77, 255, 9))

