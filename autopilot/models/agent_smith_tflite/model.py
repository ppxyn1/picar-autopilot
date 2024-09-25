import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.layers import Dense
import cv2


class Model:
    def __init__(self):
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' # enable GPU
    
        path = '/home/pi/autopilot/autopilot/models/agent_smith_tflite/models.tflite'
        self.custom_model_path = path #'/home/pi/autopilot/autopilot/models/agent_smith/test12.tflite'
        
        try: 
            delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1') 
            print('Using Edge TPU')
            self.interpreter = tf.lite.Interpreter(model_path=self.custom_model_path, experimental_delegates=[delegate])
        
        except ValueError:
            print('Fallback to CPU')
            self.interpreter = tf.lite.Interpreter(model_path=self.custom_model_path)
        
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
 
    def preprocess(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.0
        #im = tf.image.convert_image_dtype(im, tf.float32)
        im = im[:,:, None]
        im = tf.image.resize(im, [100, 100])
        im = tf.expand_dims(im, axis=0)
        return im

    
    def predict(self, image):
        image = self.preprocess(image)
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], image)

        # Run inference
        self.interpreter.invoke()

        # Get output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        batch_prd = output_data
        b = batch_prd[0][1] 
        if b <  0.05: 
            b = 0
        if b >= 0.05: 
            b = 1
        batch_prd[0][1] = b


        angle_all = [0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,  0.5,
                    0.5625, 0.625, 0.6875, 0.75, 0.8125  , 0.875, 0.9375   , 1]
        
        Angle = []
        Prd_angle = list(np.transpose(batch_prd)[0])
        Prd_speed = np.transpose(batch_prd)[1]
        
        for pd_angle in Prd_angle:
            min_ = min(abs(angle_all - pd_angle))
            index = list(abs(angle_all - pd_angle)).index(min_)
            post_angle = angle_all[index]
            Angle.append(post_angle)

        Angle = np.array(Angle)
       
        res = np.concatenate([Angle.reshape(-1, 1), Prd_speed.reshape(-1, 1)], axis = 1)

        print(res)
        angle, speed = res[0][0], res[0][1]

        post_angle = 80*np.clip(angle,0,1)+50
        speed      = 35*np.clip(speed,0,1)
        
        # print(post_angle, speed)

        return post_angle, speed 
        
    

    
    