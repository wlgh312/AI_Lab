import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime

import keras
from keras.applications import imagenet_utils, mobilenet
from keras.models import Model
from keras.preprocessing import image


#magenta = (255, 0, 255)
yellow = (0, 255, 255)
font = cv2.FONT_ITALIC #FONT_HERSHEY_SCRIPT_SIMPLEX  # hand-writing style font

def process_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    plmg = mobilenet.preprocess_input(img_array)
    return plmg

capture = cv2.VideoCapture(0)#카메라띄우기
capture.set(3, 224)
capture.set(4, 224)
sess = tf.InteractiveSession()
cnt = 0
while True:
    ret, img1 = capture.read()#카메라 캡쳐
    cv2.imshow('cam',img1)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        height, width = img1.shape[:2]
        print('{} x {}'.format(width, height))

        cv2.imshow('captured',img1)
        now = datetime.now() # current date and time
        date_time = now.strftime("%Y%m%d_%H%M%S")
        fn1 = './out/{}_{}.png'.format(date_time, cnt)
        filename = '{}_{}.png'.format(date_time,cnt)
        cv2.imwrite(fn1, img1)

        test_img_path = fn1
        plmg = process_image(test_img_path)
        mobilenet = mobilenet.MobileNet()
        prediction = mobilenet.predict(plmg)
        results = imagenet_utils.decode_predictions(prediction)
        print(results)

        cv2.putText(img1, "{}".format(results[0][0]),(10, 70), font, 0.4, yellow, 1, cv2.LINE_AA)
        cv2.imshow('recognized',img1)
        fn2 = './result/{}_{}.png'.format(date_time, cnt)
        cv2.imwrite(fn2, img1)
        cnt = cnt+1






