import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import vgg16
import cv2
from dd_nnutil_hallym3 import *
from datetime import datetime

#magenta = (255, 0, 255)
yellow = (0, 255, 255)
font = cv2.FONT_ITALIC

def print_prob(prob, file_path):
    synset = [I.strip() for I in open(file_path).readlines()]
    pred = np.argsort(prob)[::-1]
    top1 = synset[pred[0]]
    top5 = [(synset[pred[i]],prob[pred[i]]) for i in range(5)]
    print("Top1 : ", top1, prob[pred[0]])
    print("Top 1~5 : ", top5)
    return top1

vgg = vgg16.Vgg16('vgg16.npy')
images = tf.placeholder("float",[1, 224, 224, 3])

vgg.build(images)


capture = cv2.VideoCapture(0)#카메라띄우기
capture.set(3, 480)
capture.set(4, 320)
sess = tf.InteractiveSession()
cnt = 0
while True:
    ret, img1 = capture.read()
    cv2.imshow('cam',img1)

    if cv2.waitKey(1) & 0xFF == ord('q'):#카메라 캡쳐

        height, width = img1.shape[:2]
        print('{} x {}'.format(width, height))

        # crop
        img1 = img1[:, :, :3]
        print(img1.shape)
        img1c=centered_crop(img1, output_side_length=224)

        #print(img1c.shape)
        img1r = img1c.reshape((1, 224, 224, 3))
        #print(img1r.shape)

        feed_dict = {images:img1r}
        prob = sess.run(vgg.prob, feed_dict=feed_dict)

        predstr = print_prob(prob[0], 'synset.txt')
        str2 = 'No.{}'.format(np.argmax(prob[0]))

        cv2.putText(img1, predstr, (10,50), font, 0.4, yellow, 1, cv2.LINE_AA)
        cv2.putText(img1, str2, (10,70), font, 0.4, yellow, 1, cv2.LINE_AA)

        cv2.imshow('recognized',img1)

        now = datetime.now() # current date and time
        date_time = now.strftime("%Y%m%d_%H%M%S")
        fn1 = './out/{}_{}.png'.format(date_time, cnt)#저장되는 위치 및 이미지 파일 이름
        cv2.imwrite(fn1, img1)
        print(fn1)
        cnt = cnt+1
