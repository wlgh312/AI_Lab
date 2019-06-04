import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import vgg16
import utils
import cv2
from skimage import io
import matplotlib.pyplot as plt


vgg = vgg16.Vgg16('vgg16.npy')

capture = cv2.VideoCapture(0)#카메라띄우기
capture.set(3, 480)
capture.set(4, 320)

while True:
    ret, frame = capture.read()#카메라 캡쳐
    cv2.imshow('image',frame)
    cv2.imwrite('test.jpg', frame)#이미지저장
capture.release()#카메라 해제

img = "test.jpg"

i1 = io.imread(img)

plt.imshow(i1)
plt.show()

img1 = utils.load_image(img)

img1 = img1[:, :, :3]

print(img1.shape)

img1r = img1.reshape((1, 224, 224, 3))

print(img1r.shape)

images = tf.placeholder("float",[1, 224, 224, 3])

vgg.build(images)

sess = tf.InteractiveSession()

feed_dict = {images:img1r}
prob = sess.run(vgg.prob, feed_dict=feed_dict)

print(prob[0])

print(np.argmax(prob[0]))

def print_prob(prob, file_path):
    synset = [I.strip() for I in open(file_path).readlines()]
    pred = np.argsort(prob)[::-1]
    top1 = synset[pred[0]]
    print("Top1 : ", top1, prob[pred[0]])
    return top1

get_ipython().system('type synset.txt')

top1 = print_prob(prob[0], 'synset.txt')

