import cv2
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
#from cv2 import CascadeClassifier
from cv2 import rectangle

classifier  = cv2.CascadeClassifier("/Users/wlgh3/repos/AI_Lab/CamLive_VGG16/haarcascade_frontalface.xml")

pixels = imread('/Users/wlgh3/repos/AI_Lab/CamLive_VGG16/face.jpeg')
bboxes = classifier.detectMultiScale(pixels, 1.05, 8)

for box in bboxes:
  x, y, width, height = box
  x2, y2 = x+width, y+height
  rectangle(pixels, (x,y), (x2, y2), (0,0,255), 1)

cv2.imshow('face detection', pixels)

waitKey(0)

destroyAllWindows()
