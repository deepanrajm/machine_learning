import numpy as np
import sys
import cv2
import imutils
import pickle as cPickle
from skimage import exposure
from skimage import feature
from numpy import asarray

gray = cv2.imread("test.jpg",0)

loaded_model = cPickle.load(open("classifier.cPickle", 'rb'))

(T, threshInv) = cv2.threshold(gray, 80, 255,cv2.THRESH_BINARY_INV)
cv2.imshow("threshInv",threshInv)
cv2.waitKey(0)


for i in range(0, 6):
	dilated = cv2.dilate(threshInv, None, iterations=i + 1)

cv2.imshow("Dilated", dilated)

cnts = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
clone = gray.copy()

for cnt in cnts:
	x,y,w,h = cv2.boundingRect(cnt)
	if (w<120 and w>10 and h<90 and h>40):
		crp = threshInv[y:y+h,x:x+w]
		crp = cv2.resize(crp, (28,28))
		cv2.imshow("crop",crp)
		data_arr = asarray(crp)
		data_arr = data_arr/255
		pred = loaded_model.predict(data_arr.reshape(1, -1))[0]
		cv2.putText(clone, pred, (x-10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(200, 200, 200), 3)
		cv2.rectangle(clone,(x,y),(x+w,y+h),(200,200,200),2)
		cv2.imshow("output", clone)
		cv2.waitKey(0)

cv2.destroyAllWindows()
