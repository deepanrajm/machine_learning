import numpy as np
import sys
import cv2
import imutils
import pickle as cPickle
from skimage import exposure
from skimage import feature
from numpy import asarray

gray = cv2.imread("1.png",0)

loaded_model = cPickle.load(open("classifier1.cPickle", 'rb'))

(T, threshInv) = cv2.threshold(gray, 80, 255,cv2.THRESH_BINARY_INV)
cv2.imshow("threshInv",threshInv)
cv2.waitKey(0)

for i in range(0, 6):
	dilated = cv2.dilate(threshInv.copy(), None, iterations=i + 1)

cv2.imshow("Dilated", dilated)

cnts = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
clone = gray.copy()

for cnt in cnts:
	x,y,w,h = cv2.boundingRect(cnt)
	if (w<150 and w>10 and h<150 and h>40):
		crp = threshInv[y:y+h,x:x+w]
		crp = cv2.resize(crp, (28,28))
		cv2.imshow("cropped",crp)

		(H, hogImage) = feature.hog(crp, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, visualize=True)
		pred = loaded_model.predict(H.reshape(1, -1))[0]
		cv2.putText(clone, pred, (x-10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(200, 200, 200), 3)
		cv2.rectangle(clone,(x,y),(x+w,y+h),(200,200,200),2)
		cv2.imshow("output", clone)
		cv2.waitKey(0)

cv2.destroyAllWindows()
