##Usage  python fire_detect.py

import cv2
import numpy as np
import sys
from imutils import paths 

image_paths = "fire"

for path in sorted(list(paths.list_images(image_paths))):
	frame = cv2.imread(path)

	blur = cv2.GaussianBlur(frame, (21, 21), 0)
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
 
	lower = [18, 50, 50]
	upper = [35, 255, 255]
	lower = np.array(lower, dtype="uint8")
	upper = np.array(upper, dtype="uint8")
	mask = cv2.inRange(hsv, lower, upper)
	cv2.imshow("mask",mask)
	cv2.imshow("input",frame)
	output = cv2.bitwise_and(frame, hsv, mask=mask) 
	(cnts,hierarchy,_) = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	if hierarchy:

		for cnt in hierarchy:
			x,y,w,h = cv2.boundingRect(cnt)
			print (x,y,w,h)
			cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
		#print ("fire_detected")
			cv2.putText(output,'Fire Detected!!',(20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75 ,(0,0,255),2,cv2.LINE_AA)
			cv2.imshow("output", output)
			cv2.waitKey(0)
	else:
		print ("Fire Not Detected")
		cv2.putText(output,'Fire Not Detected!!',(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.75 ,(0,0,255),2,cv2.LINE_AA)
		cv2.imshow("output", output)
		cv2.waitKey(0)


