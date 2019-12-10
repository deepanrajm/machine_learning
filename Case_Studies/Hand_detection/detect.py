# USAGE
# python detect.py --bounding-box "10,350,225,590"

# import the necessary packages
from tool.gesture_recognition import MotionDetector
import numpy as np
import imutils
import cv2
import os

folder = 'a'
try : 
	os.mkdir(folder)
except FileExistsError:
	pass 

Alphabet = folder
camera = cv2.VideoCapture(0)


ROI = "10,350,225,590"

# unpack the hand ROI, then initialize the motion detector and the total number of
# frames read thus far
(top, right, bot, left) = np.int32(ROI.split(","))
md = MotionDetector()
numFrames = 0
k = 0
# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()


	# resize the frame and flip it so the frame is no longer a mirror view
	frame = imutils.resize(frame, width=600)
	frame = cv2.flip(frame, 1)
	clone = frame.copy()
	(frameH, frameW) = frame.shape[:2]

	# extract the ROI, passing in right:left since the image is mirrored, then
	# blur it slightly
	roi = frame[top:bot, right:left]
	hand = roi.copy()
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)

	# if we not reached 32 initial frames, then calibrate the skin detector
	if numFrames < 32:
		md.update(gray)

	# otherwise, detect skin in the ROI
	else:
		# detect motion (i.e., skin) in the image
		skin = md.detect(gray)


		# check to see if skin has been detected
		if skin is not None:
			# unpack the tuple and draw the contours on the image
			(thresh, c) = skin
			masked = cv2.bitwise_and(hand, hand, mask=thresh)
			cv2.imshow("Mask", masked)
			name = folder +"\\" + Alphabet + str(k) + ".jpg"
			cv2.imwrite(name,masked)
			cv2.drawContours(clone, [c + (right, top)], -1, (0, 255, 0), 2)
			cv2.imshow("Thresh", thresh)
			k = k+1


	# draw the hand ROI and increment the number of processed frames
	cv2.rectangle(clone, (left, top), (right, bot), (0, 0, 255), 2)
	numFrames += 1
	if numFrames >= 30:
		if fl ==1:
			print ("Calibration Completed")
			fl=0
	else :
		print (numFrames)
		fl = 1

	# show the frame to our screen
	cv2.imshow("Frame", clone)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()