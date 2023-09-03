# USAGE
# python detect_faces_video.py --face cascades/haarcascade_frontalface_default.xml --video video.mp4

# import the necessary packages
import argparse
import imutils
import cv2

i = 0 
inp = ""


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="Path to where the face cascade resides")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector
detector = cv2.CascadeClassifier(args["face"])


# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame, then we have
	# reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, convert it to grayscale, and detect faces in the
	# frame
	frame = imutils.resize(frame, width=400)
	frame_o = frame.copy()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# handle face detection for OpenCV 2.4
	if imutils.is_cv2():
		faceRects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5,
			minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

	# otherwise handle face detection for OpenCV 3+
	else:
		faceRects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

	# loop over the faces and draw a rectangle around each
	for (x, y, w, h) in faceRects:
		cv2.rectangle(frame_o, (x, y), (x + w, y + h), (255,0, 0), 10)
		gra = frame[y:y+h,x:x+h]
		inp = str(i) + ".jpg"
		cv2.imwrite(inp,gra)
		i = i+1

	# show the frame to our screen
	cv2.imshow("Frame", frame_o)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()