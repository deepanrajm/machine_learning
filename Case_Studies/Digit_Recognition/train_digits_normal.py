from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from skimage import exposure
from skimage import feature
from imutils import paths
import imutils
import cv2
import pickle as cPickle
from sklearn.linear_model import LogisticRegression
from PIL import Image as im
from numpy import asarray
import numpy as np


print("[INFO] extracting features...")
data = []
labels = []

for imagePath in paths.list_images("dataset\\training"):
	digit = imagePath.split("\\")[-2]
	x_data = np.array(cv2.imread(imagePath,0))
	x_data = x_data/255

	data.append(x_data)
	labels.append(digit)


data_arr = asarray(data)
pixels = data_arr.flatten().reshape(60000, 784)

print("[INFO] training classifier...")

model = RandomForestClassifier(n_estimators=30, random_state=42)
model.fit(pixels, labels)
print("[INFO] evaluating...")

f = open("classifier.cPickle", "wb")
f.write(cPickle.dumps(model))
f.close()


act_res = []
pred_res = []

for imagePath in paths.list_images("dataset\\testing"):

	digit_val = imagePath.split("\\")[-2]
	x_data = np.array(cv2.imread(imagePath,0))
	x_data = x_data/255
	pred = model.predict(x_data.reshape(1, -1))[0]
	act_res.append(digit_val)
	pred_res.append(pred)


print(classification_report(act_res, pred_res))
