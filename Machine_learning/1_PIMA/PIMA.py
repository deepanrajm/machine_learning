# Create your first MLP in Keras

import numpy as np
import sklearn
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# handle older versions of sklearn
if int((sklearn.__version__).split(".")[1]) < 18:
	from sklearn.cross_validation import train_test_split

# otherwise we're using at lease version 0.18
else:
	from sklearn.model_selection import train_test_split


dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (data) and output (labels) variables

data = dataset[:,0:8]
labels = dataset[:,8]

# construct the training and testing split by taking 75% of the data for training
# and 25% for testing

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data),
	np.array(labels), test_size=0.25, random_state=42)

# initialize the model as a decision tree
#splitter = best, random , max_features = int, auto, log, none
#model = DecisionTreeClassifier(random_state=84,splitter='random', max_features=8)

# Random Forest
#model = RandomForestClassifier(n_estimators=10, random_state=42,max_features="auto")

#KNN
#weights = uniform, weights
#model = KNeighborsClassifier(n_neighbors=9)

#SVM
#kernel='rbf', linear, poly, C=1,gamma=0
#model = SVC(kernel="rbf",C=100)

#Logistic Regression
#penality = l1,l2, elasticnet
#solver = liblinear, sag, saga, lbfgs, newton-cg
#max-iter = 100
model = LogisticRegression(max_iter=15)#,solver = 'sag')
#penalty='l1',solver="saga",max_iter=100

# train the decision tree
print("[INFO] training model...")
model.fit(trainData, trainLabels)

# evaluate the classifier
print("[INFO] evaluating...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))