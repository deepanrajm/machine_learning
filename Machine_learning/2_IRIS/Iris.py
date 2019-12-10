# Multiclass Classification with the Iris Flowers Dataset
import numpy as np
from pandas import read_csv
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


# load dataset
dataframe = read_csv("iris.csv")
dataset = dataframe.values
data = dataset[:,0:4].astype(float)
labels = dataset[:,4]
# construct the training and testing split by taking 75% of the data for training
# and 25% for testing

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data),
	np.array(labels), test_size=0.25, random_state=42)



# initialize the model as a decision tree
model = DecisionTreeClassifier(random_state=84)

# Random Forest
#model = RandomForestClassifier(n_estimators=20, random_state=42)

# Knn
#model = KNeighborsClassifier(n_neighbors=3)

#SVM
#model = SVC(kernel="linear")

#Logistic Regression
#model = LogisticRegression()

# train the decision tree
print("[INFO] training model...")
model.fit(trainData, trainLabels)

# evaluate the classifier
print("[INFO] evaluating...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))