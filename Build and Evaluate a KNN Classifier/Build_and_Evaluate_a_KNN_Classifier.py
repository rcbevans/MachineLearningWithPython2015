import csv
import numpy
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import knnplots

fileName = '..\wdbc.csv'
fileOpen = open(fileName, "rU")

csvData = csv.reader(fileOpen)
dataList = list(csvData)

# Inspect the data
dataArray = numpy.array(dataList)

X = dataArray[:,2:32].astype(float)
y = dataArray[:,1]
yTransformed = preprocessing.LabelEncoder().fit(y).transform(y)

nbrs = neighbors.NearestNeighbors(n_neighbors = 3, algorithm = "ball_tree").fit(X)
distance, indices = nbrs.kneighbors(X)

knnK3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
knnK3 = knnK3.fit(X, yTransformed)
predictedK3 = knnK3.predict(X)

knnK15 = neighbors.KNeighborsClassifier(n_neighbors = 15)
knnK15 = knnK15.fit(X, yTransformed)
predictedK15 = knnK15.predict(X)

nonAgreement = predictedK3[predictedK3 != predictedK15]
print "Number of discrepancies ", len(nonAgreement)

knnWD = neighbors.KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
knnWD = knnWD.fit(X, yTransformed)
predictedWD = knnWD.predict(X)

knnWU = neighbors.KNeighborsClassifier(n_neighbors = 3, weights = 'uniform')
knnWU = knnWU.fit(X, yTransformed)
predictedWU = knnWU.predict(X)

nonAgreement = predictedWD[predictedWD != predictedWU]
print "Number of discrepancies between distance and uniform weights ", len(nonAgreement)

XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed)

knn = neighbors.KNeighborsClassifier(n_neighbors = 3)
knn = knn.fit(XTrain, yTrain)
predicted = knn.predict(XTest)

mat = confusion_matrix(yTest, predicted)

print "Confusion matrix:"
print mat

print "Accuracy: ", accuracy_score(yTest, predicted)
print "Report: ", classification_report(yTest, predicted)

knnplots.plotaccuracy(XTrain, yTrain, XTest, yTest, 310)
knnplots.decisionplot(XTrain, yTrain, 15, 'distance')