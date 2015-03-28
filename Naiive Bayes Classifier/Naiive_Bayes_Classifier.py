import csv
import numpy
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split, cross_val_score
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

XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed)

nbmodel = GaussianNB().fit(XTrain, yTrain)
predicted = nbmodel.predict(XTest)

mat = confusion_matrix(yTest, predicted)
print mat

print classification_report(yTest, predicted)
print "accuracy: ", accuracy_score(yTest, predicted)