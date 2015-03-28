import csv
import numpy
import scipy.stats as stats
import matplotlib.pyplot as plt

fileName = '..\wdbc.csv'
fileOpen = open(fileName, "rU")

csvData = csv.reader(fileOpen)
dataList = list(csvData)

print "List:"
print dataList[0:2]

dataArray = numpy.array(dataList)

print "Numpy Array:"
print dataArray[0:2]

X = dataArray[:,2:32]
print "Original Type:"
print X[0,0], type(X[0,0])
print
X = dataArray[:,2:32].astype(float)
print "Converted to Float:"
print X[0,0], type(X[0,0])
print
y = dataArray[:,1]
print "X dimension: ", X.shape
print "y dimension: ", y.shape

print "First sample:"
print X[1,:]
print "First sample type:"
print y[0]

yFreq = stats.itemfreq(y)
print yFreq

plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.bar(left = 0, height = int(yFreq[0][1]))
plt.bar(left = 1, height = int(yFreq[1][1]))
plt.subplot(1,2,2)
plt.bar(left = 0, height = int(yFreq[0][1]), color = 'r')
plt.bar(left = 1, height = int(yFreq[1][1]), color = 'b')
plt.xticks([])
plt.legend(['B', 'M'])
plt.show()