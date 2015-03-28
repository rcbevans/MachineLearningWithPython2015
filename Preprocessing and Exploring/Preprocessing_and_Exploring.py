import csv
import numpy
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Load the data

fileName = '..\wdbc.csv'
fileOpen = open(fileName, "rU")

csvData = csv.reader(fileOpen)
dataList = list(csvData)

# Inspect the data

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

# Analyze the data

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

# Encode the data

le = preprocessing.LabelEncoder()
le.fit(y)
yTransformed = le.transform(y)

print "Original Labels:\n", y[:10]
print "Encoded Labels:\n", yTransformed[:10]

# Inspect for patterns - Heatmap of relationships

correlationMatrix = numpy.corrcoef(X, rowvar = 0)

fig, ax = plt.subplots()
heatmap = ax.pcolor(correlationMatrix, cmap = plt.cm.Blues)
plt.show()

# Simple scatter plot

plt.scatter(x = X[:, 0], y = X[:, 1], c = y)
plt.show()

# Matrix of scatter plots!

def scatter_plot(X,y):
    plt.figure(figsize = (2*X.shape[1],2*X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            plt.subplot(X.shape[1],X.shape[1],i+1+j*X.shape[1])
            if i == j:
                plt.hist(X[:, i][y=="M"], alpha = 0.4, color= 'm', bins = numpy.linspace(min(X[:, i]), max(X[:, i]), 30))
                plt.hist(X[:, i][y=="B"], alpha = 0.4, color = 'b', bins = numpy.linspace(min(X[:, i]), max(X[:,i]), 30))
                plt.xlabel(i)
            else:
                plt.gca().scatter(X[:, i], X[:, j], c = y, alpha = 0.4)
                plt.xlabel(i)
                plt.ylabel(j)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
    plt.tight_layout()
    plt.show()

scatter_plot(X[:,:5],y)