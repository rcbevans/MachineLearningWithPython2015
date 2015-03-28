
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

# Simple 2D

#X = iris.data
#labels = iris.target_names
##Symbols to represent the points for the three classes on the graph.
#gMarkers = ["+", "_", "x"]
##Colours to represent the points for the three classes on the graph
#gColors = ["blue", "magenta", "cyan"]
##The index of the class in target_names
#gIndices = [0, 1, 2]
##Column indices for the two features you want to plot against each other:
#f1 = 0
#f2 = 1

#for mark, col, i, iris.target_name in zip(gMarkers, gColors, gIndices, labels):
#   plt.scatter(x = X[iris.target == i, f1], y = X[iris.target == i, f2], marker = mark, c = col, label=iris.target_name)
#plt.legend(loc='upper right')
#plt.xlabel(iris.feature_names[f1])
#plt.ylabel(iris.feature_names[f2])
#plt.show()

# Multiplot 2D

#X = iris.data
#labels = iris.target_names
##Symbols to represent the points for the three classes on the graph.
#gMarkers = ["+", "_", "x"]
##Colours to represent the points for the three classes on the graph
#gColors = ["blue", "magenta", "cyan"]
##The index of the class in target_names
#gIndices = [0, 1, 2]
##Column indices for the two features you want to plot against each other:
#f1 = 0
#f2 = 1

#nrow, ncol = iris.data.shape

#for j in range(ncol):
#    for k in range(nrow):
#        plt.subplot(ncol, ncol, j+1+k*ncol)
#        for mark, col, i, iris.target_name in zip(gMarkers, gColors, gIndices, labels):
#            plt.scatter(x = X[iris.target == i, j],
#                        y = X[iris.target == i, k],
#                        marker = mark,
#                        c = col,
#                        label = iris.target_name)
#            plt.xlabel(iris.feature_names[j])
#            plt.ylabel(iris.feature_names[k])
#plt.tight_layout()
#plt.show()

# 3D plot

from mpl_toolkits.mplot3d import Axes3D

X = iris.data
labels = iris.target_names
#Symbols to represent the points for the three classes on the graph.
gMarkers = ["+", "_", "x"]
#Colours to represent the points for the three classes on the graph
gColors = ["blue", "magenta", "cyan"]
#The index of the class in target_names
gIndices = [0, 1, 2]
#Column indices for the two features you want to plot against each other:
f1 = 0
f2 = 1
f3 = 2

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-140, azim=190)

for mark, col, i, iris.target_name in zip(gMarkers, gColors, gIndices, labels):
    ax.scatter( X[iris.target == i, f1], X[iris.target == i, f2], X[iris.target == f3], c=col)
ax.set_xlabel(iris.feature_names[f1])
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel(iris.feature_names[f2])
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel(iris.feature_names[f3])
ax.w_zaxis.set_ticklabels([])
plt.show()