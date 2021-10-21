import numpy as np
import matplotlib.pyplot as plot
from matplotlib.pyplot import style
style.use("seaborn-darkgrid")
from sklearn.cluster import KMeans

data = np.array([[2, 10], [2,5], [8, 4] ,[8,5], [7,5], [6,4], [1,2], [4,9]])

km = KMeans(n_clusters=3)
km.fit(data)

centroids = km.cluster_centers_
label = km.labels_
print ("Cluster Centers:", centroids)
print ("Labels :", label)

colours = ['b.','g.','r.']
for i in range(len(data)):
    #print "Coordinates ", a[i], "labels", label[i]
    plot.plot(data[i][0], data[i][1], colours[label[i]], markersize = 10 )
for i in range(len(centroids)):
    plot.plot(centroids[i][0], centroids[i][1], colours[i], markersize = 15)
plot.show()