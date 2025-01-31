import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')

#ORIGINAL:

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [1,3],
              [8,9],
              [0,3],
              [5,4],
              [6,4]])


#plt.scatter(X[:, 0],X[:, 1], s=150, linewidths = 5, zorder = 10)
#plt.show()

colors = 10*["g","r","c","b","k"]

class K_Means:
	def __init__(self, k=2, tol=0.001, max_iter=300):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):
		self.centroids = {}

		for i in range(self.k):
			self.centroids[i] = data[i]

		# classification is dict having key: centroids and 
		# values having feature set contained within those centroids	

		for i in range(self.max_iter):
			self.classifications = {}


			for i in range(self.k):
				self.classifications[i]=[]


			for featureset in data:
				distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)


			# if we do prev_centroids = self.centroids, 
			# prev_centroids will change as self.centroids changes which we don't want
			# Hence we're using the dictionary approach
			prev_centroids = dict(self.centroids)


			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)
			optimizaed = True


			for c in self.centroids:
				original_centroids = prev_centroids[c]
				currect_centroids = self.centroids[c]
				if np.sum((currect_centroids - original_centroids)/original_centroids*100.0) > self.tol:
					print((currect_centroids - original_centroids)/original_centroids*100.0)
					optimizaed=False


			if optimizaed:
				break		
			

	def predict(self, data):
		distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification


clf = K_Means()
clf.fit(X)


for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], 
		marker="o", color="k", s=150, linewidths=5)


for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker="x", color = color, s = 150, linewidths=5)

#unknowns = np.array([[1,3],[8,9],[0,3],[5,4],[6,4],])

#for unknown in unknowns:
#	classification = clf.predict(unknown)
#	plt.scatter(unknown[0], unknown[1], marker="*", color = colors[classification], s=150, linewidths=5)

plt.show()






