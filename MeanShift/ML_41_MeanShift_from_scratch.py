import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np



X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3]])
colors = 10*["g","r","b","c","y","k"]

# plt.scatter(X[:,0], X[:,1], s= 150)
# plt.show()

class Mean_Shift:
	def __init__(self, radius=4):
		self.radius = radius

	def fit(self, data):
		centroids = {}

		for i in range(len(data)):
			centroids[i] = data[i]

		while True:
			new_centroids = []
			for i in centroids:
				in_bandwidth = []
				centroid = centroids[i]

				for featureset in data:
					if np.linalg.norm(featureset - centroid) < self.radius:
						in_bandwidth.append(featureset)

				new_centroid = np.average(in_bandwidth, axis = 0)

				### we're converting it to tuple so we can
				### get "set" of the new_centroids
				new_centroids.append(tuple(new_centroid))		

			uniques = sorted(list(set(new_centroids)))

			## using this approach makes sure that
			## if we modify centroids, the prev_centroids 
			## don't get modified
			prev_centroids = dict(centroids)

			centroids = {}

			for i in range(len(uniques)):
				centroids[i] = np.array(uniques[i])

			optimized = True

			for i in centroids:
				if not np.array_equal(centroids[i], prev_centroids[i]):
					optimized = False
				if not optimized:
					break
			if optimized:
				break
		self.centroids = centroids

	def predict(self, data) :
		distance =  [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
		classification = (distance.index(min(distance)))
		return classification	

clf = Mean_Shift()
clf.fit(X)

cluster = clf.predict([6,6])
print(cluster)

#plt.scatter(6,6, s= 150, color=colors[cluster], marker='+')

centroids = clf.centroids
plt.scatter(X[:,0], X[:,1], s=150)						
for c in centroids:
	plt.scatter(centroids[c][0], centroids[c][1], color=colors[c], marker='*', s=150)
plt.show()	

















