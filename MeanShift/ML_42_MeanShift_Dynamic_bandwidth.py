'''

In this machine learning tutorial, we cover the idea of a 
dynamically weighted bandwidth with our Mean Shift 
clustering algorithm

'''
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random


X, y = make_blobs(n_samples=50, centers=3, n_features=2)


# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8 ],
#               [8, 8],
#               [1, 0.6],
#               [9,11],
#               [8,2],
#               [10,2],
#               [9,3]])
colors = 10*["g","r","b","c","y","k"]



class Mean_Shift:
	def __init__(self, radius=None, radius_norm_step = 100):
		self.radius = radius
		self.radius_norm_step = radius_norm_step


	def fit(self, data):
		if self.radius == None:
			all_data_centroids = np.average(data, axis=0)
			
			## This gives the magnitude from the origin
			all_data_norm = np.linalg.norm(all_data_centroids)
			self.radius = all_data_norm/self.radius_norm_step



		centroids = {}

		for i in range(len(data)):
			centroids[i] = data[i]

		## [::-1] reverses the order of the list
		weights = [i for i in range(self.radius_norm_step)][::-1]
	
		while True:
			new_centroids = []
			for i in centroids:
				in_bandwidth = []
				centroid = centroids[i]

				for featureset in data:

					distance = np.linalg.norm(featureset - centroid)
					
					# this is for when the feature set is comparing distacne to itself
					if distance == 0:
						distance = 0.0000001

					weight_index = int(distance/self.radius)
					if weight_index > self.radius_norm_step-1:
						weight_index = self.radius_norm_step-1

					to_add = (weights[weight_index]**2)*[featureset]
					in_bandwidth += to_add



				new_centroid = np.average(in_bandwidth, axis = 0)

				### we're converting it to tuple so we can
				### get "set" of the new_centroids
				new_centroids.append(tuple(new_centroid))



			uniques = sorted(list(set(new_centroids)))

			to_pop = []

			for i in uniques:
				for ii in uniques:
					if i==ii:
						pass

					elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius and ii not in to_pop:
						to_pop.append(ii)
						break
			for i in to_pop:
				try:
					uniques.remove(i)
				except:
					pass


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

		self.classifications = {}

		for i in range(len(self.centroids)):
			self.classifications[i] = []

		for featureset in data:
			distances = [np.linalg.norm(featureset-centroid) for centroid in self.centroids]

			classification = distances.index(min(distances))
			self.classifications[classification].append(featureset)

	def predict(self, data) :
		distance =  [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
		classification = (distance.index(min(distance)))
		return classification	

clf = Mean_Shift()
clf.fit(X)

# cluster = clf.predict([6,6])
# print(cluster)


centroids = clf.centroids

for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker='x', color= color, s=150, linewidths = 5)
					
for c in centroids:
	plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)
plt.show()	














