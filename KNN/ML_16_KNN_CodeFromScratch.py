import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
style.use('ggplot')

#euclidean_distance = sqrt( ((plot1[0] - plot2[0])**2) + ((plot1[1] - plot2[1])**2) )

dataset = {'k': [[1,2], [2,3], [3,1]], 'r' : [[6,5], [7,7], [8,6]]}

new_features1 = [5,7]

new_features2 = [1,1]


def k_nearest_neighbors(data, predict, k=3):
	if(len(data) >= k):
		warnings.warn('K is lesser than total data group')

	distances = []

	for group in data:
		for features in data[group]:
			#euclidean_distance = np.sqrt( ((features[0] - predict[0])**2) + ((features[1] - predict[1])**2) )
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict) )
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	#print(Counter(votes).most_common(1))
	vote_results = Counter(votes).most_common(1)[0][0]

	return vote_results		

result = k_nearest_neighbors(dataset, new_features1, k=3)

print(result)	

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features1[0], new_features1[1], s=200, color = result)
plt.scatter(new_features2[0], new_features2[1], s=200, color = result)
result = k_nearest_neighbors(dataset, new_features2, k=3)

print(result)	

plt.scatter(new_features2[0], new_features2[1], s=200, color = result)
plt.show()