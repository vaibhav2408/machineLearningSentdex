import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


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
	vote_result = Counter(votes).most_common(1)[0][0]
	#print(vote_result)
	confidence = Counter(votes).most_common(1)[0][1] / k

	return vote_result, confidence
Accuracies = []
for i in range(25):
	df = pd.read_csv('breast-cancer-wisconsin.data.txt')
	df.replace('?', -99999, inplace = True)
	df.drop(['id'], 1, inplace = True)
	full_data = df.astype(float).values.tolist()

	random.shuffle(full_data)
	test_size = 0.4
	train_set = {2:[], 4:[]}

	test_set = {2:[], 4:[]}

	train_data = full_data[: -int(test_size*len(full_data))]
	test_data = full_data[-int(test_size*len(full_data)):]



	#'i in train_data' is each internal list inside train_data.
	#i[-1] is 2.0 or 4.0 (2-benign, 4-malignant). 
	#if i[-1] is 2 or 4 it will select 2:[] or 4:[] inside train_set dictionary and 
	#append the internal list for that key, except the last item i.e. 2.0 or 4.0 (since i[:-1]).


	for i in train_data :
		train_set[i[-1]].append(i[:-1])

	for i in test_data :
		test_set[i[-1]].append(i[:-1])	



	correct = 0
	total = 0


	# group here is 2 or 4. So we're checking that if we pass 
	# group whose label is '2' ,are we getting 2 as result, 
	# if yes, then increment correct value. Same logic appliaes when
	# group' label is 4


	for group in test_set:
		for data in test_set[group]:
			vote, confidence = k_nearest_neighbors(train_set, data, k=5)
			if group == vote:
				correct+=1	
			total+=1

	#print('Accuracy : ', correct/total)		
	Accuracies.append(correct/total)
print(sum(Accuracies)/len(Accuracies))		

