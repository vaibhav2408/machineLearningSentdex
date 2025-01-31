import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

accuracies = []

for i in range(25):
	df = pd.read_csv('breast-cancer-wisconsin.data.txt')
	df.replace('?', -99999, inplace=True)
	df.drop(['id'], 1, inplace=True)

	y = np.array(df['class'])
	X = np.array(df.drop(['class'],1))
	
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
	clf = neighbors.KNeighborsClassifier(n_jobs = -1)
	clf.fit(X_train, y_train)

	accuracy = clf.score(X_test, y_test)
	accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))	