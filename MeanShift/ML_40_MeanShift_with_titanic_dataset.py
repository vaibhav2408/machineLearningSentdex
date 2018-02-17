import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import MeanShift
import pandas as pd

df = pd.read_excel('titanic.xls')

original_df = pd.DataFrame.copy(df) #if we do original_df = df, 
									# as we modify df, it'll also modify original df

df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

#print(df.head())

def handle_non_numerical_data(df):
	columns = df.columns.values

	for column in columns:
		text_digit_vals={}
		def convert_to_int(val):
			return text_digit_vals[val]

		if df[column].dtype != np.int64 and df[column].drop != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x+=1

			#  map(convert_to_int, df[column])  is nothing but
			#  [convert_to_int(x) for x in df[column]]
			df[column] = list(map(convert_to_int, df[column]))

	return df

df = handle_non_numerical_data(df)

print(df.head())
df.drop(['boat', 'sex'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

# adding one more column to original_df
original_df['cluster_groups'] = np.nan

for i in range(len(X)):
	# iloc[i] is referncing the i row in the data frame
	# original_df['cluster_groups'].iloc[i] is referencing the 
	# cluster_groups column in ith row
	# and assigning it the value of label[i]
	original_df['cluster_groups'].iloc[i] = labels[i]

# number of clusters is the number of unique labels 
n_clusters = len(np.unique(labels))

survival_rates = {}

for i in range(n_clusters):
	temp_df = original_df[	(original_df['cluster_groups'] == float(i))	]

	survival_cluster = temp_df[	 (temp_df['survived']==1)	]
	survival_rate = len(survival_cluster)/ len(temp_df)
	survival_rates[i] = survival_rate

print(survival_rates)	











