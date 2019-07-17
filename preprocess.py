import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

df = pd.read_csv('german_data-numeric',delim_whitespace=True,header=None)
df = df.astype('float64')
#Normalizing the data
scaler = MinMaxScaler().fit(df.iloc[:,:-1])
df2 = scaler.transform(df.iloc[:,:-1])
#Target classes
classes = df.iloc[:,-1]
#Conversion of Numpy Array to Dataframe
new_df = pd.DataFrame(df2)
# print(new_df)


def mahalanobis(a,b):
	return scipy.spatial.distance.euclidean(a,b)
	
n = len(df2)
dist_mat = np.zeros((n,n))
for i in range(n):
	for j in range(n):
		if i==j:
			dist_mat[i][i] = 0	
		if i>j:
			dist_mat[i][j] = mahalanobis(df2[i],df2[j])
			dist_mat[j][i] = dist_mat[i][j]

print(dist_mat)



f = open('last_mat','w+')
for r in dist_mat:
	t = ''
	for s in r:
		t += str("%.6f"%(s)) + ','
	f.write(t[:-1]+'\n')

f.close()

