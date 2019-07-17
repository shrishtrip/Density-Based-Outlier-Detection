import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
df = pd.read_csv('dist_matrix1',delim_whitespace=False,sep=',',header=None)

print("***************************START**************************")


eps = 2
min_pts = 3
print("EPS =",eps)
print("Min_points=",min_pts)
print("-----------------------------------------------------------")
classes = []
core_points = []
noise = []
n = len(df)

for i in range(n):
	min_p = 0
	for j in range(n):
		if i!=j:
			if df[i][j] >= eps:
				min_p+=1
	if min_p>=min_pts:
		core_points.append(i)
		classes.append(1)
	else:
		noise.append(i)
		classes.append(2)


print("Total Core points = ",len(core_points))

print("Total Noise points = ",len(noise))

df_ = pd.read_csv('german_data-numeric',delim_whitespace=True,header=None)

classes = list(df_.iloc[:,-1])
acc = 0
for i in noise:
    if classes[i]== 2:
        acc+=1
for i in core_points:
	if classes[i] == 1:
		acc+=1


print("Accuracy =",acc/len(classes)*100)


df = pd.read_csv('german_data-numeric',delim_whitespace=True,header=None)
df = df.astype('float64')
scaler = MinMaxScaler().fit(df.iloc[:,:-1])
df2 = scaler.transform(df.iloc[:,:-1])
classes = pd.DataFrame(classes)
new_df = pd.DataFrame(df2)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(new_df)

df = principalComponents
p_df = pd.DataFrame(principalComponents)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)


# ax.scatter()
core_x,core_y=[],[]
for i in core_points:
	core_x.append(df[i][0])
	core_y.append(df[i][1])

noise_x,noise_y=[],[]
for i in noise:
	noise_x.append(df[i][0])
	noise_y.append(df[i][1])

ax.scatter(core_x,core_y,color='g',s=20)
ax.scatter(noise_x,noise_y,color='r',s=20)


targets = [1.0,2.0]
ax.legend(targets)
ax.grid()
plt.show()
# plt.savefig('eps=3min_points=40.png')
print("***************************END****************************")