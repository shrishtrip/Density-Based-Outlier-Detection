import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


path = 'C:/Users/khushal/Desktop/DM_ASS_3/dist_matrix1'
df = pd.read_csv(path,delim_whitespace=False,sep=',',header=None)

k =50
print("***************************START**************************")
print("K value =",k)

print("---------------------------------------------------------")

def knn_(point):
    global k
    distances=df.iloc[:,point].tolist()
    y = [[distances[i],i] for i in range(len(distances)) if i!=point]
    y.sort()
    k_dist = y[k-1][0]
    neighbours = [i[1] for i in y if i[0]<=k_dist]
    return k_dist,neighbours


k_dist_dict={}
k_nn_dict={}
for p in range(len(df)):
    x,y = knn_(p)
    k_dist_dict[p] = x
    k_nn_dict[p] = y


def reachability_distance(a,b):
    global k_dist_dict
    return max(k_dist_dict[a],df[a][b])

def local_reachability_density(a):
    global k_nn_dict
    sum = 0
    for nn in k_nn_dict[a]:
        sum+=reachability_distance(nn,a)
    return len(k_nn_dict[a])/sum

def local_outlier_factor(a):
    global k_nn_dict
    
    sum=0
    for nn in k_nn_dict[a]:
        sum+=local_reachability_density(nn)
    sum/=len(k_nn_dict[a])
    
    sum/=local_reachability_density(a)
    
    return sum

outliers,inliers=[],[]

for p in range(len(df)):
    if local_outlier_factor(p) > 1:
        outliers.append(p)
    else:
        inliers.append(p)
    #print(p,local_outlier_factor(p))
        
print("Total number of Inliers = ",len(inliers))
print("Total number of Outliers = ",len(outliers))



df_ = pd.read_csv('german_data-numeric',delim_whitespace=True,header=None)

classes = list(df_.iloc[:,-1])



acc = 0
for i in outliers:
    if classes[i]== 2:
        acc+=1
for i in inliers:
	if classes[i] == 1:
		acc+=1


print("Accuracy =",acc/len(classes)*100)



df = pd.read_csv('german_data-numeric',delim_whitespace=True,header=None)
df = df.astype('float64')
scaler = MinMaxScaler().fit(df.iloc[:,:-1])
df2 = scaler.transform(df.iloc[:,:-1])

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



core_x,core_y=[],[]
for i in inliers:
	core_x.append(df[i][0])
	core_y.append(df[i][1])

noise_x,noise_y=[],[]
for i in outliers:
	noise_x.append(df[i][0])
	noise_y.append(df[i][1])

ax.scatter(core_x,core_y,color='g',s=20)
ax.scatter(noise_x,noise_y,color='r',s=20)


targets = [1.0,2.0]

ax.legend(targets)
ax.grid()
plt.show()
# plt.savefig('k=10.png')
print("***************************END****************************")