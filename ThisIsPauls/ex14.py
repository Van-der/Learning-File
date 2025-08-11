import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
iris=load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
df_new=df[['petal length (cm)','petal width (cm)']]
print("df_new.head():\n", df_new.head())
'''plt.scatter(df_new['petal length (cm)'], df_new['petal width (cm)'])
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Dataset - Petal Length vs Width')
plt.show()'''
scaler = MinMaxScaler()
df_new_scaled = scaler.fit_transform(df_new)
plt.scatter(df_new_scaled[:,0], df_new_scaled[:,1]) 
plt.xlabel('Petal Length (scaled)')
plt.ylabel('Petal Width (scaled)')
plt.title('Iris Dataset - Scaled Petal Length vs Width')
#plt.show()
from sklearn.cluster import KMeans
kmeans= KMeans(n_clusters=3,n_init=10)
y_predicted=kmeans.fit_predict(df_new_scaled)
df_new=pd.DataFrame(df_new_scaled, columns=['petal length (cm)', 'petal width (cm)'])

df_new['cluster'] = y_predicted 
print("df_new.head():\n", df_new.head())
df1=df_new[df_new['cluster']==0]
df2=df_new[df_new['cluster']==1]
df3=df_new[df_new['cluster']==2]
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='red', label='Cluster 1')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='blue', label='Cluster 2')
plt.scatter(df3['petal length (cm)'], df3['petal width (cm)'], color='green', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black', marker='x', label='Centroids')
plt.xlabel('Petal Length (scaled)')
plt.ylabel('Petal Width (scaled)')
plt.title('K-Means Clustering on Iris Dataset')
plt.show()
k_rng=range(1, 11)
sse=[]
for k in k_rng:
    km=KMeans(n_clusters=k, n_init=10)
    km.fit(df_new)
    sse.append(km.inertia_)
plt.plot(k_rng, sse)
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal K')
plt.show()