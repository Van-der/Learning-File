import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
iris=load_iris()
print("dir(iris):", dir(iris))  
df=pd.DataFrame(iris.data,columns=iris.feature_names) 
df['target']=iris.target
df['target_name']=iris.target_names[df['target']]
print("df.head():\n", df.head())
x=df.drop(['target', 'target_name'], axis='columns')
y=df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17)
rf=RandomForestClassifier(n_estimators=40)
rf.fit(x_train, y_train)
print("Score is:", rf.score(x_test, y_test))
plt.figure(figsize=(10, 6))
cm= confusion_matrix(y_test, rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt='d',cmap='coolwarm', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
