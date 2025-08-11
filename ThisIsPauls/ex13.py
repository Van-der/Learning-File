import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sl=StandardScaler()
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
iris =load_iris()
print("dir(iris):", dir(iris))
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
print("df.head():\n", df.head())
x = df.drop('target', axis='columns')
x_scaled=sl.fit_transform(x)
y = df['target']
print(cross_val_score(LogisticRegression(), x_scaled, y,cv=3))
print(cross_val_score(RandomForestClassifier(n_estimators=40), x, y,cv=3))
print(cross_val_score(SVC(), x, y,cv=3))
