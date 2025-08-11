import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df= pd.read_csv('titanic.csv')
df['Age']=df['Age'].fillna(df['Age'].median())
y=df['Survived']
#print(x.head()) 
#print(y.head())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])
x=df[['Pclass','Sex','Age','Fare']]

#print(df['Sex'].head()) 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=17)
lm=LogisticRegression()
lm.fit(x_train,y_train)
print("Score is : ",lm.score(x_test,y_test))
