import pandas as pd
import numpy as np
from sklearn import linear_model as lm

from sklearn.model_selection import train_test_split
df=pd.read_csv('carprices.csv')
print(df)
x=df[['Mileage','Age(yrs)']]
y=df['Sell Price($)']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
reg=lm.LinearRegression()
reg.fit(x_train, y_train)
#print(x_train.shape, y_train.shape)
print(reg.score(x_test, y_test))