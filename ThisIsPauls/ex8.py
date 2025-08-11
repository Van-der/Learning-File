import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
df=pd.read_csv('HR_comma_sep.xls')
#print(df.head())
df_encoded=df.copy()
salaries_dum=pd.get_dummies(df_encoded.salary, drop_first=True).astype(int)
#print(salaries_dum.head())
df_encoded=pd.concat([df_encoded,salaries_dum], axis='columns')
departments_dum=pd.get_dummies(df_encoded.Department,drop_first=True).astype(int)
#print(departments_dum.head())
df_encoded=pd.concat([df_encoded,departments_dum], axis='columns')
df_encoded=df_encoded.drop(['salary', 'Department'], axis='columns')
#print(df_encoded.head())
x=df_encoded.drop(['left'], axis='columns')
scaler=ss()
x_Scaled=scaler.fit_transform(x )
y=df_encoded['left']
x_train, x_test, y_train, y_test = train_test_split(x_Scaled, y, test_size=0.3)
reg=lr()
reg.fit(x_train,y_train)
print(reg.score(x_test, y_test))
