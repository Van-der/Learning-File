import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn import linear_model as lm  
df = pd.read_csv('carprices.csv')
print(df)
x=df[['Car Model', 'Mileage','Age(yrs)']]
print("Initial x:")
print(x)
from sklearn.preprocessing import OneHotEncoder  
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x=ct.fit_transform(x)
print("Transformed x:")
print(x)
x=x[:, 1:]  # Avoiding the first column to prevent dummy variable trap
print("Final x after avoiding dummy variable trap:")
print(x)
reg= lm.LinearRegression()
reg.fit(x, df['Sell Price($)'])
import joblib
joblib.dump(reg, 'ex6.pkl')
loadedreg = joblib.load('ex6.pkl')
print("Using joblib")
print(f"Coefficients: {loadedreg.coef_} , Intercept: {loadedreg.intercept_}")
print(f"predicting for Car Model: 'BMW', Mileage: 20000, Age(yrs): 5")  
print(f"Predicted sell price: {loadedreg.predict(pd.DataFrame([[1,0,69000, 6]]))}")  #Corrected the input format

