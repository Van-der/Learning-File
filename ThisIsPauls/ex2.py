import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
df= pd.read_csv('homeprices.csv')
print(df.columns)
print(df)
import math as mt
median=mt.floor(df.bedrooms.median())
print(f"median: {median}")
df.bedrooms.fillna(median,inplace=True)
print(df)
reg=lm.LinearRegression()
reg.fit(df[['area', 'bedrooms','age']], df['price'])
print("Without using joblib")
print(f"Coefficients: {reg.coef_} , Intercept: {reg.intercept_}")
print("price will be",reg.predict(pd.DataFrame({'area': [3000], 'bedrooms': [4], 'age': [40]})))
print(reg.predict([[3000, 4, 40]]))  # Corrected from 'predit' to 'predict'  
import joblib
joblib.dump(reg, 'ex2.pkl')
loadedreg=joblib.load('ex2.pkl')
print("Using joblib")
print(f"Coefficients: {loadedreg.coef_} , Intercept: {loadedreg.intercept_}")
ques=int(input("Enter the area, bedrooms and age for which the price is to be predicted: "))
print(f"Predicted Price for area: {ques} is: {loadedreg.predict(pd.DataFrame({'area': [ques], 'bedrooms': [4], 'age': [40]}))}")