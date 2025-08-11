import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
df= pd.read_csv('canada_per_capita_income.csv')
print(df.columns)
print(df.head(5))
# plt.scatter(df['year'], df['per capita income (US$)'], color='blue', marker='x')
# plt.xlabel('Year')
# plt.ylabel('Per Capita Income (USD)')
# plt.title('Per Capita Income in Canada Over Years')
reg=lm.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])
#print(reg.predict(pd.DataFrame({'year': [2025]})))
import pickle as pkl
with open('ex1.pkl', 'wb') as f:
    pkl.dump(reg, f)
with open('ex1.pkl', 'rb') as f:
    reg_loaded = pkl.load(f)
    print(f"Coefficients: {reg_loaded.coef_} , Intercept: {reg_loaded.intercept_}")
    ques=int(input("Enter the year for which the Per capita income is to be predicted: "))
    print(f"Predicted Per Capita Income for {ques} is: {reg_loaded.predict(pd.DataFrame({'year': [ques]}))}")