import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv('Housing.csv')
print(df.head())

df['mainroad'] = LabelEncoder().fit_transform(df['mainroad'])
df['guestroom'] = LabelEncoder().fit_transform(df['guestroom'])
df['basement'] = LabelEncoder().fit_transform(df['basement'])
df['hotwaterheating'] = LabelEncoder().fit_transform(df['hotwaterheating'])
df['airconditioning'] = LabelEncoder().fit_transform(df['airconditioning'])
df['prefarea'] = LabelEncoder().fit_transform(df['prefarea'])
df['furnishingstatus'] = LabelEncoder().fit_transform(df['furnishingstatus'])
x = df.drop('price', axis=1)
y = df['price']
print(df.isna().sum())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
reg = LinearRegression()
reg.fit(x_train, y_train)
print("score without any reg is:",reg.score(x_test, y_test))

from sklearn.linear_model import Lasso,Ridge
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(x_train, y_train)
print("score with lasso reg is:", lasso_reg.score(x_test, y_test))

ridge_Reg=Ridge(alpha=0.1)
ridge_Reg.fit(x_train, y_train)
print("score with ridge reg is:", ridge_Reg.score(x_test, y_test))
# Saving the model using joblib