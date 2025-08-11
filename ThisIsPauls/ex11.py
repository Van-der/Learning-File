import pandas as pd
import numpy as np  
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
digits = load_digits()
print(dir(digits))
#print(digits.feature_names)
df=pd.DataFrame(data=digits.data,columns=digits.feature_names)
df['target'] = digits.target
#print(df['target'])
#df['target_name']=digits.target_names[df['target']]
print(df.head())
x = df.drop('target', axis='columns')
y=df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17)
svm=SVC()
svm.fit(x_train,y_train)
print("Score is : ", svm.score(x_test, y_test)) 