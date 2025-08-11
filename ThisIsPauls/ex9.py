import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
df=pd.read_csv('Iris.csv')
#print(df.head())
df=df.drop(['Id'], axis='columns')
print(df.head())
x=df.drop(['Species'], axis='columns') 
y=df['Species']
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2, random_state=17)
reg=lr()
reg.fit(x_train,y_train)
print(f"Testing Score: {reg.score(x_test, y_test)}")
y_predicted=reg.predict(x_test)
print(y_predicted)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm 
cm_iris=cm(y_test, y_predicted)
plt.figure(figsize=(10,10))

sns.heatmap(cm_iris,annot=True,fmt='d',cmap='coolwarm',xticklabels=reg.classes_,yticklabels=reg.classes_)
plt.xlabel('Predicted')
plt.ylabel('actual')
plt.title('Confusion Matrix for Iris Species Prediction')
plt.show()