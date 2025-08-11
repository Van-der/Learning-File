import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

digit = load_digits()
print(dir(digit))
df=pd.DataFrame(digit.data)
df['target'] = digit.target
print(df.head())
x = df.drop('target', axis=1)
y = df['target']
model=KNeighborsClassifier()
params={
    'n_neighbors': [3, 5, 7, 9],
}
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

clf=GridSearchCV(model, params, cv=3, return_train_score=False)
clf.fit(x_train, y_train)
print("Best Parameters:", clf.best_params_)

model=KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'])
model.fit(x_train, y_train)
print("Test Score:", model.score(x_test, y_test))


from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(x_test)
cm= confusion_matrix(y_test, y_pred)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(cm,annot=True,fmt='d',cmap='coolwarm')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print(classification_report(y_test, y_pred))