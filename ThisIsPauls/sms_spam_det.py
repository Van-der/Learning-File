import pandas as pd
from sklearn.preprocessing import LabelEncoder as le
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix 

df=pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
print(df.head())
df['label'] = le().fit_transform(df['label'])
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
cpl=Pipeline([
    ('vectorize',CountVectorizer()),
    ('class',MultinomialNB())
])
x_train,x_test,y_train,y_test= train_test_split(df['message'], df['label'], test_size=0.25, random_state=17)
cpl.fit(x_train, y_train)
print(cpl.score(x_test, y_test))
cm= confusion_matrix(y_test, cpl.predict(x_test))
plt.figure(figsize=(10, 6))
sns.heatmap(cm,annot=True,fmt='d',cmap='coolwarm', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
email=["Hey! Are we still meeting for lunch at 1pm?","Congratulations! You've won a free ticket to Bahamas. Text WIN to 12345 to claim now!"]
print('prediction is',cpl.predict(email))