import pandas as pd
df= pd.read_csv('heart.csv')
print(df.head())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#print(df['RestingBP'].nunique())

df['ChestPainType'] = le.fit_transform(df['ChestPainType'])
df['RestingECG'] = le.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = le.fit_transform(df['ST_Slope'])
df['Sex']=le.fit_transform(df['Sex'])
print(df.head())

x= df.drop('HeartDisease', axis=1)
y= df['HeartDisease']
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
model=LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
model.fit(x_train, y_train)
print("Model Score:", model.score(x_test, y_test))


from sklearn.decomposition import PCA
x_pca = PCA(9).fit_transform(x)
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(x_pca, y, test_size=0.25, random_state=42)
model_pca=LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
model_pca.fit(x_train_pca,y_train_pca)
print("Model Score with PCA:", model_pca.score(x_test_pca, y_test_pca))

