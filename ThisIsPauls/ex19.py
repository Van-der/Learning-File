import pandas as pd
from scipy import stats
df=pd.read_csv('heart.csv')
#print(df.head())

num_cols=df.select_dtypes(include=['int64', 'float64']).columns
#print(num_cols)

zscores=stats.zscore(df[num_cols])
z_scores_df=pd.DataFrame(zscores, columns=num_cols)
#print(z_scores_df.head())
mask=(z_scores_df.abs() < 3).all(axis=1)
#print(mask)
df_no_outliers=df[mask]
#print(df_no_outliers.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df1=df_no_outliers.copy()
label_encoder = LabelEncoder()
for col in df1.select_dtypes(include=['object']).columns:
    df1[col] = label_encoder.fit_transform(df1[col])
print(df1.head())

X = df1.drop('HeartDisease', axis=1)
y= df1['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

#3print(df1.HeartDisease.value_counts())

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
dec_Tree=DecisionTreeClassifier(random_state=42)
dec_Tree.fit(x_train_scaled, y_train)
score=dec_Tree.score(X_test_scaled, y_test)
print(f"Test score of decision tree classifier: {score}")


from sklearn.model_selection import cross_val_score
scores= cross_val_score(DecisionTreeClassifier(), x_train_scaled, y_train, cv=5)
print(f"Cross-validation score: {scores.mean()}")

from sklearn.ensemble import BaggingClassifier
bag=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42,
    oob_score=True,
)
bag.fit(x_train_scaled, y_train)
#print(f"Out-of-bag score: {bag.oob_score_}")
score=bag.score(X_test_scaled, y_test)
print(f"Test score of bagging classifier with Dec Tree without CVS: {score}")

score=cross_val_score(bag,x_train_scaled,y_train,cv=5)
print(f"Cross-validation score of bagging classifier with Dec Tree: {score.mean()}")

from sklearn.svm import SVC
svm=SVC(kernel='rbf', random_state=42)
svm.fit(x_train_scaled, y_train)
score=svm.score(X_test_scaled, y_test)
print(f"Test score of SVM classifier: {score}")

score=cross_val_score(svm,x_train_scaled,y_train,cv=5)
print(f"Cross-validation score of SVM classifier: {score.mean()}")

bad=BaggingClassifier(
    estimator=svm,
    n_estimators=100,
    random_state=42,
    oob_score=True
)
bad.fit(x_train_scaled, y_train)
score=bad.score(X_test_scaled, y_test)
print(f"Test score of bagging classifier with SVM without CVS: {score}")
score=cross_val_score(bad,x_train_scaled,y_train,cv=5)
print(f"Cross-validation score of bagging classifier with SVM: {score.mean()}")
#creating table with results
results = {
    'Model': ['Decision Tree', 'Bagging with Decision Tree', 'SVM', 'Bagging with SVM'],
    'Test Score': [dec_Tree.score(X_test_scaled, y_test), bag.score(X_test_scaled, y_test), svm.score(X_test_scaled, y_test), bad.score(X_test_scaled, y_test)],
    'Cross-Validation Score': [scores.mean(), cross_val_score(bag, x_train_scaled, y_train, cv=5).mean(), score.mean(), cross_val_score(bad, x_train_scaled, y_train, cv=5).mean()]
}
print(pd.DataFrame(results))