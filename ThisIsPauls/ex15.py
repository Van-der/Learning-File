import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

digit=load_digits()
df=pd.DataFrame(digit.data)
df['target'] = digit.target
print(df.head())

x=df.drop('target', axis=1)
y=df['target']
model_param={
    'svm': {
        'model': SVC(gamma='auto'),
        'params': {'C': [1, 10, 20, 30], 
                   'kernel': ['linear', 'rbf']
                    }
    },
    'gausian NB': {
        'model': GaussianNB(),
        'params': {'var_smoothing': [1e-9,1e-10,1e-11]
                    }
    },
    'multinomial NB': {
        'model': MultinomialNB(),
        'params': {'alpha': [1,0]
                    }
    },
    'decision Tree': {
        'model': DecisionTreeClassifier(),
        'params': {'criterion': ['entropy', 'gini'],
                   }
    },
    'random Forest': {
        'model': RandomForestClassifier(),
        'params': {'n_estimators': [1, 10, 20, 30]
                    }
    },
    'log Regression': {
        'model': LogisticRegression(),
        'params': {'C': [1,5,10]
                    }
    }
}

scores=[]
for model_name,mp in model_param.items():
    clf = GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    clf.fit(x, y)
    score = clf.score(x, y)
    scores.append({
        'model': model_name,
        'best_params': clf.best_params_,
        'best_score': clf.best_score_
    })

df=pd.DataFrame(scores,columns=['model','best_params','best_score'])
print(df)