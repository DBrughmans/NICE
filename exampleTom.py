import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

#os.chdir()
feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                 'income']

df_train = pd.read_csv('data/adult.data', names=feature_names)
df_test = pd.read_csv('data/adult.test', names=feature_names)

num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'native-country']

all_features = num_features + cat_features

# %% take all features

income_train = df_train['income']
income_test = df_test['income']

y_train = pd.get_dummies(income_train).iloc[:, 1]
y_test = pd.get_dummies(income_test).iloc[:, 1]

# low income (0) and high income (1)

# Data with ordered features
X_train_raw = df_train[all_features].values
X_test_raw = df_test[all_features].values
cat_feat = [6, 7, 8, 9, 10, 11, 12, 13]
num_feat = [0, 1, 2, 3, 4, 5]



Pipe = Pipeline(
    [('PP',ColumnTransformer([
        ('con',StandardScaler(),num_feat),
        ('cat',OneHotEncoder(handle_unknown = 'ignore'),cat_feat)
    ])),
     ('RF',RandomForestClassifier())
])

parameters = {'RF__n_estimators': (10,50,100)}
gs_clf = GridSearchCV(Pipe, parameters, n_jobs=-1, cv=5, scoring='accuracy',verbose= 2)
gs_clf = gs_clf.fit(X_train_raw, y_train)
clf = gs_clf.best_estimator_

y_test_pred = clf.predict(X_test_raw)
print(accuracy_score(y_test, y_test_pred))  # on test set
print(confusion_matrix(y_test, y_test_pred))

# %%
from NICE.CF import NICE

predict_fn = lambda x: clf.predict_proba(x)

NICE_adult = NICE(optimization='sparsity')
NICE_adult.fit(X_train_raw, y_train.values, predict_fn, feature_names=all_features, cat_feat=cat_feat,
               con_feat=num_feat)

CF = []
for idx in range(X_test_raw.shape[0]):#explain all instances from test set
    CF.append(NICE_adult.explain(X_test_raw[idx:idx+1,:]))
    print('{} of {} explained'.format(idx + 1, X_test_raw.shape[0]), end='\r')

CF = np.concatenate(CF)
print(np.mean(clf.predict(X_test_raw)!=clf.predict(CF))) #1.0 = all valid CF
print(np.mean(np.sum(X_test_raw!=CF,axis=1)))#average sparsity = 2.3633
print(pd.Series(np.sum(X_test_raw!=CF,axis=0),index= all_features))#frequency of each feature in exp

