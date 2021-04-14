import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score

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

Pipe = Pipeline([
    ('PP',ColumnTransformer([
        ('con',Pipeline([('imp',SimpleImputer(strategy='mean')),('scl',StandardScaler())]),num_feat),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown = 'ignore'))]),cat_feat)
    ])),
     ('RF',RandomForestClassifier())
])

parameters = {'RF__n_estimators': (10,20)}
gs_clf = GridSearchCV(Pipe, parameters, n_jobs=-1, cv=2, scoring='accuracy',verbose= 2)
gs_clf = gs_clf.fit(X_train_raw, y_train)
best_params = gs_clf.best_params_
clf = Pipe.set_params(**best_params)
clf.fit(X_train_raw,y_train)

y_test_pred = clf.predict(X_test_raw)
y_train_pred = clf.predict(X_train_raw)
print(accuracy_score(y_test, y_test_pred))  # on test set
print(confusion_matrix(y_test, y_test_pred))

print(accuracy_score(y_train, y_train_pred))  # on test set
print(confusion_matrix(y_train, y_train_pred))
mask = clf.predict(X_train_raw)!= y_train

X_explain= X_train_raw[mask.values,:].copy()
# %%
from NICE.explainers import NICE

predict_fn = lambda x: clf.predict_proba(x)

NICE_adult = NICE(optimization='sparsity')
NICE_adult.fit(X_train_raw, y_train.values, predict_fn, feature_names=all_features, cat_feat=cat_feat,
               num_feat=num_feat, distance_metric='HEOM', num_normalization='std')

CF = []
for idx in range(X_explain.shape[0]):#explain all instances from test set
    CF.append(NICE_adult.explain(X_explain[idx:idx+1,:]))
    print('{} of {} explained'.format(idx + 1, X_test_raw.shape[0]), end='\r')

CF = np.concatenate(CF)
X_mod = X_train_raw.copy()#average sparsity = 2.3633
X_mod[mask,:]=CF

clf2 = Pipe.set_params(**best_params)
clf2.fit(X_train_raw,y_train)

y_test_pred = clf2.predict(X_test_raw)
print(accuracy_score(y_test, y_test_pred))  # on test set
print(confusion_matrix(y_test, y_test_pred),'b')


