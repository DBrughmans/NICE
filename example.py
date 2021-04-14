from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from NICE.explainers import NICE

adult = fetch_data('adult')
X = adult.drop(columns=['education-num','fnlwgt','target','native-country'])
y = adult.loc[:,'target']
feature_names = list(X.columns)
X = X.values #only supports arrays atm
y= y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cat_feat = [1,2,3,4,5,6,7]
num_feat = [0,8,9,10]

Pipe = Pipeline([
    ('PP',ColumnTransformer([
        ('con',Pipeline([('imp',SimpleImputer(strategy='mean')),
                         ('scl',StandardScaler())]),num_feat),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                         ('ohe',OneHotEncoder(handle_unknown = 'ignore'))]),cat_feat)
    ])),
     ('RF',RandomForestClassifier())
])
RF = RandomForestClassifier()
RF.fit(X_train,y_train)

predict_fn = lambda x: RF.predict_proba(x) #NICE needs the prediction score of both classes

NICE_adult = NICE(justified_cf=True,
                  optimization='proximity')
NICE_adult.fit(X_train = X_train,
               predict_fn=predict_fn,
               y_train = y_train,
               cat_feat=cat_feat,
               num_feat=num_feat)

CF = NICE_adult.explain(X_test[0:1,:])
