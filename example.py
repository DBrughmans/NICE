from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from NICE.CF import NICE

adult = fetch_data('adult')
X = adult.iloc[:,:-1]
y = adult.iloc[:,-1]
feature_names = list(adult.columns)
X = X.values #only supports arrays atm
y= y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
RF = RandomForestClassifier()
RF.fit(X_train,y_train)

predict_fn = lambda x: RF.predict_proba(x) #NICE needs the prediction score of both classes

cat_feat = [1,3,4,5,6,7,8,9,13]
con_feat = [0,2,10,11,12]
NICE_adult = NICE(optimization='sparsity')
NICE_adult.fit(X_train,
               y_train,
               predict_fn,
               feature_names= feature_names,
               cat_feat=cat_feat,
               con_feat=con_feat)
CF = NICE_adult.explain(X_test[0:1,:])

