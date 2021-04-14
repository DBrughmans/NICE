from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import numpy as np

class OHE_minmax:
    def __init__(self,cat_feat,con_feat):
        self.cat_feat = cat_feat
        self.con_feat = con_feat

    def fit(self,X):
        if self.cat_feat != []:
            self.OHE = OneHotEncoder(handle_unknown='ignore',sparse= False)
            self.OHE.fit(X[:,self.cat_feat])
            self.nb_OHE = self.OHE.transform(X[0:1,self.cat_feat]).shape[1]
        if self.con_feat != []:
            self.minmax = MinMaxScaler(feature_range=(-1, 1))
            self.minmax.fit(X[:,self.con_feat])

    def transform(self,X):
        if self.cat_feat == []:
            return self.minmax.transform(X[:, self.con_feat])
        elif self.con_feat == []:
            return self.OHE.transform(X[:,self.cat_feat])
        else:
            X_minmax = self.minmax.transform(X[:,self.con_feat])
            X_ohe = self.OHE.transform(X[:,self.cat_feat])
            return np.c_[X_ohe,X_minmax]

    def inverse_transform(self,X):
        if self.cat_feat == []:
            return self.minmax.inverse_transform(X)
        elif self.con_feat == []:
            return self.OHE.inverse_transform(X[:,:self.nb_OHE])
        else:
            X_con = self.minmax.inverse_transform(X[:,self.nb_OHE:])
            X_cat = self.OHE.inverse_transform(X[:,:self.nb_OHE])
            return np.c_[X_cat,X_con]
