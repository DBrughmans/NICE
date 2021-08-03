import numpy as np
from abc import ABC,abstractmethod


class distance_metric(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def measure(self):
        pass


def std_scale(X_train,num_feat,eps):
    con_scale = X_train[:,num_feat].std(axis=0, dtype=np.float64)
    con_scale[con_scale < eps] = eps
    return con_scale


def minmax_scale(X_train,num_feat,eps):
    con_scale = X_train[:, num_feat].max(axis=0) - X_train[:, num_feat].min(axis=0)
    con_scale[con_scale < eps] = eps
    return con_scale

class HEOM(distance_metric):
    def __init__(self, data, normalization):
        self.cat_feat = data.cat_feat
        self.num_feat = data.num_feat
        self.eps = data.eps
        self.num_scale = normalization(data.X_train,self.num_feat,self.eps)
    def measure(self,X1,X2):
        distance = X2.copy()
        distance[:, self.num_feat] = abs(distance[:, self.num_feat] - X1[0, self.num_feat]) / self.num_scale
        distance[:, self.cat_feat] = distance[:, self.cat_feat] != X1[0, self.cat_feat]
        distance = np.sum(distance, axis=1)
        return distance

def find_NearestNeighbour(data,distance_metric):
    distances = distance_metric.measure(data.X,data.candidates_view)
    min_idx = distances.argmin()
    return data.candidates_view[min_idx,:].copy()[np.newaxis,:]



