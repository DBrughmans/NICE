from abc import ABC,abstractmethod
import numpy as np
from nice.utils.distance import DistanceMetric


class RewardFunction(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def calculate_reward(self):
        pass


class SparsityReward(RewardFunction):

    def __init__(self,data,**kwargs):
        self.data = data
        pass

    def calculate_reward(self,X_prune, previous_CF_candidate):
        score_prune = -self.data.predict_fn(previous_CF_candidate) + self.data.predict_fn(X_prune)
        score_diff = score_prune[:, self.data.target_class]# - score_prune[:, self.data.X_class][:,np.newaxis] #multiclas
        score_diff = score_diff.max(axis = 1)
        idx_max = np.argmax(score_diff)
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        return CF_candidate


class ProximityReward(RewardFunction):
    def __init__(self,data,distance_metric:DistanceMetric,**kwargs):
        self.data = data
        self.distance_metric = distance_metric
    def calculate_reward(self,X_prune, previous_CF_candidate):
        score_prune = self.data.predict_fn(X_prune)
        score_diff = self.data.predict_fn(X_prune)[:, self.data.target_class]\
                     - self.data.predict_fn(previous_CF_candidate)[:, self.data.target_class]
        distance = self.distance_metric.measure(self.data.X,X_prune)\
                   -self.distance_metric.measure(self.data.X, previous_CF_candidate)
        idx_max = np.argmax(score_diff / (distance + self.data.eps)[:,np.newaxis])
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        return CF_candidate

class PlausibilityReward(RewardFunction):
    def __init__(self,data, auto_encoder, **kwargs):
        self.data = data
        self.auto_encoder = auto_encoder

    def calculate_reward(self,X_prune,previous_CF_candidate):
        score_prune = self.data.predict_fn(X_prune)
        score_diff = self.data.predict_fn(X_prune)[:, self.data.target_class_class]\
                     - self.data.predict_fn(previous_CF_candidate[:, self.data.target_class])#target_class for multiclass
        AE_loss_diff = self.auto_encoder(previous_CF_candidate)-self.auto_encoder(X_prune)
        idx_max = np.argmax(score_diff * (AE_loss_diff)[:,np.newaxis])
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        return CF_candidate

