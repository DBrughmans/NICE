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
    def __init__(self,data,distance_function=None):
        self.data = data
        pass
    def calculate_reward(self,X_prune,CF_candidate):
        score_prune = self.data.predict_fn(X_prune)
        score_diff = score_prune[:, self.data.target_class] - score_prune[:, self.data.X_class][:,np.newaxis]
        score_diff = score_diff.max(axis = 1)
        CF_candidate = X_prune[np.argmax(score_diff), :][np.newaxis, :]
        #stop = True if score_diff.max()>0 in self.data.target_class else False
        return CF_candidate

class ProximityReward(RewardFunction):
    def __init__(self,data,distance_metric:DistanceMetric):
        self.data = data
        self.distance_metric = distance_metric
    def calculate_reward(self,X_prune,CF_candidate):
        score_prune = self.data.predict_fn(X_prune)
        score_diff = self.data.X_score[:, self.data.X_class] - score_prune[:, self.data.X_class]
        distance = self.distance_metric.measure(self.data.X,X_prune)
        distance -= self.distance_metric.measure(self.data.X,CF_candidate)#todo multiclass
        idx_max = np.argmax(score_diff / (distance + self.data.eps))
        CF_candidate = X_prune[idx_max, :][np.newaxis,:]  # select the instance that has the highest score diff per unit of distance
        return CF_candidate

class PlausibilityReward(RewardFunction):
    def __init__(self,data,distance_metric:DistanceMetric,auto_encoder):
        self.data = data
        self.distance_metric = distance_metric
        self.auto_encoder = auto_encoder
    def calculate_reward(self,X_prune,CF_candidate):
        score_prune = self.data.predict_fn(X_prune)
        score_diff = self.data.X_score[:, self.data.X_class] - score_prune[:, self.data.X_class]
        distance = self.distance_metric.measure(self.data.X,X_prune)
        distance -= self.distance_metric.measure(self.data.X,CF_candidate)#todo multiclass
        idx_max = np.argmax(score_diff / (distance + self.data.eps))
        CF_candidate = X_prune[idx_max, :][np.newaxis,:]  # select the instance that has the highest score diff per unit of distance
        return CF_candidate

