from abc import ABC,abstractmethod
import numpy as np

class reward_function(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def calculate_reward(self):
        pass

class sparisity_reward(reward_function):
    def __init__(self,data=None,distance_function=None):
        pass
    def calculate_reward(self,X_prune,data,CF_candidate):
        score_prune = data.predict_fn(X_prune)
        score_diff = score_prune[:, data.target_class] - score_prune[:, data.X_class][:,np.newaxis]
        score_diff = score_diff.max(axis = 1)
        CF_candidate = X_prune[np.argmax(score_diff), :][np.newaxis, :]
        stop = True if score_diff.max()>0 in data.target_class else False

        return CF_candidate,stop

class proximity_reward(reward_function):
    def __init__(self,data,distance_metric):
        self.distance_metric = distance_metric
    def calculate_reward(self,X_prune,data,CF_candidate):
        score_prune = self.predict_fn(X_prune)
        score_diff = data.X_score - score_prune[:, self.X_class]
        distance = self.distance_metric.measure(data.X,X_prune)-self.distance_metric(data.X,CF_candidate)#todo multiclass
        idx_max = np.argmax(score_diff / (distance + data.eps))
        CF_candidate = X_prune[idx_max, :][np.newaxis,:]  # select the instance that has the highest score diff per unit of distance
        X_score = score_prune[idx_max, self.X_class]

