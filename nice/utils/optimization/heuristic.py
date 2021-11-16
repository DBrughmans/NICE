from abc import ABC,abstractmethod
from nice.utils.optimization.reward import *
import numpy as np

class optimization(ABC):
    @abstractmethod
    def optimize(self):
        pass

class best_first(optimization):
    def __init__(self,data,reward_function:RewardFunction):
        self.reward_function = reward_function
        self.data = data

    def optimize(self,NN):
        CF_candidate = self.data.X.copy()
        stop = False
        while stop == False:
            diff = np.where(CF_candidate != NN)[1]
            X_prune = np.tile(CF_candidate, (len(diff), 1))
            for r, c in enumerate(diff):
                X_prune[r, c] = NN[0, c]
            CF_candidate = self.reward_function.calculate_reward(X_prune,CF_candidate)
            if self.data.predict_fn(CF_candidate).argmax() in self.data.target_class:
                return CF_candidate
