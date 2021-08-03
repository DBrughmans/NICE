from abc import ABC,abstractmethod
from nice.utils.optimization.reward import *
import numpy as np

class optimization(ABC):
    @abstractmethod
    def optimize(self):
        pass

class best_first(optimization):
    def optimize(self,data,reward_function):
        CF_candidate = data.X.copy()
        stop = False
        while stop == False:
            diff = np.where(CF_candidate != data.replace_values)[1]
            X_prune = np.tile(CF_candidate, (len(diff), 1))
            for r, c in enumerate(diff):
                X_prune[r, c] = data.replace_values[0, c]
            CF_candidate,stop = reward_function.calculate_reward(X_prune,data,CF_candidate)
            if stop:
                return CF_candidate
        if data.predict_fn(CF_candidate).argmax in data.target_class:
            return CF_candidate
        else:
            CF_candidate[:]=np.nan
            return CF_candidate

