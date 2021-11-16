from nice.utils.distance import*
from nice.utils.data import data_NICE,data_SEDC
from nice.utils.optimization.heuristic import *
from typing import Optional
import numpy as np

# =============================================================================
# Types and constants
# =============================================================================
CRITERIA_DIS = {'HEOM':HEOM}
CRITERIA_NRM = {'std':StandardDistance,
                'minmax':MinMaxDistance}
CRITERIA_REW = {'sparsity':SparsityReward,
                'proximity':ProximityReward}


class NICE:
    def __init__(self):
        pass
    def fit(self,
            predict_fn,
            X_train:np.ndarray,
            cat_feat:list,
            num_feat ='auto',
            y_train: Optional[np.ndarray]=None,
            optimization='sparsity',
            justified_cf:bool = True,
            distance_metric:str ='HEOM',
            num_normalization:str = 'minmax'):

        self.data = data_NICE(X_train,y_train,cat_feat,num_feat,predict_fn,justified_cf,0.0000001)
        self.distance_metric = CRITERIA_DIS[distance_metric](self.data, CRITERIA_NRM[num_normalization])
        self.nearest_neighbour = NearestNeighbour(self.data, self.distance_metric)
        self.reward_function = CRITERIA_REW[optimization](self.data,self.distance_metric)
        self.optimization = best_first(self.data,self.reward_function)


    def explain(self,X,target_class ='other'):#todo target class 'other'
        self.data.fit_to_X(X,target_class)
        NN = self.nearest_neighbour.find_neighbour(self.data.X)
        CF = self.optimization.optimize(NN)
        return CF