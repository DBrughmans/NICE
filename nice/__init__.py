from nice.utils.distance import*
from nice.utils.data import data_NICE
from nice.utils.optimization.heuristic import best_first
from nice.utils.optimization.reward import SparsityReward, ProximityReward, PlausibilityReward
from typing import Optional
import numpy as np

# =============================================================================
# Types and constants
# =============================================================================
CRITERIA_DIS = {'HEOM':HEOM}
CRITERIA_NRM = {'std':StandardDistance,
                'minmax':MinMaxDistance}
CRITERIA_REW = {'sparsity':SparsityReward,
                'proximity':ProximityReward,
                'plausibility':PlausibilityReward}


class NICE:
    def __init__(
            self,
            predict_fn,
            X_train:np.ndarray,
            cat_feat:list,
            num_feat ='auto',
            y_train: Optional[np.ndarray]=None,
            optimization='sparsity',
            justified_cf:bool = True,
            distance_metric:str ='HEOM',
            num_normalization:str = 'minmax',
            auto_encoder = None):

        self.optimization = optimization
        self.data = data_NICE(X_train,y_train,cat_feat,num_feat,predict_fn,justified_cf,0.00000000001)
        self.distance_metric = CRITERIA_DIS[distance_metric](self.data, CRITERIA_NRM[num_normalization])
        self.nearest_neighbour = NearestNeighbour(self.data, self.distance_metric)
        if optimization != 'none':
            self.reward_function = CRITERIA_REW[optimization](
                self.data,
                distance_metric = self.distance_metric,
                auto_encoder= auto_encoder
            )
            self.optimizer = best_first(self.data,self.reward_function)


    def explain(self,X,target_class ='other'):#todo target class 'other'
        self.data.fit_to_X(X,target_class)
        NN = self.nearest_neighbour.find_neighbour(self.data.X)
        if self.optimization != 'none':
            CF = self.optimizer.optimize(NN)
            return CF
        return NN