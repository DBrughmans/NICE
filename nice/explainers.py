import pandas as pd
import numpy as np
from nice.utils.distance import*
from nice.utils.preprocessing import OHE_minmax
from nice.utils.AE import AE_model
from nice.utils.data import data_NICE,data_SEDC
from nice.utils.optimization.heuristic import best_first
from nice.utils.optimization.heuristic import *
from math import ceil
# =============================================================================
# Types and constants
# =============================================================================
CRITERIA_DIS = {'HEOM':HEOM}
CRITERIA_NRM = {'std':std_scale,
                'minmax':minmax_scale}
CRITERIA_REW = {'sparsity':sparisity_reward,
                'proximity':proximity_reward}


class NICE:
    def __init__(self):
        pass
    def fit(self,
            predict_fn,
            X_train,
            cat_feat,
            num_feat ='auto',
            y_train=None,
            optimization='sparsity',
            justified_cf:bool = True,
            distance_metric='HEOM',
            num_normalization= 'minmax'):

        self.data = data_NICE(X_train,y_train,cat_feat,num_feat,predict_fn,justified_cf,0.0000001)
        self.distance_metric = CRITERIA_DIS[distance_metric](self.data, CRITERIA_NRM[num_normalization])
        self.optimization = best_first()
        self.reward_function = CRITERIA_REW['sparsity'](self.data,self.distance_metric)


    def explain(self,X,target_class ='other'):#todo target class 'other'
        self.data.fit_to_X(X,target_class,self.distance_metric)
        CF = self.optimization.optimize(self.data,self.reward_function)
        return CF

class SEDC:
    def __init__(self):
        pass

    def fit(self,
            predict_fn,
            X_train,
            cat_feat,
            num_feat='auto'):
        self.data = data_SEDC(X_train,predict_fn,cat_feat,num_feat)
        self.data.fit()
        self.optimization = best_first()
        self.reward_function = sparisity_reward()

    def explain(self,X,target_class):
        self.data.fit_to_X(X,target_class)
        CF = self.optimization.optimize(self.data,self.reward_function)
        return CF



