import pandas as pd
import numpy as np
from nice.utils.distance import HEOM
from nice.utils.preprocessing import OHE_minmax
from nice.utils.AE import AE_model
from math import ceil
class NICE:
    def __init__(self,optimization:str = 'sparsity',justified_cf:bool = True):
        """
        Initialize Nearest Instance Counterfactul Explanations
        :param optimization: {"none", "sparsity", "proximity", "plausibility"}, default = "sparsity"
        The optimization method.
        :param justified_cf: bool, default = True
        Whether the nearest neighbours are only searched whithin the correctly classified instances from X_train.
        If True, the labels are required (y_train)
        """
        if optimization not in ['none','sparsity','proximity','plausibility']:
            msg = 'Invalid argument for optimization: "{}"'
            raise ValueError(msg.format(optimization))
        self.optimization = optimization
        self.justified_cf = justified_cf
        self.eps = 0.0000001
        #todo check if all inputs are correct. elevate error
    def fit(self,
            predict_fn,
            X_train,
            cat_feat,
            num_feat ='auto',
            y_train=None,
            distance_metric='HEOM',
            num_normalization='minmax'):
        """

        :param predict_fn:
        Function returning class probabilities. predict_fn(X) should return a np.array with class probablities.
        :param X_train: np.array
        The training input samples.
        :param cat_feat: list
        List with column numbers of all categorical features
        :param num_feat: list
        List with column numbers of all numerical features
        :param y_train: np.array
        The training target values. Only required if justified_CF = True
        :param distance_metric: {"HEOM"}, default= "HEOM"
        Distance metric to select nearest neighbour and measure proximity.
        Currently only the Heterogeneous Overlap Method (HEOM) is supported
        :param num_normalization:{"minmax","std"}, default= "std"
        Normalization method for numerical features under HEOM distance metric. "minmax" normalizes values with the
        feature range, "std" normalizes with the standard deviation.
        Normalization method for numerical features
        """

        self.distance_metric = distance_metric
        self.X_train = X_train
        self.cat_feat = cat_feat
        self.predict_fn = predict_fn
        self.num_feat = num_feat
        self.num_normalization = num_normalization
        #todo raise error when wrong options are selected
        if self.optimization == 'plausibility':
            self._train_AE(self.X_train)

        if self.num_feat == 'auto':
            self.num_feat = [feat for feat in range(self.X_train.shape[1]) if feat not in self.cat_feat]
        self.X_train[:,self.num_feat] = X_train[:, self.num_feat].astype(np.float64)



        if self.distance_metric == 'HEOM':
            if self.num_normalization == 'minmax':
                self.con_scale = self.X_train[:, self.num_feat].max(axis=0) - self.X_train[:, self.num_feat].min(axis=0)
            elif self.num_normalization == 'std':
                self.con_scale = self.X_train[:, self.num_feat].std(axis=0, dtype=np.float64)
            else:
                msg = 'Invalid argument for con_normalization: "{}"'
                raise ValueError(msg.format(self.num_normalization))
            self.con_scale[self.con_scale < self.eps]=self.eps
        else:
            msg ='Invalid argument for distance_metric: "{}"'
            raise ValueError(msg.format(self.distance_metric))

        self.X_train_class = np.argmax(self.predict_fn(self.X_train), axis=1)
        if self.justified_cf:
            if y_train is None:
                raise TypeError("fit() missing 1 required positional argument: 'y_train'")
            mask_justified = (self.X_train_class == y_train)
            self.X_train = self.X_train[mask_justified, :]
            self.X_train_class = self.X_train_class[mask_justified]


    def explain(self,X,target_class ='other'):#todo target class 'other'
        """

        :param X: np.array
        Instance to explain.
        :param target_class: {"other"},default= "other"
        Class of the Counterfactual instance.
        Currently multi-class is not supported. Therefore this parameter should always be "other"
        :return: Counterfactual instance.
        """
        self.X = X
        self.X[:, self.num_feat] = X[:, self.num_feat].astype(np.float64)
        self.X_class = np.argmax(self.predict_fn(self.X), axis=1)[0]
        self.target_class = target_class
        if target_class =='other':
            candidate_mask = self.X_train_class!=self.X_class
            if self.X_class ==1:
                self.target_class=0
            else:
                self.target_class=1
        else:
            candidate_mask = self.X_train_class == target_class

        if self.distance_metric in ['ABDM','MVDM']:
            X_discrete = self.disc.discretize(self.X)
            X_candidates_discrete = self.X_train_discrete[candidate_mask,:]
        else:
            X_candidates=self.X_train[candidate_mask,:]


        if self.distance_metric == 'HVDM':
            distance = HVDM(self.X, X_candidates, self.cat_distance, self.con_distance, self.cat_feat, self.num_feat, normalization='N2')#Todo make normalizatin conditional parameter for .fit method
        elif self.distance_metric == 'HEOM':
            distance = HEOM(self.X, X_candidates, self.cat_feat, self.num_feat, self.con_scale)
        elif self.distance_metric in ['ABDM','MVDM']:
            distance = pw_to_distance(X_discrete, X_candidates_discrete, self.pw_distance)

        NN =  self.X_train[candidate_mask,:][distance.argmin(),:][np.newaxis,:]#todo return best when equal
        if self.optimization=='sparsity':
            NN = self._optimize_sparsity(self.X,NN)
        elif self.optimization == 'proximity':
            NN = self._optimize_proximity(self.X,NN)
        elif self.optimization == 'plausibility':
            NN = self._optimize_plausibility(self.X,NN)
        return NN.copy()



    def _fit_HVDM(self,X,y,cat_feat,con_feat):
        cat_distance = {}
        for feat in cat_feat:
            cat_distance[feat]=VDM_pairwise_distance(X[:, feat], y)

        con_distance=np.array([4*np.std(X[:,feat]) for feat in con_feat])
        con_distance[con_distance<self.eps]=self.eps
        return cat_distance,con_distance

    def _optimize_sparsity(self,X,NN):
        CF_candidate = X.copy()
        stop = False
        while stop == False:
            diff = np.where(CF_candidate!=NN)[1]
            X_prune = np.tile(CF_candidate, (len(diff), 1))
            for r, c in enumerate(diff):
                X_prune[r, c] = NN[0, c]
            score_prune = self.predict_fn(X_prune)
            score_diff = score_prune[:, self.target_class] - score_prune[:, self.X_class]
            CF_candidate = X_prune[np.argmax(score_diff), :][np.newaxis, :]
            if score_diff.max() > 0:
                stop = True
        return CF_candidate

    def _optimize_proximity(self,X,NN):
        CF_candidate = X.copy()
        X_score = self.predict_fn(X)[:,self.X_class]
        while self.predict_fn(CF_candidate).argmax()==self.X_class:
            diff = np.where(CF_candidate!=NN)[1]
            X_prune = np.tile(CF_candidate, (len(diff), 1))
            for r, c in enumerate(diff):
                X_prune[r, c] = NN[0, c]
            score_prune = self.predict_fn(X_prune)
            score_diff = X_score - score_prune[:, self.X_class]

            if self.distance_metric == 'HEOM':
                distance = HEOM(X, X_prune, self.cat_feat, self.num_feat, self.con_scale)
                distance -= HEOM(X, CF_candidate, self.cat_feat, self.num_feat, self.con_scale)
            idx_max = np.argmax(score_diff/(distance+self.eps))
            CF_candidate = X_prune[idx_max, :][np.newaxis, :]#select the instance that has the highest score diff per unit of distance
            X_score = score_prune[idx_max,self.X_class]
        return CF_candidate

    def _train_AE(self,X_train):
        self.PP = OHE_minmax(cat_feat=self.cat_feat, con_feat=self.num_feat)
        self.PP.fit(X_train)
        self.AE = AE_model(self.PP.transform(X_train).shape[1],2)
        self.AE.fit(self.PP.transform(X_train),self.PP.transform(X_train),
                    batch_size= ceil(X_train.shape[0]/10), epochs=20,verbose = 0)

    def _optimize_plausibility(self,X,NN):
        X_score = self.predict_fn(X)[:,self.X_class]
        CF_candidate = X.copy()
        while self.predict_fn(CF_candidate).argmax()==self.X_class:
            diff = np.where(CF_candidate!=NN)[1]
            X_prune = np.tile(CF_candidate, (len(diff), 1))
            for r, c in enumerate(diff):
                X_prune[r, c] = NN[0, c]
            score_prune = self.predict_fn(X_prune)
            score_diff = X_score-score_prune[:, self.X_class]

            X_prune_pp = self.PP.transform(X_prune)
            AE_loss_candidates = np.mean((X_prune_pp - self.AE.predict(X_prune_pp))**2,axis = 1)
            X_pp = self.PP.transform(CF_candidate)
            AE_loss_X = np.mean((X_pp - self.AE.predict(X_pp))**2)
            idx_max = np.argmax(score_diff*(AE_loss_X-AE_loss_candidates))
            CF_candidate = X_prune[idx_max, :][np.newaxis, :]#select the instance that has the highest score diff per unit of distance
            X_score = score_prune[idx_max,self.X_class]
        return CF_candidate