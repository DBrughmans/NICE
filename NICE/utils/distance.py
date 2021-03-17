import pandas as pd
import numpy as np
from functools import partial
from typing import Dict, Callable, List, Sequence, Union

def distance_cf(original,CF,X_tr,cat_features):
    con_features = [i for i in range(0,X_tr.shape[1]) if i not in cat_features]
    mad = pd.DataFrame(X_tr).iloc[:,con_features].astype(float).mad().values

    distance_cat = np.sum(original[cat_features]!=CF[cat_features])/len(cat_features)
    distance_con = np.sum(abs(original[con_features].astype(float)-CF[con_features].astype(float))/mad)
    distance = distance_cat + distance_con
    return distance

def sparsity_cf(original,CF):
   return np.sum(original !=CF)

def VDM_pairwise_distance(X:np.array, y:np.array, alpha = 1, label =1):
    #todo error message if shapes are not the same
    #todo select positive label
    #todo missing values
    values = np.array(list(set(X)))
    distance = np.ones(shape = (len(values),len(values)))
    for i,value1 in enumerate(values):
        ci1 = np.sum(y[X==value1]==label)
        c1 = np.sum(X==value1)

        for j,value2 in enumerate(values):
            ci2 = np.sum(y[X==value2]==label)
            c2 = np.sum(X == value2)
            distance[i,j] = abs(ci1/c1-ci2/c2)**alpha
    return {'values':values,'distance':distance}


def HVDM(X1, X2, cat_pw_distance, con_std, cat_feat, con_feat, normalization ='N1'):
    '''

    :param X1:instance to calculate distance with X2. Shape = (1,number of features)
    :param distance:all instances to calculate distance from X1. shape = (number of instances,number of features)
    :param cat_pw_distance:dictionary with pairwise distances genereated by VDM_pairwise_distances
    :param con_std: array with 4*stdev of continous features. shape = (number of continous features)
    :param normalizatoin: N1,N2 or N3. See Improved Heteregeneous Distance functions, Wilson, Martinez 1997
    :return:distances between X1 and all instances in X2
    '''
    normalization_types = ['N1', 'N2', 'N3']
    if normalization not in normalization_types:
        raise ValueError("Invalid Normalization method. Expected one of: %s" % normalization_types)

    total_distance = np.zeros(X2.shape[0])
    for feat in cat_feat:
        values = cat_pw_distance[feat]['values']
        mask = values == X1[0, feat]
        pairwise_distances = cat_pw_distance[feat]['distance'][mask, :][0]
        cat_distance = X2[:,feat].copy()
        for value, pw_distance in zip(values, pairwise_distances):
            cat_distance[cat_distance == value] = pw_distance #replace cat value with pw distance
        if normalization in ['N2','N3']:
            cat_distance = cat_distance**2
        total_distance += cat_distance
    if normalization == 'N2':
        cat_distance = cat_distance**(1/2)
    elif normalization == 'N3':
        cat_distance = (len(cat_feat)*cat_distance)**(1/2)

    con_distance = np.sum(abs(X2[:, con_feat] - X1[0, con_feat]) / con_std,axis = 1)#todo fix error. always returns nan
    distance = cat_distance+con_distance
    return distance

def HEOM(X1, X2, cat_feat, con_feat, con_range):
    '''

    :param X1:
    :param X2:
    :param cat_feat:
    :param con_feat:
    :param con_range: difference between max and minimum of each continous feature
    :return:
    '''
    distance = X2.copy()
    distance[:, con_feat]= abs(distance[:, con_feat] - X1[0, con_feat]) / con_range
    distance[:, cat_feat]= distance[:, cat_feat] != X1[0, cat_feat]
    distance = np.sum(distance, axis =1)
    return distance

def IOF_pairwise_distance(X:np.array,scale=True):
    '''

    :param X:
    :param scale:
    :return:
    '''
    values = np.array(list(set(X)))
    distance = np.zeros(shape = (len(values),len(values)))
    for i,value1 in enumerate(values):
        freq_value1 = np.sum(X==value1)
        for j,value2 in enumerate(values):
            if value1 !=value2:
                freq_value2 = np.sum(X == value2)
                distance[i,j] = 1/(np.log(freq_value1)*np.log(freq_value2))
    if scale:
        distance /= np.max(distance)
    return {'values':values,'distance':distance}

def SOF_distance(X:np.array):
    '''
    simple occurancy factor
    :param X:
    :return:
    '''
    values = np.array(list(set(X)))
    distance = np.zeros(shape = len(values))
    for i,value1 in enumerate(values):
        distance[i] = 1-np.sum(X==value1)/len(X)
    return {'values':values,'distance':distance}




#source alibi.utils.discretizer
def mvdm_alibi(X: np.ndarray,
         y: np.ndarray,
         cat_vars: dict,
         alpha: int = 1) -> np.ndarray:
    """
    Calculate the pair-wise distances between categories of a categorical variable using
    the Modified Value Difference Measure based on Cost et al (1993).
    https://link.springer.com/article/10.1023/A:1022664626993
    Parameters
    ----------
    X
        Batch of arrays.
    y
        Batch of labels or predictions.
    cat_vars
        Dict with as keys the categorical columns and as optional values
        the number of categories per categorical variable.
    alpha
        Power of absolute difference between conditional probabilities.
    Returns
    -------
    Dict with as keys the categorical columns and as values the pairwise distance matrix for the variable.
    """
    # TODO: handle triangular inequality
    # infer number of categories per categorical variable
    n_y = len(np.unique(y))
    cat_cols = list(cat_vars.keys())
    for col in cat_cols:
        if cat_vars[col] is not None:
            continue
        cat_vars[col] = len(np.unique(X[:, col]))

    # conditional probabilities and pairwise distance matrix
    d_pair = {}
    for col, n_cat in cat_vars.items():
        d_pair_col = np.zeros([n_cat, n_cat])
        p_cond_col = np.zeros([n_cat, n_y])
        for i in range(n_cat):
            idx = np.where(X[:, col] == i)[0]
            for i_y in range(n_y):
                p_cond_col[i, i_y] = np.sum(y[idx] == i_y) / (y[idx].shape[0] + 1e-12)

        for i in range(n_cat):
            j = 0
            while j < i:  # symmetrical matrix
                d_pair_col[i, j] = np.sum(np.abs(p_cond_col[i, :] - p_cond_col[j, :]) ** alpha)
                j += 1
        d_pair_col += d_pair_col.T
        d_pair[col] = d_pair_col
    return d_pair

def abdm_alibi(X: np.ndarray,
         cat_vars: dict,
         cat_vars_bin: dict = dict()):
    """
    Calculate the pair-wise distances between categories of a categorical variable using
    the Association-Based Distance Metric based on Le et al (2005).
    http://www.jaist.ac.jp/~bao/papers/N26.pdf
    Parameters
    ----------
    X
        Batch of arrays.
    cat_vars
        Dict with as keys the categorical columns and as optional values
        the number of categories per categorical variable.
    cat_vars_bin
        Dict with as keys the binned numerical columns and as optional values
        the number of bins per variable.
    Returns
    -------
    Dict with as keys the categorical columns and as values the pairwise distance matrix for the variable.
    """
    # TODO: handle triangular inequality
    # ensure numerical stability
    eps = 1e-12

    # infer number of categories per categorical variable
    cat_cols = list(cat_vars.keys())
    for col in cat_cols:
        if cat_vars[col] is not None:
            continue
        cat_vars[col] = len(np.unique(X[:, col]))

    # combine dict for categorical with binned features
    cat_vars_combined = {**cat_vars, **cat_vars_bin}

    d_pair = {}  # type: Dict
    X_cat_eq = {}  # type: Dict
    for col, n_cat in cat_vars.items():
        X_cat_eq[col] = []
        for i in range(n_cat):  # for each category in categorical variable, store instances of each category
            idx = np.where(X[:, col] == i)[0]
            X_cat_eq[col].append(X[idx, :])

        # conditional probabilities, also use the binned numerical features
        p_cond = []
        for col_t, n_cat_t in cat_vars_combined.items():
            if col == col_t:
                continue
            p_cond_t = np.zeros([n_cat_t, n_cat])
            for i in range(n_cat_t):
                for j, X_cat_j in enumerate(X_cat_eq[col]):
                    idx = np.where(X_cat_j[:, col_t] == i)[0]
                    p_cond_t[i, j] = len(idx) / (X_cat_j.shape[0] + eps)
            p_cond.append(p_cond_t)

        # pairwise distance matrix
        d_pair_col = np.zeros([n_cat, n_cat])
        for i in range(n_cat):
            j = 0
            while j < i:
                d_ij_tmp = 0
                for p in p_cond:  # loop over other categorical variables
                    for t in range(p.shape[0]):  # loop over categories of each categorical variable
                        a, b = p[t, i], p[t, j]
                        d_ij_t = a * np.log((a + eps) / (b + eps)) + b * np.log((b + eps) / (a + eps))  # KL divergence
                        d_ij_tmp += d_ij_t
                d_pair_col[i, j] = d_ij_tmp
                j += 1
        d_pair_col += d_pair_col.T
        d_pair[col] = d_pair_col
    return d_pair

def pw_to_distance(X1,X2,d_pair):
    '''

    :param X1:instance to calculate distance with X2. Shape = (1,number of features)
    :param distance:all instances to calculate distance from X1. shape = (number of instances,number of features)
    :param cat_pw_distance:dictionary with pairwise distances genereated by VDM_pairwise_distances
    :param con_std: array with 4*stdev of continous features. shape = (number of continous features)
    :param normalizatoin: N1,N2 or N3. See Improved Heteregeneous Distance functions, Wilson, Martinez 1997
    :return:distances between X1 and all instances in X2
    '''

    total_distance = np.zeros(X2.shape[0])
    for key, value in d_pair.items():
        pw_distance = value[int(X1[0,key]),:]
        cat_distance = X2[:,key].copy()
        for i,d in enumerate(pw_distance):
            cat_distance[cat_distance == i]=pw_distance[i]#todo make new array. This way may cause problems if feature name = distance
        total_distance += cat_distance
    return total_distance
