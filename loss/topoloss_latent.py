import numpy
import gudhi as gd
from pylab import *
import torch
import math
from ripser import ripser
import cripser as cr
import os
from gudhi.wasserstein import wasserstein_distance

def getCriticalPoints_cr(likelihood, threshold):
        
    lh = 1 - likelihood
    pd = cr.computePH(lh, maxdim=1, location="birth")
    pd_arr_lh = pd[pd[:, 0] == 0] # 0-dim topological features
    pd_lh = pd_arr_lh[:, 1:3] # birth time and death time
    # birth critical points
    bcp_lh = pd_arr_lh[:, 3:5]
    dcp_lh = pd_arr_lh[:, 6:8]
    pairs_lh_pa = pd_arr_lh.shape[0] != 0 and pd_arr_lh is not None

    # if the death time is inf, set it to 1.0
    for i in pd_lh:
        if i[1] > 1.0:
            i[1] = 1.0
    
    pd_pers = abs(pd_lh[:, 1] - pd_lh[:, 0])
    valid_idx = np.where(pd_pers > threshold)[0]
    noisy_idx = np.where(pd_pers <= threshold)[0]

    pd_lh_filtered = pd_lh[valid_idx]
    bcp_lh_filtered = bcp_lh[valid_idx]
    dcp_lh_filtered = dcp_lh[valid_idx]

    #return pd_lh_filtered, bcp_lh_filtered, dcp_lh_filtered, pairs_lh_pa
    return pd_lh, bcp_lh, dcp_lh, pairs_lh_pa, valid_idx, noisy_idx


def get_matchings(lh_stu, lh_tea):
    
    cost, matchings = wasserstein_distance(lh_stu, lh_tea, matching=True)

    # print(f"Wasserstein distance value = {cost:.2f}")
    # dgm1_to_diagonal = matchings[matchings[:,1] == -1, 0]
    # dgm2_to_diagonal = matchings[matchings[:,0] == -1, 1]
    # off_diagonal_match = np.delete(matchings, np.where(matchings == -1)[0], axis=0)

    return cost


def compute_dgm_force(stu_lh_dgm, tea_lh_dgm):
    """
    Compute the persistent diagram of the image

    Args:
        stu_lh_dgm: likelihood persistent diagram of student model.
        tea_lh_dgm: likelihood persistent diagram of teacher model.

    Returns:
        idx_holes_to_remove: The index of student persistent points that require to remove for the following training process
        off_diagonal_match: The index pairs of persistent points that requires to fix in the following training process
    
    """
    if stu_lh_dgm.shape[0] == 0:
        idx_holes_to_remove, off_diagonal_match = np.zeros((0,2)), np.zeros((0,2))
        return idx_holes_to_remove, off_diagonal_match
    
    if (tea_lh_dgm.shape[0] == 0):
        tea_pers = None
        tea_n_holes = 0
    else:
        tea_pers = abs(tea_lh_dgm[:, 1] - tea_lh_dgm[:, 0])
        tea_n_holes = tea_pers.size

    if (tea_pers is None or tea_n_holes == 0):
        idx_holes_to_remove = list(set(range(stu_lh_dgm.shape[0])))
        off_diagonal_match = list()
    else:
        cost = get_matchings(stu_lh_dgm, tea_lh_dgm)
    
    return cost

def get_latent_wasserstein(latent1, latent2, threshold=0.7):
    latent1 = latent1.cpu().detach().numpy()
    latent2 = latent2.cpu().detach().numpy()

    pd_lh, bcp_lh, dcp_lh, pairs_lh_pa, valid_idx, noisy_idx = getCriticalPoints_cr(latent1, threshold)
    pd_gt, bcp_gt, dcp_gt, pairs_lh_gt, valid_idx_fake, noisy_idx_fake = getCriticalPoints_cr(latent2, threshold)

    pd_lh_for_matching = pd_lh[valid_idx]
    pd_gt_for_matching = pd_gt[valid_idx_fake]

    cost = compute_dgm_force(pd_lh_for_matching, pd_gt_for_matching)
    return cost

