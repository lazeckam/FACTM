#region packages

import numpy as np
import os
import dill
from tqdm import tqdm
import pandas as pd
from pandas.api.types import CategoricalDtype
from copy import deepcopy
import sys
import torch

from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import completeness_score, fowlkes_mallows_score

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

sys.path.append('factm')
from FACTM_model import *

#endregion

#region functions computing metrics etc

def covariance_to_correlation(mat):
    return mat/np.sqrt(np.outer(np.diag(mat), np.diag(mat)))

def compute_corr_matrix(x1, x2, corr='spearman'):
    p = x1.shape[1]

    res = np.zeros((p,p))

    for i in range(p):
        for j in range(p):
            res[i,j] = compute_corr(x1[:,i], x2[:,j], corr)
    return res

def compute_corr_pairs(x1, x2, corr='spearman'):
    p = x1.shape[1]

    res = np.zeros((p))

    for i in range(p):
        res[i] = compute_corr(x1[:,i], x2[:,i], corr)
    return res

def compute_corr(x1, x2, corr='spearman'):

    res = 0
    if np.unique(x1).shape[0] > 1 and np.unique(x2).shape[0] > 1:
        if corr == 'spearman':
            res  = spearmanr(x1, x2)[0]
        if corr == 'pearson':
            res = pearsonr(x1, x2)[0]
    else:
        res = 0
        # print('corr one vec const')
    return res


def compure_r_squared_intercept(y, x):
    n = y.shape[0]
    p = y.shape[1]

    res = np.zeros((n))

    for i in range(n):
        denominator_i = np.sum((y[i,:] - np.mean(y[i,:]))**2)
        if denominator_i == 0:
            res[i] = 0
        else:
            res[i] = 1 - np.sum((y[i,:] - x[i,:] - np.mean(y[i,:]))**2)/denominator_i

    return np.mean(res)

def kl(p1, p2):
    return entropy(p1, p2)

def compute_dist_matrix(x1, x2, corr='kl'):
    p = x1.shape[1]

    res = np.zeros((p,p))

    for i in range(p):
        for j in range(p):
            res[i,j] = compute_dist(x1[:,i], x2[:,j], corr)
    return res


def compute_dist_pairs(x1, x2, corr='kl'):
    p = x1.shape[1]

    res = np.zeros((p))

    for i in range(p):
        res[i] = compute_dist(x1[:,i], x2[:,i], corr)
    return res

def compute_dist(x1, x2, corr='kl'):
    res = 0
    if corr == 'kl':
        res  = kl(x1, x2)
    if corr == 'wasserstein':
        res = wasserstein_distance(x1, x2)
    return res
def assignment_problem(cost_matrix):
    clusters_order = linear_sum_assignment(cost_matrix)[1]
    return clusters_order

def best_corr(x1, x2, corr='spearman'):
    p = x1.shape[1]
    corr_matrix = compute_corr_matrix(x1, x2, corr)
    best_order = assignment_problem(-np.abs(corr_matrix))
    best_order_corr = np.mean(np.diag(np.abs(corr_matrix[:,best_order])))
    max_corr = np.max(np.abs(corr_matrix), axis=1).mean()
    return best_order_corr, max_corr

def rotation_corr(x1, x2):
    # kabsch

    corr_matrix = compute_corr_matrix(x1, x2, corr='pearson')
    best_order = assignment_problem(-np.abs(corr_matrix))
    corr_matrix_best_order = corr_matrix[:,best_order]
    x2_best_order_good_sign = x2[:,best_order] * (-1*(np.diag(corr_matrix_best_order) < 0) + 1*(np.diag(corr_matrix_best_order) >= 0)) 
    corr_matrix_best_order_good_sign = compute_corr_matrix(x1, x2_best_order_good_sign, corr='pearson')

    H = corr_matrix
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    x2_rotated = np.dot(x2, R)
    corr_matrix_rotated = compute_corr_matrix(x1, x2_rotated, corr='pearson')

    return np.mean(np.diag(corr_matrix_rotated))

def compare_Sigma0(Sigma0, Sigma0_est):
    return np.linalg.norm(Sigma0_est - Sigma0, ord='fro')/np.linalg.norm(Sigma0, ord='fro')

def compare_clustering_supervised(classes, clusters, metric):
    res = 0
    if metric == 'ARI':
        res =  metrics.adjusted_rand_score(classes, clusters)
    if metric == 'completeness':
        res =  completeness_score(classes, clusters)
    if metric == 'FM':
        res = fowlkes_mallows_score(classes, clusters)
    return res


#endregion

#region FA and CTM - all metrics

def compute_latent_space_evaluation(z, z_est, mod_fa):

    results_fa = {
        'z_corr_rotated': None,
        'z_corr_best_order': None,
        'z_corr_max': None,
        'var_exp': None}
    
    results_fa['z_corr_best_order'], results_fa['z_corr_max'] = best_corr(z, z_est, corr='spearman')
    results_fa['z_corr_rotated'] = rotation_corr(z, z_est)
    results_fa['var_exp'] = compute_variance_explained(mod_fa)
    return results_fa

def compute_latent_space_evaluation_other(z, z_est):

    results_fa = {
        'z_corr_rotated': None,
        'z_corr_best_order': None,
        'z_corr_max': None}
    
    results_fa['z_corr_best_order'], results_fa['z_corr_max'] = best_corr(z, z_est, corr='spearman')
    results_fa['z_corr_rotated'] = rotation_corr(z, z_est)

    return results_fa

def compute_variance_explained(mod_fa):
    var_exp = mod_fa.variance_explained_per_view()
    if var_exp.shape[0] == 2:
        var_exp = np.hstack((var_exp, None))
    return var_exp


def compute_ctm_evaluation(params_true, params_estim, z_order, topic_order):
    eta_corr_pearson = np.mean(compute_corr_pairs(params_true['eta'].T, params_estim['eta'][:,topic_order].T, 'pearson'))
    eta_corr_spearmann = np.mean(compute_corr_pairs(params_true['eta'].T, params_estim['eta'][:,topic_order].T, 'spearman'))
    
    eta_r_squared = compure_r_squared_intercept(params_true['eta'], params_estim['eta'][:,topic_order])
    eta_prob_kl = np.mean(compute_dist_pairs(params_estim['prob'][:,topic_order].T, params_true['prob'].T, 'kl'))
    # print('eta_prob_kl', compute_dist_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'kl').shape)
    eta_prob_wasserstein = np.mean(compute_dist_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'wasserstein'))
    eta_est_prob_kl = np.mean(compute_dist_pairs(params_true['prob_est'].T, params_estim['prob'][:,topic_order].T, 'kl'))
    # print('eta_prob_kl', compute_dist_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'kl').shape)
    eta_est_prob_wasserstein  = np.mean(compute_dist_pairs(params_true['prob_est'].T, params_estim['prob'][:,topic_order].T, 'wasserstein'))
    # eta_prob_wasserstein = np.mean(compute_corr_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'pearson'))
    eta_prob_spearmann = np.mean(compute_corr_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'spearman'))
    

    # eta_prob_kl = np.mean(np.diag(compute_dist_matrix(params_true['prob'], params_estim['prob'][:,topic_order], 'kl')))
    # eta_prob_wasserstein = np.mean(np.diag(compute_dist_matrix(params_true['prob'], params_estim['prob'][:,topic_order], 'wasserstein')))


    # muFA_corr_pearson = np.mean(np.diag(compute_corr_matrix(params_true['muFA'], params_estim['muFA'][:,topic_order], 'pearson')))
    # muFA_corr_spearmann = np.mean(np.diag(compute_corr_matrix(params_true['muFA'], params_estim['muFA'][:,topic_order], 'spearman')))
    # muFA_r_squared = compure_r_squared_intercept(params_true['muFA'], params_estim['muFA'][:,topic_order])

    muFA_corr_pearson = np.mean(compute_corr_pairs(params_true['muFA'].T, params_estim['muFA'][:,topic_order].T, 'pearson'))
    muFA_corr_spearmann = np.mean(compute_corr_pairs(params_true['muFA'].T, params_estim['muFA'][:,topic_order].T, 'spearman'))
    muFA_r_squared = compure_r_squared_intercept(params_true['muFA'], params_estim['muFA'][:,topic_order])

    mu0_corr = compute_corr(params_true['mu0'], params_estim['mu0'][topic_order])
    Sigma0_Frobenius = compare_Sigma0(params_true['Sigma0'], params_estim['Sigma0'])
    Sigma0_corr_Frobenius = compare_Sigma0(params_true['Sigma0_corr'], params_estim['Sigma0_corr'])

    # np.mean(np.diag(compute_dist_matrix(params_true['topics'].T, params_estim['topics'][topic_order,:].T, 'kl')))
    topics_kl = np.mean(np.diag(compute_dist_matrix(params_estim['topics'][topic_order,:].T, params_true['topics'].T, 'kl')))
    topics_wasserstein = np.mean(np.diag(compute_dist_matrix(params_true['topics'].T, params_estim['topics'][topic_order,:].T, 'wasserstein')))
    topics_corr_spearman = np.mean(np.diag(compute_corr_matrix(params_true['topics'].T, params_estim['topics'][topic_order,:].T, 'spearman')))

    clusters_ARI = compare_clustering_supervised(params_true['clusters_all'], params_estim['clusters_all'], 'ARI')
    clusters_completeness = compare_clustering_supervised(params_true['clusters_all'], params_estim['clusters_all'], 'completeness')
    clusters_FM = compare_clustering_supervised(params_true['clusters_all'], params_estim['clusters_all'], 'FM')

    results_ctm = {
            'eta_corr_pearson': eta_corr_pearson,
            'eta_corr_spearmann': eta_corr_spearmann,
            'eta_r_squared': eta_r_squared,
            'eta_prob_kl': eta_prob_kl,
            'eta_prob_wasserstein': eta_prob_wasserstein,
            'eta_prob_est_kl': eta_est_prob_kl,
            'eta_prob_est_wasserstein': eta_est_prob_wasserstein,
            'eta_prob_spearmann': eta_prob_spearmann,
            'muFA_corr_pearson': muFA_corr_pearson,
            'muFA_corr_spearmann': muFA_corr_spearmann,
            'muFA_r_squared': muFA_r_squared,
            'mu0_corr': mu0_corr,
            'Sigma0_Frobenius': Sigma0_Frobenius,
            'Sigma0_corr_Frobenius': Sigma0_corr_Frobenius,
            'topics_kl': topics_kl,
            'topics_wasserstein': topics_wasserstein,
            'topics_corr_spearmann': topics_corr_spearman,
            'clusters_ARI': clusters_ARI,
            'clusters_completeness': clusters_completeness,
            'clusters_FM': clusters_FM
        }

    return results_ctm

    


#endregion

#region get params - true and in models 

def get_params_true(data_fa_info, data_ctm_info, params_sim):
    
    sample_artificial_info = [data_fa_info, data_ctm_info]

    counts_all = []
    for i in range(250):
        counts_0 = np.zeros(sample_artificial_info[1]['topics'].shape[0])
        ytu = np.unique(data_ctm_info['topics_per_sentence'][i], return_counts=True)
        counts_0[ytu[0]] = ytu[1]
        counts_0 = counts_0/100
        counts_all.append(counts_0)
    eta_est = np.array(counts_all)

    params = {
        'z': sample_artificial_info[0]['z'],
        'topics': sample_artificial_info[1]['topics'],
        'clusters_all': np.concatenate(sample_artificial_info[1]['topics_per_sentence']),
        'muFA': sample_artificial_info[1]['muFA'],
        'eta': sample_artificial_info[1]['eta'],
        'prob': np.array(sample_artificial_info[1]['topics_proportion_per_observation']),
        'prob_est': eta_est,
        'mu0': params_sim['simulations_sample_ctm_params']['mu0'],
        'Sigma0': params_sim['simulations_sample_ctm_params']['Sigma0'],
        'Sigma0_corr': covariance_to_correlation(params_sim['simulations_sample_ctm_params']['Sigma0'])
    }

    return params

def predict_clusters(mod_ctm):
    N = len(mod_ctm.node_xi.vi_par)
    return [np.argmax(mod_ctm.node_xi.vi_par[n], axis=1) for n in range(N)]
def predict_topics(mod_ctm):
    L = mod_ctm.node_beta.vi_alpha.shape[0]
    return np.array([mod_ctm.node_beta.vi_alpha[l,:]/np.sum(mod_ctm.node_beta.vi_alpha[l, :]) for l in range(L)])
def predict_z(mod_fa):
    return mod_fa.node_z.vi_mu
def predict_muFA(mod_fa):
    return mod_fa.nodelist_y[2].data
def predict_eta(mod_ctm):
    return mod_ctm.node_eta.vi_mu

def get_params_estim_factm(mod_fa, mod_ctm=None):
    
    if mod_ctm is None:
        params = {
            'z': predict_z(mod_fa)
        }
    else:
        eta_est = predict_eta(mod_ctm)
        prob_est = np.exp(eta_est)
        prob_est = prob_est/np.outer(np.sum(prob_est, axis=1), np.ones(eta_est.shape[1]))

        params = {
            'z': predict_z(mod_fa),
            'topics': predict_topics(mod_ctm),
            'clusters_all': np.concatenate(predict_clusters(mod_ctm)),
            'muFA': predict_muFA(mod_fa),
            'eta': eta_est,
            'prob': prob_est, 
            'mu0': mod_ctm.node_mu0.mu0,
            'Sigma0': mod_ctm.node_Sigma0.Sigma0,
            'Sigma0_corr': covariance_to_correlation(mod_ctm.node_Sigma0.Sigma0)
        }

    return params

#endregion

def compute_metrics(model_name, model_path, ds_path):
    # read 
    with open(model_path, "rb") as input_file:
        fitted_model = dill.load(input_file)
    with open(ds_path, "rb") as input_file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(input_file)
    
    # get params
    params_true = get_params_true(data_fa_info, data_ctm_info, params_sim)
    
    if model_name == 'factm':
        mod_fa = fitted_model.fa
        mod_ctm = fitted_model.ctm_list['M2']
    if model_name == 'fa_ctm':
        mod_fa = fitted_model[1].fa
        mod_ctm = fitted_model[0]
    if model_name == 'fa':
        mod_fa = fitted_model.fa
        mod_ctm = None
    if model_name == 'fa_oracle':
        mod_fa = fitted_model.fa
        mod_ctm = None
    params_estim = get_params_estim_factm(mod_fa, mod_ctm)

    results_fa = compute_latent_space_evaluation(params_true['z'], params_estim['z'], mod_fa)

    if mod_ctm is not None:
        corr_matrix = compute_corr_matrix(params_true['z'], params_estim['z'], 'spearman')
        z_order = assignment_problem(-np.abs(corr_matrix))

        # cont_matrix = np.array(pd.crosstab(params_true['clusters_all'], params_estim['clusters_all']))
        cont_matrix = compute_dist_matrix(params_true['topics'].T, params_estim['topics'].T)
        # topic_order = assignment_problem(-cont_matrix)
        topic_order = assignment_problem(cont_matrix)
        
        results_ctm = compute_ctm_evaluation(params_true, params_estim, z_order, topic_order)
    else:
        results_ctm = None


    return results_fa, results_ctm

def compute_fa_other(model_name, model_path, ds_path):
    # read 
    with open(model_path, "rb") as input_file:
        [z, w0, w1] = dill.load(input_file)
    with open(ds_path, "rb") as input_file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(input_file)
    
    # get params
    params_true = get_params_true(data_fa_info, data_ctm_info, params_sim)
    
    params_estim = {'z': z}

    results_fa = compute_latent_space_evaluation_other(params_true['z'], params_estim['z'])
    
    return results_fa


def compute_simulation_scenario(simulation_scenario, simulation_params, model_name, run_num, seed=123,
                                which_results = 'standard'):
    simulation_params = np.sort(simulation_params)
    par_num = simulation_params.shape[0]

    results_fa = {
            'z_corr_rotated': np.zeros((run_num, par_num)),
            'z_corr_best_order': np.zeros((run_num, par_num)),
            'z_corr_max': np.zeros((run_num, par_num)),
            'var_exp': np.zeros((run_num, par_num, 3))
        }

    results_ctm = {
        'eta_corr_pearson': np.zeros((run_num, par_num)),
        'eta_corr_spearmann': np.zeros((run_num, par_num)),
        'eta_r_squared': np.zeros((run_num, par_num)),
        'eta_prob_kl': np.zeros((run_num, par_num)),
        'eta_prob_wasserstein': np.zeros((run_num, par_num)),
        'eta_prob_est_kl': np.zeros((run_num, par_num)),
        'eta_prob_est_wasserstein': np.zeros((run_num, par_num)),
        'eta_prob_spearmann': np.empty((run_num, par_num)),
        'muFA_corr_pearson': np.zeros((run_num, par_num)),
        'muFA_corr_spearmann': np.zeros((run_num, par_num)),
        'muFA_r_squared': np.zeros((run_num, par_num)),
        'mu0_corr': np.zeros((run_num, par_num)),
        'Sigma0_Frobenius': np.zeros((run_num, par_num)),
        'Sigma0_corr_Frobenius': np.zeros((run_num, par_num)),
        'topics_kl': np.zeros((run_num, par_num)),
        'topics_wasserstein': np.zeros((run_num, par_num)),
        'topics_corr_spearmann': np.empty((run_num, par_num)),
        'clusters_ARI': np.zeros((run_num, par_num)),
        'clusters_completeness': np.zeros((run_num, par_num)),
        'clusters_FM': np.zeros((run_num, par_num))
    }
    
    for i in range(run_num):
        for param_ind in range(par_num):
            param = simulation_params[param_ind]
            if param != 1:
                if param == 5:
                    param = int(param)
                if param == 10:
                    param = int(param)
                if which_results == 'reconstruction':
                    file_name_mod = simulation_scenario + "_" + str(param) + "_" + str(seed+i) +'_reconstruction.pkl'
                else:
                    file_name_mod = simulation_scenario + "_" + str(param) + "_" + str(seed+i) +'.pkl'
                file_name = simulation_scenario + "_" + str(param) + "_" + str(seed+i) +'.pkl'
            else:
                file_name = 'basic' + "_" + str(None) + "_" + str(seed+i) +'.pkl'
                file_name_mod = file_name
                
            model_path = os.path.join('results', model_name, file_name_mod)
            ds_path = os.path.join('artificial_datasets', file_name)

            

            res_fa, res_ctm = compute_metrics(model_name, model_path, ds_path)

            for k in res_fa.keys():
                results_fa[k][i, param_ind,] = res_fa[k]
            if res_ctm is not None:
                for k in res_ctm.keys():
                    results_ctm[k][i, param_ind] = res_ctm[k]
            else:
                results_ctm = None

    return results_fa, results_ctm

def compute_simulation_scenario_fa(dataset_setting_scenario, simulation_scenario, simulation_params, model_name, run_num, seed=123,
                                which_results = None):
    simulation_params = np.sort(simulation_params)
    par_num = simulation_params.shape[0]

    results_fa = {
        'z_corr_rotated': np.full((run_num, par_num), np.nan),
        'z_corr_best_order': np.full((run_num, par_num), np.nan),
        'z_corr_max': np.full((run_num, par_num), np.nan),
        'var_exp': np.full((run_num, par_num, 3), np.nan)
    }

    for i in range(run_num):
        for param_ind in range(par_num):
            param = simulation_params[param_ind]
            if param != 1:
                if param == 5:
                    param = int(param)
                if param == 10:
                    param = int(param)
                file_name_mod = simulation_scenario + "_" + str(param) + "_" + str(seed+i) +'.pkl'
                file_name = simulation_scenario + "_" + str(param) + "_" + str(seed+i) +'.pkl'
            else:
                file_name = 'basic' + "_" + str(None) + "_" + str(seed+i) +'.pkl'
                file_name_mod = file_name
                
            model_path = os.path.join('results', dataset_setting_scenario, model_name, file_name_mod)
            ds_path = os.path.join('artificial_datasets', dataset_setting_scenario, file_name)


            res_fa = compute_fa_other(model_name, model_path, ds_path)

            if res_fa is not None:
                for k in res_fa.keys():
                    results_fa[k][i, param_ind] = res_fa[k]
            else:
                results_fa = None

    return results_fa
    

def compute_simulation_scenario_octis(dataset_setting_scenario, simulation_scenario, simulation_params, model_name, run_num, seed=123,
                                which_results = None):
    simulation_params = np.sort(simulation_params)
    par_num = simulation_params.shape[0]

    results_ctm = {
        'eta_corr_pearson': np.full((run_num, par_num), np.nan),
        'eta_corr_spearmann': np.full((run_num, par_num), np.nan),
        'eta_r_squared': np.full((run_num, par_num), np.nan),
        'eta_prob_kl': np.empty((run_num, par_num)),
        'eta_prob_wasserstein': np.empty((run_num, par_num)),
        'eta_prob_spearmann': np.empty((run_num, par_num)),
        'muFA_corr_pearson': np.full((run_num, par_num), np.nan),
        'muFA_corr_spearmann': np.full((run_num, par_num), np.nan),
        'muFA_r_squared': np.full((run_num, par_num), np.nan),
        'mu0_corr': np.full((run_num, par_num), np.nan),
        'Sigma0_Frobenius': np.full((run_num, par_num), np.nan),
        'Sigma0_corr_Frobenius': np.full((run_num, par_num), np.nan),
        'topics_kl': np.empty((run_num, par_num)),
        'topics_wasserstein': np.empty((run_num, par_num)),
        'topics_corr_spearmann': np.empty((run_num, par_num)),
        'clusters_ARI': np.full((run_num, par_num), np.nan),
        'clusters_completeness': np.full((run_num, par_num), np.nan),
        'clusters_FM': np.full((run_num, par_num), np.nan),
    }
    
    for i in range(run_num):
        for param_ind in range(par_num):
            param = simulation_params[param_ind]
            if param != 1:
                if param == 5:
                    param = int(param)
                if param == 10:
                    param = int(param)
                if which_results == 'reconstruction':
                    file_name_mod = simulation_scenario + "_" + str(param) + "_" + str(seed+i) +'_reconstruction.pkl'
                else:
                    file_name_mod = simulation_scenario + "_" + str(param) + "_" + str(seed+i) +'.pkl'
                file_name = simulation_scenario + "_" + str(param) + "_" + str(seed+i) +'.pkl'
            else:
                file_name = 'basic' + "_" + str(None) + "_" + str(seed+i) +'.pkl'
                file_name_mod = file_name
                
            model_path = os.path.join('results', dataset_setting_scenario, model_name, file_name_mod)
            ds_path = os.path.join('artificial_datasets', dataset_setting_scenario, file_name)

            

            res_ctm = compute_ctm_octis(model_name, model_path, ds_path)

            if res_ctm is not None:
                for k in res_ctm.keys():
                    results_ctm[k][i, param_ind] = res_ctm[k]
            else:
                results_ctm = None

    return results_ctm

def compute_simulation_scenario_other(dataset_setting_scenario, simulation_scenario, simulation_params, model_name, run_num, seed=123,
                                which_results = None):
    simulation_params = np.sort(simulation_params)
    par_num = simulation_params.shape[0]


    results_ctm = {
        'eta_corr_pearson': np.full((run_num, par_num), np.nan),
        'eta_corr_spearmann': np.full((run_num, par_num), np.nan),
        'eta_r_squared': np.full((run_num, par_num), np.nan),
        'eta_prob_kl':  np.full((run_num, par_num), np.nan),
        'eta_prob_wasserstein':  np.full((run_num, par_num), np.nan),
        'eta_prob_est_kl': np.full((run_num, par_num), np.nan),
        'eta_prob_est_wasserstein': np.full((run_num, par_num), np.nan),
        'eta_prob_spearmann':  np.full((run_num, par_num), np.nan),
        'muFA_corr_pearson': np.full((run_num, par_num), np.nan),
        'muFA_corr_spearmann': np.full((run_num, par_num), np.nan),
        'muFA_r_squared': np.full((run_num, par_num), np.nan),
        'mu0_corr': np.full((run_num, par_num), np.nan),
        'Sigma0_Frobenius': np.full((run_num, par_num), np.nan),
        'Sigma0_corr_Frobenius': np.full((run_num, par_num), np.nan),
        'topics_kl':  np.full((run_num, par_num), np.nan),
        'topics_wasserstein':  np.full((run_num, par_num), np.nan),
        'topics_corr_spearmann':  np.full((run_num, par_num), np.nan),
        'clusters_ARI': np.full((run_num, par_num), np.nan),
        'clusters_completeness': np.full((run_num, par_num), np.nan),
        'clusters_FM': np.full((run_num, par_num), np.nan),
    }
    
    for i in range(run_num):
        for param_ind in range(par_num):
            param = simulation_params[param_ind]
            if param != 1:
                if param == 5:
                    param = int(param)
                if param == 10:
                    param = int(param)
                file_name_mod = simulation_scenario + "_" + str(param) + "_" + str(seed+i) +'.pkl'
                file_name = simulation_scenario + "_" + str(param) + "_" + str(seed+i) +'.pkl'
            else:
                file_name = 'basic' + "_" + str(None) + "_" + str(seed+i) +'.pkl'
                file_name_mod = file_name
                
            model_path = os.path.join('results', dataset_setting_scenario, model_name, file_name_mod)
            ds_path = os.path.join('artificial_datasets', dataset_setting_scenario, file_name)

            if model_name == 'lda_sklearn':
                res_ctm = compute_ctm_lda_sklearn(model_name, model_path, ds_path)
            if model_name == 'prodlda_pyro':
                res_ctm = compute_ctm_prodlda_pyro(model_name, model_path, ds_path)

            if res_ctm is not None:
                for k in res_ctm.keys():
                    results_ctm[k][i, param_ind] = res_ctm[k]
                    # print('res', results_ctm[k][i, param_ind])
            else:
                results_ctm = None

    return results_ctm

def compute_ctm_octis(model_name, model_path, ds_path):
    with open(model_path, "rb") as input_file:
        topics, clusters_prob = dill.load(input_file)
    with open(ds_path, "rb") as input_file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(input_file)
    
    # get params
    params_true = get_params_true(data_fa_info, data_ctm_info, params_sim)

    if model_name == 'prodlda_octis':
        exp_beta = np.exp(topics).T
        exp_beta= exp_beta/np.sum(exp_beta, axis=0)
        exp_beta = exp_beta.T
        topics = exp_beta

    params_estim = {
            'topics': topics,
            'prob': clusters_prob.T
        }

    # cont_matrix = compute_corr_matrix(params_true['prob'], params_estim['prob'])
    # cont_matrix = compute_dist_matrix(params_true['topics'].T, params_estim['topics'].T)
    cont_matrix = compute_dist_matrix(params_true['prob'], params_estim['prob'])
    # topic_order = assignment_problem(-cont_matrix)
    topic_order = assignment_problem(cont_matrix)
        
    eta_prob_kl = np.mean(compute_dist_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'kl'))
    # print('eta_prob_kl', compute_dist_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'kl').shape)
    eta_prob_wasserstein = np.mean(compute_dist_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'wasserstein'))
    eta_prob_spearmann = np.mean(compute_corr_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'spearman'))
    eta_corr_spearmann = np.mean(compute_corr_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'spearman'))

    topics_kl = np.mean(np.diag(compute_dist_matrix(params_true['topics'].T, params_estim['topics'][topic_order,:].T, 'kl')))
    topics_wasserstein = np.mean(np.diag(compute_dist_matrix(params_true['topics'].T, params_estim['topics'][topic_order,:].T, 'wasserstein')))
    topics_corr_spearman = np.mean(np.diag(compute_corr_matrix(params_true['topics'].T, params_estim['topics'][topic_order,:].T, 'spearman')))

    results_ctm = {
            'eta_prob_kl': eta_prob_kl,
            'eta_prob_wasserstein': eta_prob_wasserstein,
            'eta_prob_spearmann': eta_prob_spearmann,
            'eta_corr_spearmann': eta_corr_spearmann,
            'topics_kl': topics_kl,
            'topics_wasserstein': topics_wasserstein,
            'topics_corr_spearmann': topics_corr_spearman
        }
    return results_ctm


def compute_ctm_prodlda_pyro(model_name, model_path, ds_path):
    with open(model_path, "rb") as input_file:
        logtheta_loc, logtheta_scale, beta = dill.load(input_file)
    with open(ds_path, "rb") as input_file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(input_file)
    
    # get params
    params_true = get_params_true(data_fa_info, data_ctm_info, params_sim)

    params_estim = {
            'topics': beta.numpy(),
            'prob': logtheta_loc
        }
    
    # print(logtheta_loc)

    cont_matrix = compute_corr_matrix(params_true['topics'].T, params_estim['topics'].T)
    # cont_matrix = compute_dist_matrix(params_true['topics'].T, params_estim['topics'].T)
    # cont_matrix = compute_dist_matrix(params_true['prob'], params_estim['prob'])
    # topic_order = assignment_problem(-cont_matrix)
    topic_order = assignment_problem(-cont_matrix)
        
    # eta_prob_kl = np.mean(compute_dist_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'kl'))
    # print('eta_prob_kl', compute_dist_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'kl').shape)
    # eta_prob_wasserstein = np.mean(compute_dist_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'wasserstein'))
    eta_prob_spearmann = np.mean(compute_corr_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'spearman'))
    eta_corr_spearmann = np.mean(compute_corr_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'spearman'))
    
    # topics_kl = np.mean(np.diag(compute_dist_matrix(params_estim['topics'][topic_order,:].T, params_true['topics'].T, 'kl')))
    # topics_wasserstein = np.mean(np.diag(compute_dist_matrix(params_true['topics'].T, params_estim['topics'][topic_order,:].T, 'wasserstein')))
    topics_corr_spearman = np.mean(np.diag(compute_corr_matrix(params_true['topics'].T, params_estim['topics'][topic_order,:].T, 'spearman')))

    results_ctm = {
            # 'eta_prob_kl': eta_prob_kl,
            # 'eta_prob_wasserstein': eta_prob_wasserstein,
            'eta_prob_spearmann': eta_prob_spearmann,
            'eta_corr_spearmann': eta_corr_spearmann,
            # 'topics_kl': topics_kl,
            # 'topics_wasserstein': topics_wasserstein,
            'topics_corr_spearmann': topics_corr_spearman
        }
    return results_ctm

def compute_ctm_lda_sklearn(model_name, model_path, ds_path):
    with open(model_path, "rb") as input_file:
        topics, clusters_prob, exp_topic_word = dill.load(input_file)
    with open(ds_path, "rb") as input_file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(input_file)
    
    # get params
    params_true = get_params_true(data_fa_info, data_ctm_info, params_sim)

    params_estim = {
            'topics': topics,
            'prob': clusters_prob
        }

    # cont_matrix = compute_corr_matrix(params_true['prob'], params_estim['prob'])
    # cont_matrix = compute_dist_matrix(params_true['topics'].T, params_estim['topics'].T)
    cont_matrix = compute_dist_matrix(params_true['prob'], params_estim['prob'])
    # topic_order = assignment_problem(-cont_matrix)
    topic_order = assignment_problem(cont_matrix)
    
        
    # eta_prob_kl = np.mean(compute_dist_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'kl'))
    # print('eta_prob_kl', compute_dist_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'kl').shape)
    eta_prob_wasserstein = np.mean(compute_dist_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'wasserstein'))
    eta_prob_spearmann = np.mean(compute_corr_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'spearman'))
    eta_corr_spearmann = np.mean(compute_corr_pairs(params_true['prob'].T, params_estim['prob'][:,topic_order].T, 'spearman'))
    eta_prob_kl = np.mean(compute_dist_pairs(params_estim['prob'][:,topic_order].T, params_true['prob'].T, 'kl'))
    eta_est_prob_kl = np.mean(compute_dist_pairs(params_true['prob_est'].T, params_estim['prob'][:,topic_order].T, 'kl'))
    eta_est_prob_wasserstein  = np.mean(compute_dist_pairs(params_true['prob_est'].T, params_estim['prob'][:,topic_order].T, 'wasserstein'))

    topics_kl = np.mean(np.diag(compute_dist_matrix(params_true['topics'].T, params_estim['topics'][topic_order,:].T, 'kl')))
    topics_wasserstein = np.mean(np.diag(compute_dist_matrix(params_true['topics'].T, params_estim['topics'][topic_order,:].T, 'wasserstein')))
    topics_corr_spearman = np.mean(np.diag(compute_corr_matrix(params_true['topics'].T, params_estim['topics'][topic_order,:].T, 'spearman')))

    results_ctm = {
            'eta_prob_kl': eta_prob_kl,
            'eta_prob_est_kl': eta_est_prob_kl,
            'eta_corr_spearmann': eta_corr_spearmann,
            'eta_prob_wasserstein': eta_prob_wasserstein,
            'eta_prob_est_wasserstein': eta_est_prob_wasserstein,
            'eta_prob_spearmann': eta_prob_spearmann,
            'topics_kl': topics_kl,
            'topics_wasserstein': topics_wasserstein,
            'topics_corr_spearmann': topics_corr_spearman
        }
    return results_ctm


#region plots

def boxplots(results, var, model_names, simulation_params, title, *args, **kwargs):
    df = pd.DataFrame()
    for i in range(len(results)):
        df_tmp = pd.DataFrame(results[i][var], columns=simulation_params)
        df_tmp = pd.melt(df_tmp, value_name=var, var_name='param')
        df_tmp['model'] = model_names[i]
        df = pd.concat([df_tmp, df])
    df = df.iloc[::-1]
    df.dropna(inplace=True)
    return sns.boxplot(df, x='param', y=var, hue='model', flierprops={"marker": "."}, *args, **kwargs)

def boxplots_var_exp(results, which_modality, model_names, simulation_params, title, *args, **kwargs):
    df = pd.DataFrame()
    var = 'var_exp'
    for i in range(len(results)):
        df_tmp = pd.DataFrame(results[i][var][:,:,which_modality], columns=simulation_params)
        df_tmp = pd.melt(df_tmp, value_name=var, var_name='param')
        df_tmp['model'] = model_names[i]
        df = pd.concat([df_tmp, df])
    df = df.iloc[::-1]
    model_cat = CategoricalDtype(model_names, ordered=True)
    df["model"] = df["model"].astype(model_cat)
    df.dropna(inplace=True)
    return sns.boxplot(df, x='param', y=var, hue='model', flierprops={"marker": "."}, *args, **kwargs)

def plot_latent_space(results, model_names, simulation_params):
    fig = plt.figure(constrained_layout=True, figsize=(10,9))
    fig.suptitle('Performance of the FA part')

    subfigs = fig.subfigures(nrows=3, ncols=1)
    for row, subfig in enumerate(subfigs):

        if row == 0:
            subfig.suptitle(f'Estimated vs. true hidden factors')
            vars = ['z_corr_rotated', 'z_corr_best_order', 'z_corr_max']
            vars_names = ['After optimal rotation of HFs', 'For optimally ordered HFs', 'Maximum over HFs']
        if row == 1:
            subfig.suptitle(f'Estimated vs. true hidden factors [0,1]')
            vars = ['z_corr_rotated', 'z_corr_best_order', 'z_corr_max']
            vars_names = ['After optimal rotation of HFs', 'For optimally ordered HFs', 'Maximum over HFs']
        if row == 2:
            subfig.suptitle(f'Explained variance per view')
            vars = [0,1,2]

        axs = subfig.subplots(nrows=1, ncols=3)
        for col, ax in enumerate(axs):
            if row == 0 and col == 0:
                ax0 = ax
            if row == 0:
                boxplots(results, vars[col], model_names, simulation_params, 'title', ax=axs[col]).set(ylabel='Spearmann correlation')
                ax.set_title(f''+str(vars_names[col]))
                ax.get_legend().set_visible(False)
            if row == 1:
                b = boxplots(results, vars[col], model_names, simulation_params, 'title', ax=axs[col])
                b.set_ylim([0,1])
                b.set(ylabel='Spearmann correlation')
                ax.set_title(f''+str(vars_names[col]))
                ax.get_legend().set_visible(False)
            if row == 2:
                boxplots_var_exp(results, vars[col], model_names, simulation_params, 'title', ax=axs[col]).set(ylabel='Variance explained')
                ax.set_title(f''+str(vars[col]))
                ax.get_legend().set_visible(False)
    lines_labels = [ax0.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='lower center', ncol=5)
    
def plot_ctm(results, model_names, simulation_params):
    fig = plt.figure(constrained_layout=True, figsize=(10,9*4/3))
    fig.suptitle('Performance of the CTM part')

    subfigs = fig.subfigures(nrows=6, ncols=1)
    for row, subfig in enumerate(subfigs):

        if row == 0:
            subfig.suptitle(f'Population-level variables')
            vars = ['mu0_corr', 'Sigma0_Frobenius', 'Sigma0_corr_Frobenius']
            vars_names = ['mu0', 'Sigma0', 'Sigma0 transformed to correlation']
            y_names = ['Spearmann corr', 'Frobenius', 'Frobenius']
        if row == 1:
            subfig.suptitle(f'Variables modified by the FA part')
            vars = ['eta_corr_spearmann', 'eta_corr_pearson', 'eta_r_squared']
            vars_names = ['eta', 'eta','eta = 1*eta_estim + intercept']
            y_names = ['Spearmann corr', 'Pearson corr', 'R squared']
        if row == 2:
            subfig.suptitle(f'Variables modified by the FA part')
            vars = ['eta_prob_kl', 'eta_prob_wasserstein', 'eta_prob_spearmann']
            vars_names = ['eta transformed to fractions','eta transformed to fractions', 'eta transformed to fractions']
            y_names = [ 'KL', 'Wasserstein', 'spearmann']
        if row == 3:
            subfig.suptitle(f'Variables modified by the FA part')
            vars = ['muFA_corr_spearmann', 'muFA_corr_pearson', 'muFA_r_squared']
            vars_names = ['muFA', 'muFA', 'muFA_true = 1*muFA_estim + intercept']
            y_names = ['Spearmann corr', 'Pearson corr', 'R squared']
        if row == 4:
            subfig.suptitle(f'Topic estimation')
            vars = ['topics_kl', 'topics_wasserstein', 'topics_corr_spearmann']
            vars_names = ['KL', 'Wasserstein', 'Spearman']
        if row == 5:
            subfig.suptitle(f'Evaluation of the clustering')
            vars = ['clusters_ARI', 'clusters_completeness', 'clusters_FM']
            vars_names = ['ARI score', 'Completeness score', 'Fowles-Mallows score']

        axs = subfig.subplots(nrows=1, ncols=3)
        for col, ax in enumerate(axs):
            if row == 0 and col == 0:
                ax0 = ax
            if vars[col] is not None:
                if row == 0:
                    boxplots(results, vars[col], model_names, simulation_params, 'title', ax=axs[col]).set(ylabel=y_names[col])
                    ax.set_title(f''+str(vars_names[col]))
                    ax.get_legend().set_visible(False)
                if row == 1:
                    boxplots(results, vars[col], model_names, simulation_params, 'title', ax=axs[col]).set(ylabel=y_names[col])
                    ax.set_title(f''+str(vars_names[col]))
                    ax.get_legend().set_visible(False)
                if row == 2:
                    boxplots(results, vars[col], model_names, simulation_params, 'title', ax=axs[col]).set(ylabel=y_names[col])
                    ax.set_title(f''+str(vars_names[col]))
                    ax.get_legend().set_visible(False)
                if row == 3:
                    boxplots(results, vars[col], model_names, simulation_params, 'title', ax=axs[col]).set(ylabel=y_names[col])
                    ax.set_title(f''+str(vars_names[col]))
                    ax.get_legend().set_visible(False)
                if row == 4:
                    boxplots(results, vars[col], model_names, simulation_params, 'title', ax=axs[col]).set(ylabel=vars_names[col])
                    ax.set_title(f''+str(f''+str(vars_names[col])))
                    ax.get_legend().set_visible(False)
                if row == 5:
                    boxplots(results, vars[col], model_names, simulation_params, 'title', ax=axs[col]).set(ylabel=vars_names[col])
                    ax.set_title(f''+str(f''+str(vars_names[col])))
                    ax.get_legend().set_visible(False)
    lines_labels = [ax0.get_legend_handles_labels()]
    print(ax.get_legend_handles_labels())
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='lower center', ncol=5)

pal = {'FACTM': sns.color_palette("Paired")[1],
       'FACTM(R)': sns.color_palette("Paired")[0],
       'CTM': sns.color_palette("Paired")[6],
       'LDA(O)': sns.color_palette("Paired")[2],
       'ProdLDA(O)': sns.color_palette("Paired")[4],
       'LDA': sns.color_palette("Paired")[2],
       'ProdLDA': sns.color_palette("Paired")[4],
       'NeuralLDA(O)': sns.color_palette("Paired")[8],
       'FA': sns.color_palette("Paired")[10],
       'FA+CTM': sns.color_palette("Set2")[0],
       'FA(Oracle)': sns.color_palette("Set2")[7],
       'MOFA': sns.color_palette("Paired")[8],
       'muVI': sns.color_palette("Paired")[4],
       'muVI_prior': sns.color_palette("Paired")[3]}

def plot_ctm_selected(df):
    fig = plt.figure(constrained_layout=True, figsize=(10,8))
    fig.suptitle(' ')

    scenarios = ['scaling_weights', 'scaling_topics_param', 'scaling_D_topics']

    subfigs = fig.subfigures(nrows=3, ncols=1)
    for row, subfig in enumerate(subfigs):

        df_tmp = df[df['sim_scenario'] == scenarios[row]]

        if row == 0:
            subfig.suptitle(f'Scenario 1: Scaling loadings of the CTM part by $\\lambda$')
            xlab = '$\\lambda$'
        if row == 1:
            subfig.suptitle(f'Scenario 2: Changing the hyperparameter $\\alpha$ of the topics\' distribution')
            xlab = '$\\alpha$'
        if row == 2:
            subfig.suptitle(f'Scenario 3: Changing the number of topics $L$')
            xlab = '$L$'
        vars = ['muFA_corr_spearmann', 'eta_prob_spearmann', 'topics_wasserstein', 'clusters_ARI']
        vars_names = [' $\\mu_{FA}$', '$\\eta$', '$\\beta$', '$\\xi$']
        y_names = ['Spearmann corr.', 'Spearmann corr.', 'Wasserstein dist.', 'ARI']

        axs = subfig.subplots(nrows=1, ncols=4)
        for col, ax in enumerate(axs):
            
            df_tmp_tmp = df_tmp[df_tmp['var'] == vars[col]]
            sns.boxplot(df_tmp_tmp, x='param', y='value', hue='Models', flierprops={"marker": "."}, palette=pal,
                        ax=axs[col]).set(ylabel=y_names[col], xlabel=xlab)
            ax.set_title(f''+str(vars_names[col]))
            if col != 2:
                ax.axhline(1, ls='--', c='black')
            else:
                ax.axhline(0, ls='--', c='black')
            
            if row == 0 and col == 1:
                ax0 = ax
            ax.get_legend().set_visible(False)

               
    lines_labels = [ax0.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', ncol=6)

# g = sns.FacetGrid(data=df, col="var2", sharey=False)
# g.map_dataframe(sns.boxplot, x='param', y='var', hue='Models', flierprops={"marker": "."}, palette=pal)
# g.set_axis_labels("Total bill ($)", ['a', 'b'])


#endregion