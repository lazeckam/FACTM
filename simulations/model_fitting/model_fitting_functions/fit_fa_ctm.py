import os
import numpy as np
from tqdm import tqdm
import time
import dill
import sys

sys.path.append('factm')
from model_class import *

max_iter = 200
how_many_models = 5

def run_simulations_fa_ctm(params_scenario, param, ds_seed):

    file_path_scenario = os.path.join('artificial_datasets')

    file_name = params_scenario + "_" + str(param) + '_' + str(ds_seed) +'.pkl'

    with open(os.path.join(file_path_scenario, file_name), 'rb') as file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(file)

        ctm_params = {'data': data_simulations['M2'],
              'L': params_sim['simulations_sample_ctm_params']['L']}
    ctm_list = []
    elbo_list = []

    for i in range(how_many_models):
        model0 = CTModel(**ctm_params, seed=100*ds_seed + i)
        model0.pretrain()

        model0.fit(max_iter)
            
        ctm_list.append(model0)
        elbo_list.append(model0.elbo_sequence)

    which_best = np.argmax(np.array([elbo_list[i][-1] for i in range(how_many_models)]))

    ctm_final = ctm_list[which_best]
    Observed = [True for m in range(params_sim['simulations_sample_fa_params']['M'])]

    model_params = {'data': {'M0': data_simulations['M0'],
                            'M1': data_simulations['M1'],
                            'M2': ctm_final.node_eta.vi_mu - ctm_final.node_mu0.mu0},
                    'K': params_sim['simulations_sample_fa_params']['K'], 
                    'O': Observed,
                    'S': Observed,
                    'L': []}
    model_list = []
    elbo_list = []

    for i in range(how_many_models):
        model0 = FACTModel(**model_params, seed=100*ds_seed + i)
        model0.pretrain()

        model0.fit(max_iter)
            
        model_list.append(model0)
        elbo_list.append(model0.elbo_sequence)

    which_best = np.argmax(np.array([elbo_list[i][-1] for i in range(how_many_models)]))

    model_final = model_list[which_best]

    with open(os.path.join(os.path.join('results', 'fa_ctm'), file_name), 'wb') as file:
        dill.dump([ctm_final, model_final], file)

