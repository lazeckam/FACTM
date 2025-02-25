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

def run_simulations_fa(params_scenario, param, ds_seed):

    file_path_scenario = os.path.join('artificial_datasets')

    file_name = params_scenario + "_" + str(param) + '_' + str(ds_seed) +'.pkl'

    with open(os.path.join(file_path_scenario, file_name), 'rb') as file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(file)

    Observed = [not params_sim['which_ctm'][m] for m in range(params_sim['simulations_sample_fa_params']['M'] - 1)]

    data_simulations = {'M0': data_simulations['M0'],
                        'M1': data_simulations['M1']}

    model_params = {'data': data_simulations,
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


    with open(os.path.join(os.path.join('results', 'fa'), file_name), 'wb') as file:
        dill.dump(model_final, file)

    reconstruction_loss_list = []
    for i in range(how_many_models):
        reconstruction_loss_list.append(model_list[i].fa.nodelist_y[0].elbo + model_list[i].fa.nodelist_y[1].elbo)

    which_best = np.argmax(np.array(reconstruction_loss_list))
    model_final = model_list[which_best]

    file_name2 = params_scenario + "_" + str(param) + '_' + str(ds_seed) +'_reconstruction.pkl'
    with open(os.path.join(os.path.join('results', 'fa'), file_name2), 'wb') as file:
        dill.dump(model_final, file)

