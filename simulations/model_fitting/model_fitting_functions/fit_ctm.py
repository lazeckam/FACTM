import os
import numpy as np
from simulations_settings import *
from big_simulations import *
from tqdm import tqdm
import time
import dill
import sys

sys.path.append('factm')
from model_class import *

max_iter = 200
how_many_models = 5

def run_simulations_ctm(params_scenario, param, ds_seed):

    file_path_scenario = os.path.join('artificial_datasets')

    file_name = params_scenario + "_" + str(param) + '_' + str(ds_seed) +'.pkl'
    file_name_save = params_scenario + "_" + str(param) + '_' + str(ds_seed) + '.pkl'

    with open(os.path.join(file_path_scenario, file_name), 'rb') as file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(file)

    ctm_params = {'data': data_simulations['M2'],
                  'L': int(params_sim['simulations_sample_ctm_params']['L'])}
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
    

    with open(os.path.join(os.path.join('results','ctm'), file_name_save), 'wb') as file:
        dill.dump(ctm_final, file)

