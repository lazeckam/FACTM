from tensorly.decomposition import tucker, constrained_parafac, parafac
import os
import dill
import sys
import numpy as np
import tensorly as tl

max_iter = 2000
how_many_models = 1

def run_simulations_tucker(params_scenario, param, ds_seed):

    file_path_scenario = os.path.join('artificial_datasets')

    file_name = params_scenario + "_" + str(param) + '_' + str(ds_seed) +'.pkl'

    with open(os.path.join(file_path_scenario, file_name), 'rb') as file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(file)

    data_tensor = np.array([data_simulations['M0'], data_simulations['M1']])
    data_tensor = np.moveaxis(data_tensor, [0,1,2], [2,0,1])
    data_tensor = tl.tensor(data_tensor)

    core, factors = tucker(data_tensor, 
                            rank=[params_sim['simulations_sample_fa_params']['N'], params_sim['simulations_sample_fa_params']['K'], 1])

    z = np.dot(core[:,:,0].T, factors[0]).T
    w0 = factors[1]*factors[2][0]
    w1 = factors[1]*factors[2][1]

    with open(os.path.join(os.path.join('results', 'tucker'), file_name), 'wb') as file:
        dill.dump([z, w0, w1], file)


