import os
import numpy as np
from simulations_settings import *
from big_simulations import *
from tqdm import tqdm
import time
import dill
import muvi

max_iter = 2000
how_many_models = 5

def run_simulations_muvi(params_scenario, param, ds_seed):

    file_path_scenario = os.path.join('artificial_datasets')

    file_name = params_scenario + "_" + str(param) + '_' + str(ds_seed) +'.pkl'

    with open(os.path.join(file_path_scenario, file_name), 'rb') as file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(file)

    data_muvi = {'M0': data_simulations['M0'], 'M1': data_simulations['M1']}

    model_list = []
    elbo_list = []

    for i in range(how_many_models):
        
        device = "cpu"
        model = muvi.MuVI(
            observations=data_muvi,
            n_factors=params_sim['simulations_sample_fa_params']['K'],
            device=device, 
            )
        model.fit(batch_size=params_sim['simulations_sample_fa_params']['N'], n_epochs=max_iter, seed=100*ds_seed + i)

        model_list.append(model)

        elbo_list.append(model._svi.step(None, **model._setup_training_data()))

    which_best = np.argmin(np.array(elbo_list[i]))
    print('best: ', which_best)

    model = model_list[which_best]
    z = model.get_factor_scores()
    w0 = model.get_factor_loadings()['M0']
    w1 = model.get_factor_loadings()['M1']

    with open(os.path.join(os.path.join('results', 'muvi'), file_name), 'wb') as file:
        dill.dump([z, w0, w1], file)


