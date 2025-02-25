
import os
import dill
import sys
import numpy as np
from sklearn.decomposition import PCA

max_iter = 2000
how_many_models = 1

def run_simulations_pca(params_scenario, param, ds_seed):

    file_path_scenario = os.path.join('artificial_datasets', scenario)

    file_name = params_scenario + "_" + str(param) + '_' + str(ds_seed) +'.pkl'

    with open(os.path.join(file_path_scenario, file_name), 'rb') as file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(file)

    data_pca = np.hstack([data_simulations['M0'], data_simulations['M1']])

    pca = PCA(n_components=5)

    z = pca.fit_transform(data_pca)
    w0 = pca.components_[:,:10]
    w1 = pca.components_[:,10:]
        

    with open(os.path.join(os.path.join('results', 'pca'), file_name), 'wb') as file:
        dill.dump([z, w0, w1], file)


