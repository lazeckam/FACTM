import os
import numpy as np
from tqdm import tqdm
import time
import dill
import sys
from sklearn.decomposition import LatentDirichletAllocation as LDA

max_iter = 200
how_many_models = 1

def run_simulations_lda_sklearn(params_scenario, param, ds_seed):

    file_path_scenario = os.path.join('artificial_datasets')

    file_name = params_scenario + "_" + str(param) + '_' + str(ds_seed) +'.pkl'

    with open(os.path.join(file_path_scenario, file_name), 'rb') as file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(file)
    docs0 = np.vstack([np.sum(data_simulations['M2'][i], axis=0) for i in range(len(data_simulations['M2']))])

    lda_params = {'n_components': params_sim['simulations_sample_ctm_params']['L']}

    lda_list = []


    lda_final = LDA(**lda_params, max_iter=max_iter, random_state=12345, verbose=1)
    lda_final.fit(docs0)


    topics = lda_final.components_ / lda_final.components_.sum(axis=1)[:, np.newaxis]
    clusters_prob = lda_final.transform(docs0)
    exp_topic_word = lda_final.exp_dirichlet_component_

    with open(os.path.join(os.path.join('results', 'lda_sklearn'), file_name), 'wb') as file:
        dill.dump([topics, clusters_prob, exp_topic_word], file)

