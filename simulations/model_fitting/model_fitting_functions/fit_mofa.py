import os
import numpy as np
from tqdm import tqdm
import time
import dill
import muon as mu
import anndata as ad
import mofax as mfx

max_iter = 200
how_many_models = 5

def run_simulations_mofa(params_scenario, param, ds_seed):

    file_path_scenario = os.path.join('artificial_datasets')

    file_name = params_scenario + "_" + str(param) + '_' + str(ds_seed) +'.pkl'

    with open(os.path.join(file_path_scenario, file_name), 'rb') as file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(file)

    m0 = ad.AnnData(data_simulations['M0'])
    m1 = ad.AnnData(data_simulations['M1'])
    mdata = mu.MuData({'M0': m0, 'M1': m1})

    model_list = []
    elbo_list = []

    for i in range(how_many_models):
        mu.tl.mofa(mdata, outfile='mofa_tmp_'+str(i)+"_"+str(file_name), n_iterations=max_iter, verbose=False, expectations='all', 
                   n_factors=params_sim['simulations_sample_fa_params']['K'], seed=100*ds_seed + i)
        
    for i in range(how_many_models):
        model = mfx.mofa_model("mofa_tmp_" +str(i)+"_"+str(file_name))
        elbo_list.append(model.training_stats['elbo'])
        model.close()

    which_best = np.argmax(np.array([elbo_list[i][-1] for i in range(how_many_models)]))

    model = mfx.mofa_model("mofa_tmp_" +str(which_best)+"_"+str(file_name))
    z = model.get_factors()
    w0 = model.expectations['W']['M0'][:,:]
    w1 = model.expectations['W']['M1'][:,:]
    model.close()


    with open(os.path.join(os.path.join('results', 'mofa'), file_name), 'wb') as file:
        dill.dump([z, w0, w1], file)


