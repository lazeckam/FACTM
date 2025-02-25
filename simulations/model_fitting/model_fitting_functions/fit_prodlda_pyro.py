
from fun_prodlda_pyro import *
import os
import numpy as np
from tqdm import tqdm
import time
import dill
import sys
import pyro
import torch

device = "cpu"
batch_size = 250 
learning_rate = 1e-3
num_epochs = 500
model_name = 'prodlda_pyro'

def run_simulations_prodlda_pyro(params_scenario, param, ds_seed):

    file_path_scenario = os.path.join('artificial_datasets')

    file_name = params_scenario + "_" + str(param) + '_' + str(ds_seed) +'.pkl'

    with open(os.path.join(file_path_scenario, file_name), 'rb') as file:
        [data_simulations, data_fa_info, data_ctm_info, params_sim] = dill.load(file)
    docs = np.vstack([np.sum(data_simulations['M2'][i], axis=0) for i in range(len(data_simulations['M2']))])
    docs = torch.from_numpy(docs)
    docs = docs.float().to(device) 
    num_topics = params_sim['simulations_sample_ctm_params']['L']
    

    torch.manual_seed(12345)
    pyro.set_rng_seed(12345)
        
    pyro.clear_param_store()

    prodLDA = ProdLDA(
        vocab_size=docs.shape[1],
        num_topics=num_topics,
        hidden=150,
        dropout=0.2
    )
    prodLDA.to(device)

    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    num_batches = int(math.ceil(docs.shape[0] / batch_size))

    bar = trange(num_epochs)
    for epoch in bar:
        running_loss = 0.0
        for i in range(num_batches):
            batch_docs = docs[i * batch_size:(i + 1) * batch_size, :]
            loss = svi.step(batch_docs)
            running_loss += loss / batch_docs.size(0)
        bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))

    logtheta_loc = prodLDA.encoder(docs)[0].detach().numpy()
    logtheta_scale = prodLDA.encoder(docs)[1].detach().numpy()
    beta = prodLDA.beta()

    with open(os.path.join(os.path.join('results', model_name), file_name), 'wb') as file:
        dill.dump([logtheta_loc, logtheta_scale, beta], file)