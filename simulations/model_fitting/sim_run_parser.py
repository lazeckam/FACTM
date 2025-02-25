
import sys

sys.path.append('model_fitting_functions')

# factm
from fit_factm import *

# FA&CTM
from fit_fa_ctm import *

# latent factor models
from fit_fa import *
from fit_fa_oracle import *
from fit_mofa import *
from fit_muvi_prior import *
from fit_muvi import *
from fit_pca import *
from fit_tucker import *

# topic models
from fit_ctm import *
from fit_prodlda_pyro import *
from fit_lda_sklearn import *


import argparse

parser = argparse.ArgumentParser(description='Run factm.')

parser.add_argument('i0', type=int, help='just index')
parser.add_argument('model0', type=str, help='a model')
parser.add_argument('scenario0', type=str, help='scenario')

args = parser.parse_args()
i0 = args.i0
model0 = args.model0
scenario0 = args.scenario0

# parameters depend on the simulation scenario
# lambdas = [0.0, 0.5, 1.5, 2.0, None]
# lambdas = [1.5]
# lambdas = [0.25, 0.0]
# lambdas = [5, 10]
# lambdas = [0.0, 0.25, 0.5, 0.75]
lambdas = [0.5, 1.5]

lambda0 = lambdas[i0]

for i in range(10):

    print("Lambda = ", lambda0)

    if model0 == 'factm':
        run_simulations_factm(scenario0,  lambda0, 123+i)

    if model0 == 'fa_ctm':
        run_simulations_fa_ctm(scenario0,  lambda0, 123+i)
    
    if model0 == 'fa':
        run_simulations_fa_4M(scenario0,  lambda0, 123+i)
    if model0 == 'fa_oracle':
        run_simulations_fa_oracle(scenario0,  lambda0, 123+i)
    if model0 == 'mofa':
        run_simulations_mofa(scenario0,  lambda0, 123+i)
    if model0 == 'muvi':
        print('muvi')
        run_simulations_muvi(scenario0,  lambda0, 123+i)
    if model0 == 'muvi_prior':
        print('muvi_prior')
        run_simulations_muvi_prior(scenario0,  lambda0, 123+i)
    if model0 == 'tucker':
        print('tucker')
        run_simulations_tucker(scenario0,  lambda0, 123+i)
    if model0 == 'pca':
        print('pca')
        run_simulations_pca(scenario0,  lambda0, 123+i)

    if model0 == 'ctm':
        L0 = 10 # the true number of topics in simulations
        run_simulations_ctm(scenario0,  lambda0, 123+i, L0)
    if model0 == 'prodlda_pyro':
        run_simulations_prodlda_pyro(scenario0,  lambda0, 123+i)
    if model0 == 'lda_sklearn':
        run_simulations_lda_sklearn(scenario0,  lambda0, 123+i)