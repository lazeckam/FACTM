import numpy as np

def simulations_sample_fa(K, N, D, M, 
                          sparsity_factor_fraction=0.2, check_sparsity_factor=True, 
                          sparsity_feature_fraction=0.2, check_sparsity_feature=True,
                          var_data=None, var_weights=None,
                          write_all=False, 
                          data_centered=True, muFA_scale=None, seed=123):
    
    np.random.seed(seed)
    
    ### Setting variances for data and w if not provided
    #### Default var=1

    if var_data is None:
        var_data = []
        for m in range(M):
            var_data.append(np.ones((N, D[m])))

    if var_weights is None:
        var_weights = []
        for m in range(M):
            var_weights.append(np.ones((D[m], K)))


    ### Setting weights scale to 1, if not provided
    #### This is the parameter to scale down/up the weights to test their influence on the model 
    #### Especially needed to test a link FA - CTM 
            
    if muFA_scale is None:
        muFA_scale = np.ones((M))


    ### Sampling in which view which factors are/are not active

    sparsity_factor_number = np.floor(sparsity_factor_fraction*K*M).astype('int')
    sparsity_factor_not_set = True

    while check_sparsity_factor or sparsity_factor_not_set:
        sparsity_factor_indicator_flat = np.random.choice(np.arange(0, np.array(K*M).astype('int')), size=sparsity_factor_number, replace=False)
        sparsity_factor_indicator_indeces = divmod(sparsity_factor_indicator_flat, M)
        check_sparsity_factor = np.any(np.unique(sparsity_factor_indicator_indeces[0], return_counts=True)[1] == M)
        sparsity_factor_not_set = False

    sparsity_factor_indicator = np.zeros((K, M))
    sparsity_factor_indicator[sparsity_factor_indicator_indeces[0], sparsity_factor_indicator_indeces[1]] = 1

    factorwise_sparsity = sparsity_factor_indicator


    ### Sampling weights (+ adding feature-wise sparsity)

    weights = []
    featurewise_sparsity = []
    for m in range(M):

        weights_not_sparse_m = np.random.normal(size=(D[m], K), scale=np.sqrt(var_weights[m]))

        sparsity_feature_number_m = np.floor(sparsity_feature_fraction*D[m]*K).astype('int')
        sparsity_feature_not_set_m = True

        while check_sparsity_feature or sparsity_feature_not_set_m:
            sparsity_feature_indicator_flat_m = np.random.choice(np.arange(0, np.array(D[m]*K).astype('int')), size=sparsity_feature_number_m, replace=False)
            sparsity_feature_indicator_indeces_m = divmod(sparsity_feature_indicator_flat_m, D[m])
            check_sparsity_feature = np.any(np.unique(sparsity_feature_indicator_indeces_m[0], return_counts=True)[1] == D[m])
            sparsity_feature_not_set_m = False

        sparsity_feature_indicator_m = np.zeros((D[m], K))
        sparsity_feature_indicator_m[sparsity_feature_indicator_indeces_m[1], sparsity_feature_indicator_indeces_m[0]] = 1

        weights_sparse_m = weights_not_sparse_m * (1 - sparsity_feature_indicator_m)

        weights.append(weights_sparse_m)
        featurewise_sparsity.append(sparsity_feature_indicator_m)


    ### Sampling hidden factors
        
    z = np.random.normal(size=(N, K))


    ### Sampling views (observed data)

    fa_data = dict()
    for m in range(M):
        fa_data_m = muFA_scale[m]*np.random.normal(np.dot(z, weights[m].T), np.sqrt(var_data[m]))
        if data_centered:
            fa_data_m = (fa_data_m - np.mean(fa_data_m, axis=0))
        fa_data['M'+str(m)] = fa_data_m


    ### Saving information which would not be observed for real data
        
    artificial_data_info = {'z': z,
                            'w': weights,
                            'featurewise_sparsity': featurewise_sparsity,
                            'factorwise_sparsity': factorwise_sparsity}
    
    params = {'K': K,
              'N': N, 
              'D': D,           
              'M': M}
    
    if write_all:
        params['var_weights'] = np.array([var_weights[m][0,:] for m in range(M)])
        params['var_data'] = [var_data[m][0,:] for m in range(M)]
    
    artificial_data_info['params'] = params

    return fa_data, artificial_data_info


def simulations_sample_ctm(N, G, L, 
                           sentences_per_observation=None, sentences_per_observation_params=None, 
                           words_per_sentence=None,
                           topics_params=None,
                           FA=False, 
                           mu0=None, muFA=None, Sigma0=None, seed=123):
    
    np.random.seed(seed)
    
    ### Sampling topics (distributions on words) from 
    #### The default is symmetric 
    ####  and is chosen in a way that the topics are sparse (alpha < 1) but not to sparse (alpha > 1/G)

    if topics_params is None:
        topics_params = np.ones(G)/np.sqrt(G)   

    topics = np.random.dirichlet(topics_params, size=L)


    ### Setting number of sentences in each observation
    #### That number might be constant or chosen randomly (so the number of sentences varie across samples)

    if sentences_per_observation is None:
        sentences = 10*np.ones(N)
    if sentences_per_observation == 'constant':
        sentences = sentences_per_observation_params['const']*np.ones(N)
    if sentences_per_observation == 'negative_binomial':
        sentences = np.random.negative_binomial(n=sentences_per_observation_params['n'], p=sentences_per_observation_params['p'], size=N) + 1
    if sentences_per_observation == 'poisson':
        sentences = np.random.poisson(lam=sentences_per_observation_params['lambda'], size=N) + 1
    sentences = sentences.astype('int')
    words_per_sentence = int(words_per_sentence)


    ### Topics proportion in each observation
    #### Set via mu0, the default is that all probabilities are equal
    #### If FA = True, then the proportion varies across observations

    if mu0 is None:
        mu0 = np.zeros(L)
    if Sigma0 is None:
        Sigma0 = np.eye(L)
    if muFA is None:
        if FA:
            muFA = np.random.normal(size=(N, L))
        else:
            muFA = np.zeros((N, L))
    mu_per_observation = muFA + mu0
    eta_per_observation = np.zeros((N,L))
    for n in range(N):
        eta_per_observation[n,:] = np.random.multivariate_normal(mu_per_observation[n,:], Sigma0)
        
    topics_proportion_observation = np.exp(eta_per_observation)
    topics_proportion_observation = topics_proportion_observation/np.outer(np.sum(topics_proportion_observation, axis=1), np.ones(L))


    ### Generating data
        
    ctm_data = []
    topics_per_sentence = []
    topics_proportion_per_observation = []

    for n in range(N):
        topics_ind_n = np.random.choice(int(L), p=topics_proportion_observation[n,:], size=int(sentences[n]))

        topics_proportion_per_observation.append(topics_proportion_observation[n,:])
        topics_per_sentence.append(topics_ind_n)

        ctm_data_n = np.zeros((int(sentences[n]), G))
        for i in range(int(sentences[n])):
            for j in range(words_per_sentence):
                ctm_data_n[i, np.random.choice(int(G), size=1, p=topics[topics_ind_n[i],:])] += 1
        ctm_data.append(ctm_data_n)


    ### Saving information which would not be observed for real 
    
    artificial_data_info = {'topics': topics,
                            'sentences': sentences,
                            'muFA': muFA,
                            'eta': eta_per_observation,
                            'topics_proportion_per_observation': topics_proportion_per_observation,
                            'topics_per_sentence': topics_per_sentence}

    return ctm_data, artificial_data_info

def simulations_sample_factm(which_ctm, simulations_sample_fa_params, simulations_sample_ctm_params, seed):

    np.random.seed(seed)

    if sum(which_ctm) != 1:
        pass
        #Error('There should be just one view')

    factm_data = {}
    fa_data, fa_artificial_data_info = simulations_sample_fa(**simulations_sample_fa_params, seed=seed)

    for m in range(simulations_sample_fa_params['M']):

        if which_ctm[m]:
            ctm_data, ctm_artificial_data_info = simulations_sample_ctm(muFA = fa_data['M'+str(m)], **simulations_sample_ctm_params, seed=seed)
            factm_data['M'+str(m)] = ctm_data
        else:
            factm_data['M'+str(m)] = fa_data['M'+str(m)]


    return factm_data, fa_artificial_data_info, ctm_artificial_data_info
    

