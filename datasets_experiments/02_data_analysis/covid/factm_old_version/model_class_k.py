from tqdm import tqdm
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import MinMaxScaler
from FACTM_model_k import *


class FACTModel(FACTM):
    def __init__(self, data, K, O, L, S, seed, CTM_t, *args, **kwargs):
        self.seed = seed
        np.random.seed(self.seed)
        M = len(data)
        N = data['M'+str(np.where(O)[0][0])].shape[0]
        D = []
        G = []

        # count structered modalities (CTMs)
        m_ctm = 0
        for m in range(M):
            if O[m]:
                D.append(data['M'+str(m)].shape[1])
            else:
                D.append(L[m_ctm])
                m_ctm += 1
                G.append(data['M'+str(m)][0].shape[1])

        super(FACTModel, self).__init__(data, N, M, K, D, G, O, S, CTM_t=CTM_t, *args, **kwargs)

        self.__CTM_t = CTM_t

        self.__first_fit = True
        self.elbo_sequence = []

    def pretrain(self, FA_pretrain='PCA', CTM_pretrain='CTM'):

        FA_pretrain_modalities = self.O

        if CTM_pretrain == 'CTM':
            FA_pretrain_modalities = [True for m in range(self.M)]
            for m in range(self.M):
                if not self.O[m]:
                    print('Pretrain CTM part for a modality '+str(m))

                    mod_ctm = create_CTM()
                    mod_ctm_tmp = mod_ctm(self.ctm_list['M'+str(m)].node_y.data, self.N, self.ctm_list['M'+str(m)].L,
                                          self.ctm_list['M'+str(m)].G, self.fa.K, None, CTM_t=self.__CTM_t)
                    # print(self.ctm_list['M'+str(m)].node_y.data)
                    # modCTM = CTModel(self.ctm_list['M'+str(m)].node_y.data, self.ctm_list['M'+str(m)].L, 123)
                    mod_ctm_tmp.MB()
                    for i in range(50):
                        mod_ctm_tmp.update()

                    self.ctm_list['M'+str(m)] = mod_ctm_tmp
                    self.ctm_list['M'+str(m)].node_muFA.vi_mu = self.ctm_list['M'+str(m)].node_eta.vi_mu - self.ctm_list['M'+str(m)].node_mu0.mu0

        print('Pretrain FA part')
        if FA_pretrain == 'PCA':
            modFA = PCA(n_components=self.fa.K, whiten=True)
        if FA_pretrain == 'FA':
            modFA = FactorAnalysis(n_components=self.fa.K)

        # scaled and centered data
        if CTM_pretrain == 'CTM':
            data_tmp = []
            for m in range(self.M):
                if self.O[m]:
                    data_tmp_m = self.fa.nodelist_y[m].data
                    # data_tmp.append((self.fa.nodelist_y[m].data - np.mean(self.fa.nodelist_y[m].data, axis=0))/np.std(self.fa.nodelist_y[m].data, axis=0))
                else:
                    data_tmp_m = self.ctm_list['M'+str(m)].node_muFA.vi_mu
                data_tmp.append((data_tmp_m - np.mean(data_tmp_m, axis=0))/np.std(data_tmp_m, axis=0))
            data_tmp = np.hstack(data_tmp)
        else:
            data_tmp = np.hstack([(self.fa.nodelist_y[m].data - np.mean(self.fa.nodelist_y[m].data, axis=0))/np.std(self.fa.nodelist_y[m].data, axis=0) for m in range(self.M) if self.O[m]])
        D_tmp = [self.D[m] for m in range(self.M) if FA_pretrain_modalities[m]]

        modFA.fit(data_tmp)

        views_segments = [0] + np.cumsum(np.array(D_tmp)).tolist()
        loadings_tmp = modFA.components_.T
        latent_factors_tmp = modFA.transform(data_tmp)

        m_fa = 0
        for m in range(self.M):
            if FA_pretrain_modalities[m]:
                # get weights + scale back according to the variance of the features
                self.fa.nodelist_hat_w[m].vi_mu = (np.std(self.fa.nodelist_y[m].data, axis=0)*loadings_tmp[views_segments[m_fa]:views_segments[m_fa+1],:].T).T
                m_fa += 1

        # scale to [-1, 1] as in MOFA
        min_max_scaler = MinMaxScaler((-1, 1))
        self.fa.node_z.vi_mu = min_max_scaler.fit_transform(latent_factors_tmp)


    def fit(self, num_iter, delete_factors_with_low_explained_variance=False, tres0=-0.01, tres1=-np.inf):

        if self.__first_fit:
            self.MB()
            if delete_factors_with_low_explained_variance:
                print('Deleting unwanted factors')
                # in each update up to one factor can be deleted
                # after each deleting there is one update without deleting
                # in this first step we delete only really bad factors (tresh < 0)
                self.elbo_sequence_deleting_factors = []
                self.update()
                self.ELBO()
                self.elbo_sequence_deleting_factors.append(self.get_elbo())
                last_deleted = False

                for b in tqdm(range(2*self.fa.K)):
                    self.update()
                    self.ELBO()
                    self.elbo_sequence_deleting_factors.append(self.get_elbo())
                    tres_tmp = tres0 + b*(-tres0 + tres1)/(2*self.fa.K)
                    if not last_deleted:
                        last_deleted = self.fa.delete_inactive(tres=tres_tmp)
                    else:
                        last_deleted = False
                        if (self.fa.variance_explained_per_factor() < 0).all():
                            print("Probably the starting number of factors was way too high.")

        self.__first_fit = False
        
        print('Fitting a model')
        for iter in tqdm(range(num_iter)):
            self.update()
            self.ELBO()
            self.elbo_sequence.append(self.get_elbo())
            self.fa.delete_inactive(tres=tres1)

    # Point estimators
    def get_pe_latent_factors(self):
        return self.fa.node_z.vi_mu
    
    def get_pe_loadings_dense(self, m):
        return self.fa.nodelist_hat_w[m].vi_mu

    def get_pe_loadings_sparse(self, m):
        return self.fa.nodelist_hat_w[m].E_w
    
    def get_pe_featurewise_sparsity(self, m):
        return self.fa.nodelist_s[m].vi_gamma < 0.5
    
    def get_pe_muFA(self, m):
        return self.ctm_list['M'+str(m)].node_muFA.vi_mu
    
    def get_mu0(self, m):
        return self.ctm_list['M'+str(m)].node_mu0.mu0
    
    def get_Sigma0(self, m):
        return self.ctm_list['M'+str(m)].node_Sigma0.Sigma0
    
    def get_pe_topics(self, m):
        L_m = self.L_M[np.where(np.array(self.index_CTM) == m)[0][0]]
        return np.array([self.ctm_list['M'+str(m)].node_beta.vi_alpha[l,:]/np.sum(self.ctm_list['M'+str(m)].node_beta.vi_alpha[l, :]) for l in range(L_m)])
    
    def get_pe_eta(self, m):
        return self.ctm_list['M'+str(m)].node_eta.vi_mu
    
    def get_pe_eta_probabilities_of_topics(self, m):
        L_m = self.L_M[np.where(np.array(self.index_CTM) == m)[0][0]]
        prob_est = np.exp(self.get_pe_eta(m))
        prob_est = prob_est/np.outer(np.sum(prob_est, axis=1), np.ones(L_m))
        return prob_est
    
    def get_pe_probabilities_of_topics(self, m):
        return self.ctm_list['M'+str(m)].node_xi.vi_par
    
    def get_pe_clusters(self, m):
        return [np.argmax(self.ctm_list['M'+str(m)].node_xi.vi_par[n], axis=1) for n in range(self.N)]   


CTM_ = create_CTM()

class CTModel(CTM_):
    def __init__(self, data, L, seed, *args, **kwargs):
        self.seed = seed
        np.random.seed(self.seed)
        N = len(data)
        G = data[0].shape[1]

        CTM_.__init__(self, data, N, L, G, K=2, starting_params = None, FA=False, *args, **kwargs)

        self.__first_fit = True
        self.elbo_sequence = []

    def fit(self, num_iter):
        if self.__first_fit:
            self.MB()

        self.__first_fit = False
        
        print('Fitting a model')
        for iter in tqdm(range(num_iter)):
            self.update()
            self.ELBO()
            self.elbo_sequence.append(self.get_elbo())

    def pretrain(self):
        pass

    # Point estimators    
    def get_mu0(self):
        return self.node_mu0.mu0
    
    def get_Sigma0(self):
        return self.node_Sigma0.Sigma0
    
    def get_pe_topics(self):
        return np.array([self.node_beta.vi_alpha[l,:]/np.sum(self.node_beta.vi_alpha[l, :]) for l in range(self.L)])
    
    def get_pe_eta(self):
        return self.node_eta.vi_mu
    
    def get_pe_eta_probabilities_of_topics(self, m):
        L = self.L
        prob_est = np.exp(self.get_pe_eta())
        prob_est = prob_est/np.outer(np.sum(prob_est, axis=1), np.ones(L))
        return prob_est
    
    def get_pe_probabilities_of_topics(self):
        return self.node_xi.vi_par
    
    def get_pe_clusters(self, m):
        return [np.argmax(self.node_xi.vi_par[n], axis=1) for n in range(self.N)]   
        