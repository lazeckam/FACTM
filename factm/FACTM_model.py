# Class Definition for Factor Analysis and Correlated Topic Model (FACTM)

# This class defines and fits a Factor Analysis and Correlated Topic Model. 
# It joints FA and CTM classes

from factm_ctm import *
from factm_fa import *


class FACTM():
    '''
    Class for fitting the FACTM model, combining Factor Analysis (FA) with the Correlated Topic Model (CTM).
    This class constructs a joint Bayesian network and enables simultaneous updates of all model parameters.
    '''
    def __init__(self, data, N, M, K, D, G, O, starting_params_fa=None, starting_params_ctm=None, CTM_t=1, *args, **kwargs):

        self.N = N
        self.M = M
        self.D = D
        # Numbers of niches (topics) in not observed modalities
        self.L_M = [D[m] for m in range(M) if not O[m]]
        # Numbers of types (e.g. cell types, words) in not observed modalities
        self.G_M = G
        self.O = O

        # create FA 
        self.fa = FA(data, N, M, K, D, O, starting_params_fa)

        # create all CTMs
        self.how_many_ctm = M - np.sum(np.array(O))
        self.index_CTM = np.where(~np.array(self.O))[0]
        self.key_CTM = ['M' + str(m) for m in range(len(D)) if not O[m]]
        if starting_params_ctm is None:
            starting_params_ctm = []
            for m_ctm in range(self.how_many_ctm):
                starting_params_ctm.append({})
        self.ctm_list = dict()
        for m_ctm in range(self.how_many_ctm):
            CTM = create_CTM()
            self.ctm_list[self.key_CTM[m_ctm]] = CTM(data[self.key_CTM[m_ctm]], self.N, self.L_M[m_ctm], self.G_M[m_ctm], K, 
                                                     starting_params_ctm[m_ctm], CTM_t=CTM_t)

            # data is not observed - use muFA.vi_mu values instead
            self.fa.nodelist_y[self.index_CTM[m_ctm]].data = self.ctm_list[self.key_CTM[m_ctm]].node_muFA.vi_mu

            # in CTM tau is not RV - the precision is constant and equals t=1
            self.fa.nodelist_tau[self.index_CTM[m_ctm]].vi_a = None
            self.fa.nodelist_tau[self.index_CTM[m_ctm]].vi_b = None

    def MB(self):

        self.fa.MB()

        for m_ctm in range(self.how_many_ctm):
            self.ctm_list[self.key_CTM[m_ctm]].MB()

    def update(self):

        self.fa.update()

        for m_ctm in range(self.how_many_ctm):
            # update parameters of ctm shared with fa
            self.ctm_list[self.key_CTM[m_ctm]].node_w_z.E_w = self.fa.nodelist_w[self.index_CTM[m_ctm]].E_w
            self.ctm_list[self.key_CTM[m_ctm]].node_w_z.E_w_squared = self.fa.nodelist_w[self.index_CTM[m_ctm]].E_w_squared
            self.ctm_list[self.key_CTM[m_ctm]].node_w_z.E_z = self.fa.node_z.E_z
            self.ctm_list[self.key_CTM[m_ctm]].node_w_z.E_z_squared = self.fa.node_z.E_z_squared
            self.ctm_list[self.key_CTM[m_ctm]].node_w_z.E_w_z = np.dot(self.ctm_list[self.key_CTM[m_ctm]].node_w_z.E_w, self.ctm_list[self.key_CTM[m_ctm]].node_w_z.E_z.T)
            self.ctm_list[self.key_CTM[m_ctm]].node_w_z.E_w_z_squared = self.fa.nodelist_w[self.index_CTM[m_ctm]].E_w_z_squared

            # update ctm params
            self.ctm_list[self.key_CTM[m_ctm]].update()

            # update fa params based on ctm
            self.fa.nodelist_y[self.index_CTM[m_ctm]].data = self.ctm_list[self.key_CTM[m_ctm]].node_muFA.vi_mu
            self.fa.nodelist_tau[self.index_CTM[m_ctm]].E_tau = self.ctm_list[self.key_CTM[m_ctm]].node_t.t
            self.fa.nodelist_tau[self.index_CTM[m_ctm]].E_log_tau = np.log(self.ctm_list[self.key_CTM[m_ctm]].node_t.t)

    def ELBO(self):

        self.fa.ELBO()

        for m_ctm in range(self.how_many_ctm):
            self.ctm_list[self.key_CTM[m_ctm]].ELBO()

    def get_elbo(self):

        return self.fa.elbo + np.sum([self.ctm_list[self.key_CTM[m_ctm]].elbo for m_ctm in range(self.how_many_ctm)])