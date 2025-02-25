import numpy as np
from scipy.optimize import minimize
from scipy.special import beta, gamma, digamma, gammaln

EPS = 1e-200

def create_CTM():

    class nodeCTM_general():
        N = None
        L = None
        G = None
        I = None
        J = None
        def __init__(self, N, L, G, I, J):
            nodeCTM_general.L = L
            nodeCTM_general.N = N
            nodeCTM_general.G = G
            nodeCTM_general.I = I
            nodeCTM_general.J = J


    class nodeCTM_Sigma0(nodeCTM_general):
        def __init__(self, Sigma0):
            self.Sigma0 = Sigma0

            self.inv_Sigma0 = np.linalg.inv(self.Sigma0)
            self.det_Sigma0 = np.linalg.det(self.Sigma0)
        
        def MB(self, eta_node, muFA_node, mu0_node):
            self.eta_node = eta_node
            self.muFA_node = muFA_node
            self.mu0_node = mu0_node
        
        def update(self):
            centered_mean = self.eta_node.vi_mu  - self.mu0_node.mu0 - self.muFA_node.vi_mu

            self.Sigma0 = np.dot(centered_mean.T, centered_mean)/self.N + np.diag(np.mean(self.eta_node.vi_var, axis=0)) + self.muFA_node.vi_Sigma
            
            self.inv_Sigma0 = np.linalg.inv(self.Sigma0)
            self.det_Sigma0 = np.linalg.det(self.Sigma0)


    class nodeCTM_mu0(nodeCTM_general):
        def __init__(self, mu0):
            self.mu0 = mu0

        def MB(self, eta_node, muFA_node):
            self.eta_node = eta_node
            self.muFA_node = muFA_node

        def update(self):
            self.mu0 = np.mean(self.eta_node.vi_mu - self.muFA_node.vi_mu, axis=0)
            

    class nodeCTM_w_z(nodeCTM_general):
        def __init__(self, E_w, E_w_squared, E_z, E_z_squared):
            self.E_w = E_w
            self.E_w_squared = E_w_squared
            self.E_z = E_z
            self.E_z_squared = E_z_squared
            self.E_w_z = np.dot(self.E_w, self.E_z.T)
            self.E_w_z_squared = None

        def MB(self):
            pass

        def update(self):
            pass

        def ELBO(self):
            pass


    class nodeCTM_t(nodeCTM_general):
        def __init__(self, t):
            self.t = t*np.ones(self.L)

            self.inv_t = 1/self.t

        def MB(self):
            pass

        def update(self):
            pass

        def ELBO(self):
            pass
            

    class nodeCTM_muFA(nodeCTM_general):
        def __init__(self, vi_mu, vi_Sigma):
            self.vi_mu = vi_mu
            self.vi_Sigma = vi_Sigma

            self.det_vi_Sigma = np.linalg.det(self.vi_Sigma)

            self.elbo = 0

        def MB(self, t_node, mu0_node, Sigma0_node, w_z_node, eta_node): 
            self.t_node = t_node
            self.mu0_node = mu0_node
            self.Sigma0_node = Sigma0_node
            self.w_z_node = w_z_node
            self.eta_node = eta_node

        def update(self):
            self.vi_Sigma = np.linalg.inv(np.diag(self.t_node.t) + self.Sigma0_node.inv_Sigma0)
            self.det_vi_Sigma = np.linalg.det(self.vi_Sigma)

            first_summand_tmp = np.dot(np.diag(self.t_node.t), self.w_z_node.E_w_z)
            second_summand_tmp = np.dot(self.Sigma0_node.inv_Sigma0, (self.eta_node.vi_mu - self.mu0_node.mu0).T)
            self.vi_mu = np.dot(self.vi_Sigma, (first_summand_tmp + second_summand_tmp)).T

        def ELBO(self):
            
            elbo = self.N*self.L/2  + self.N*np.sum(np.log(self.t_node.t))/2 \
                    - np.sum(self.t_node.t * (self.N*np.diag(self.vi_Sigma) + np.sum(self.vi_mu**2, axis=0) \
                                                - 2*np.sum(self.vi_mu*self.w_z_node.E_w_z.T, axis=0) + np.sum(self.w_z_node.E_w_z_squared, axis=0))/2) \
                                                    + self.N*np.log(self.det_vi_Sigma)/2
            
            self.elbo = elbo


    def f_eta_par_n(vi_eta_mu_n, vi_eta_var_n,
                    vi_muFA_mu_n, mu, Sigma_inv,
                    vi_xi_par_n, 
                    vi_zeta_n, I):
        
        term_xi = np.sum(vi_xi_par_n, axis=0)

        return -np.sum(vi_eta_var_n*np.diag(Sigma_inv))/2 \
            - np.sum(np.dot(vi_eta_mu_n - vi_muFA_mu_n - mu, Sigma_inv)*(vi_eta_mu_n - vi_muFA_mu_n - mu))/2 \
                +np.sum(vi_eta_mu_n*term_xi) - (I/vi_zeta_n)*np.sum(np.exp(vi_eta_mu_n + vi_eta_var_n/2)) \
                + np.sum(np.log(vi_eta_var_n))/2

    def fgrad_eta_par_n(vi_eta_mu_n, vi_eta_var_n,
                        vi_muFA_mu_n, mu, Sigma_inv,
                        vi_xi_par_n, 
                        vi_zeta_n, I):
        
        term_xi = np.sum(vi_xi_par_n, axis=0)
        term_MGF = (I/vi_zeta_n)*(np.exp(vi_eta_mu_n + vi_eta_var_n/2))
        
        grad_mu = - np.dot(Sigma_inv, vi_eta_mu_n - vi_muFA_mu_n - mu) + term_xi - term_MGF
        
        grad_var = - np.diag(Sigma_inv)/2 - term_MGF/2 + 1/(2*vi_eta_var_n)

        grad = np.concatenate((grad_mu, grad_var))
        return grad

    class nodeCTM_eta(nodeCTM_general):
        def __init__(self, vi_mu, vi_var, vi_zeta):
            self.vi_mu = vi_mu
            self.vi_var = vi_var
            self.vi_zeta = vi_zeta

            self.E_exp_eta = np.exp(self.vi_mu + self.vi_var/2)

            self.elbo = 0

        def MB(self, muFA_node, mu0_node, Sigma0_node, xi_node):
            self.muFA_node = muFA_node
            self.mu0_node = mu0_node
            self.Sigma0_node = Sigma0_node
            self.xi_node = xi_node

        def update(self):

            for n in range(self.N):

                self.vi_zeta[n] = np.sum(self.E_exp_eta[n,:])

                f = lambda x: -f_eta_par_n(x[:self.L], x[self.L:], 
                                        self.muFA_node.vi_mu[n,:], self.mu0_node.mu0, self.Sigma0_node.inv_Sigma0,
                                        self.xi_node.vi_par[n], 
                                        self.vi_zeta[n], self.I[n])

            
                fgrad = lambda x: -fgrad_eta_par_n(x[:self.L], x[self.L:], 
                                                self.muFA_node.vi_mu[n,:], self.mu0_node.mu0, self.Sigma0_node.inv_Sigma0,
                                                self.xi_node.vi_par[n], 
                                                self.vi_zeta[n], self.I[n])
            
                starting_point = np.concatenate((1.0*self.vi_mu[n,:], 1.0*self.vi_var[n,:]))    

                # a condition that variances are non-negative
                bnds = tuple(map(lambda x : (None, None) if x < self.L else (EPS, None), range(2*self.L)))

                result = minimize(f, x0=starting_point, method='L-BFGS-B',  jac=fgrad,  options={'disp': 0}, bounds=bnds)

                self.vi_mu[n,:] = result.x[:self.L]
                self.vi_var[n,:] = result.x[self.L:]

                self.E_exp_eta[n,:] = np.exp(self.vi_mu[n,:] + self.vi_var[n,:]/2)

        def ELBO(self):

            entropy = np.sum(np.log(self.vi_var))/2 + self.N*self.L/2

            term_mean_difference = self.vi_mu - self.muFA_node.vi_mu - self.mu0_node.mu0

            kl = - self.N*np.log(self.Sigma0_node.det_Sigma0)/2 - np.sum(np.diag(self.Sigma0_node.inv_Sigma0)*self.vi_var)/2 \
                - np.sum(term_mean_difference * np.dot(term_mean_difference, self.Sigma0_node.inv_Sigma0))/2 
            #- np.sum(np.dot(term_mean_difference, np.dot(self.Sigma0_node.inv_Sigma0, term_mean_difference.T)))/2 
            
            self.elbo = kl + entropy


    class nodeCTM_xi(nodeCTM_general):

        def __init__(self, vi_par):

            self.vi_par = vi_par

            self.vi_log_par = [np.log(vi_par[i]) for i in range(len(vi_par))]

            self.elbo = 0

        def MB(self, eta_node, beta_node, data):
            self.eta_node = eta_node
            self.beta_node = beta_node
            self.y_node = data

        def update(self):

            term_E_log_beta = self.beta_node.digamma_vi_alpha.T - self.beta_node.digamma_sum_vi_alpha

            for n in range(self.N):

                vi_par_n = np.zeros((self.I[n], self.L))
                vi_log_par_n = np.zeros((self.I[n], self.L))
                
                vi_log_par_n = self.eta_node.vi_mu[n,:] + np.dot(self.y_node.data[n], term_E_log_beta)
                vi_log_par_n = vi_log_par_n - np.outer(np.max(vi_log_par_n, axis=1), np.ones(self.L))
                vi_par_n = np.exp(vi_log_par_n)

                norm_cons_tmp = np.outer(np.sum(vi_par_n, axis = 1), np.ones(self.L))
                vi_par_n = vi_par_n/norm_cons_tmp
                vi_log_par_n = vi_log_par_n - np.log(norm_cons_tmp)

                self.vi_log_par[n] = vi_log_par_n
                self.vi_par[n] = vi_par_n

        def ELBO(self):

            kl = 0
            entropy = 0
            for n in range(self.N):
                entropy += -np.sum(self.vi_par[n] * self.vi_log_par[n])
                kl += np.sum(self.eta_node.vi_mu[n,:] * self.vi_par[n]) \
                    - self.I[n]*(np.log(self.eta_node.vi_zeta[n]) + np.sum(self.eta_node.E_exp_eta[n,:])/self.eta_node.vi_zeta[n] - 1)
                #kl += np.sum(self.eta_node.vi_mu[n,:] * self.vi_log_par[n]) \
                #    - self.I[n]*(np.log(self.eta_node.vi_zeta[n]) + np.sum(self.eta_node.vi_mu[n,:])/self.eta_node.vi_zeta[n] - 1)

            self.elbo = kl + entropy


    class nodeCTM_beta(nodeCTM_general):

        def __init__(self, alpha, vi_alpha):
            # N x L x G
            self.alpha = alpha
            self.vi_alpha = vi_alpha

            self.lnGamma_sum_vi_alpha = gammaln(np.sum(self.vi_alpha, axis=1))
            self.sum_lnGamma_alpha = self.G*gammaln(self.alpha)

            self.lnGamma_sum_alpha = gammaln(self.G*self.alpha)
            self.sum_lnGamma_vi_alpha = np.sum(gammaln(self.vi_alpha), axis=1)

            self.digamma_vi_alpha = digamma(self.vi_alpha)
            self.digamma_sum_vi_alpha = digamma(np.sum(self.vi_alpha, axis=1))

            self.elbo = 0

        def MB(self, xi_node, y_node):
            self.xi_node = xi_node
            self.y_node = y_node

        def update(self):

            vi_alpha = self.alpha*np.ones((self.L, self.G))

            for n in range(self.N):
                vi_alpha += np.dot(self.xi_node.vi_par[n].T, self.y_node.data[n])
            
            self.vi_alpha = vi_alpha

            sum_alpha_tmp = np.sum(vi_alpha, axis=1)

            self.lnGamma_sum_vi_alpha = gammaln(sum_alpha_tmp)
            self.sum_lnGamma_vi_alpha = np.sum(gammaln(vi_alpha), axis=1)

            self.digamma_vi_alpha = digamma(self.vi_alpha)
            self.digamma_sum_vi_alpha = digamma(sum_alpha_tmp)

        def ELBO(self):
            elbo = 0
            for l in range(self.L):
                elbo += self.lnGamma_sum_vi_alpha[l] - self.lnGamma_sum_alpha  \
                    - self.sum_lnGamma_vi_alpha[l] + self.sum_lnGamma_alpha \
                        + np.sum((self.vi_alpha[l,:] - self.alpha)*(self.digamma_vi_alpha[l,:] - self.digamma_sum_vi_alpha[l]))
                
            self.elbo = -elbo


    class nodeCTM_y(nodeCTM_general):

        def __init__(self, data):
            self.data = data

            self.elbo = 0

        def MB(self, beta_node, xi_node):
            self.beta_node = beta_node
            self.xi_node = xi_node

        def ELBO(self):

            likelihood = 0

            for n in range(self.N):
                for i in range(self.I[n]):
                    data_i = self.data[n][i,:]
                    for l in range(self.L):

                        term_E_log_beta_l = self.beta_node.digamma_vi_alpha[l,:] - self.beta_node.digamma_sum_vi_alpha[l]

                        likelihood += self.xi_node.vi_par[n][i,l] * np.sum(data_i * term_E_log_beta_l)
            self.elbo = likelihood


    def starting_params_Sigma(starting_params, L):

        if 'Sigma' in starting_params.keys():
            Sigma = 1*starting_params['Sigma']
        else:
            Sigma = np.eye(L)

        return Sigma


    def starting_params_mu(starting_params, L):

        if 'mu' in starting_params.keys():
            mu = 1*starting_params['mu']
        else:
            mu = np.zeros(L)

        return mu


    def starting_params_beta(starting_params, L, G):

        if 'topics' in starting_params.keys():
            topics = 1*starting_params['topics']
            # print("beta")
        else:
            # par=100*1 so the distribution is close to uniform but not uniform
            topics = np.random.dirichlet(100*np.ones(G), size=L)

        return topics


    class CTM():
        def __init__(self, data, N, L, G, K, starting_params = None, FA=True, CTM_t=1, *args, **kwargs):
            self.N=N
            self.L=L
            self.G=G
            self.data=data

            # If FA=True we use FACTM, if FA=False simple CTM is fitted
            self.FA = FA

            if starting_params is None:
                starting_params = {}

            I = []
            J = []
            init_y_data = []
            init_xi_par = []

            for n in range(N):
                data_n = data[n]

                I_n = data_n.shape[0]
                J_n = np.sum(data_n, axis=1)
                
                I.append(I_n)
                J.append(J_n)

                init_y_data.append(data_n)
                init_xi_par.append(np.ones((I_n, L))/L)

            Sigma0 = starting_params_Sigma(starting_params, self.L)
            self.node_Sigma0 = nodeCTM_Sigma0(Sigma0)

            nodeCTM_general(N, L, G, I, J)

            mu0 = starting_params_mu(starting_params, self.L)
            self.node_mu0 = nodeCTM_mu0(mu0) 
            topics = starting_params_beta(starting_params, self.L, self.G)
            self.node_beta = nodeCTM_beta(1e-5, topics)
            self.node_t = nodeCTM_t(t=CTM_t)

            

            if self.FA:
                self.node_w_z = nodeCTM_w_z(np.ones((L, K)), np.ones((L, K)), np.ones((N, K)), np.ones((N, K)))
            else:
                self.node_w_z = nodeCTM_w_z(np.zeros((L, K)), np.zeros((L, K)), np.zeros((N, K)), np.zeros((N, K)))
                self.node_w_z.E_w_z_squared = np.zeros((L, K))
            if self.FA:
                self.node_muFA = nodeCTM_muFA(np.zeros((self.N, self.L)), np.eye(L))
            else:
                self.node_muFA = nodeCTM_muFA(np.zeros((self.N, self.L)), np.zeros((L,L)))
            self.node_eta = nodeCTM_eta(np.random.normal(size=(N, L))/10, np.ones((N, L)), np.ones(N))
            self.node_xi = nodeCTM_xi(init_xi_par)
            self.node_y = nodeCTM_y(init_y_data)

            self.elbo = 0

        def MB(self):
            
            self.node_t.MB()
            self.node_mu0.MB(self.node_eta, self.node_muFA)
            self.node_Sigma0.MB(self.node_eta, self.node_muFA, self.node_mu0)
            self.node_beta.MB(self.node_xi, self.node_y)
            self.node_muFA.MB(self.node_t, self.node_mu0, self.node_Sigma0, self.node_w_z, self.node_eta) 
            self.node_eta.MB(self.node_muFA, self.node_mu0, self.node_Sigma0, self.node_xi)
            self.node_xi.MB(self.node_eta, self.node_beta, self.node_y)
            self.node_y.MB(self.node_beta, self.node_xi)


        def update(self):

            # step E
            self.node_xi.update()
            self.node_eta.update()
            if self.FA:
                self.node_muFA.update()

            # step M
            self.node_beta.update()
            self.node_mu0.update()
            self.node_Sigma0.update()


        def get_params(self):

            list_vi_params = {
                'mu': self.node_mu0.mu0,
                'Sigma': self.node_Sigma0.Sigma0,
                'beta_par': self.node_beta.vi_alpha,
                't': self.node_t.t,
                'eta_mu': self.node_eta.vi_mu,
                'eta_var': self.node_eta.vi_var,
                'muFa_mu': self.node_muFA.vi_mu,
                'muFa_Sigma': self.node_muFA.vi_Sigma,
                'xi_par': self.node_xi.vi_par,
                'zeta': self.node_eta.vi_zeta
            }

            return list_vi_params
        
        def ELBO(self):

            self.node_xi.ELBO()
            self.node_eta.ELBO()
            if self.FA:
                self.node_muFA.ELBO()
            self.node_beta.ELBO()
            self.node_y.ELBO()

            elbo = self.node_xi.elbo + self.node_eta.elbo + self.node_muFA.elbo + self.node_beta.elbo + self.node_y.elbo

            self.elbo = elbo

        def get_elbo(self):

            return self.elbo
        
        def get_elbo_per_node(self):

            return (self.node_xi.elbo, self.node_eta.elbo, self.node_muFA.elbo, self.node_beta.elbo, self.node_y.elbo)
        
        def get_BIC(self):
            par_ctm = self.L + (self.L + 1)*self.L/2 + self.L*self.G
            loglikelihood = self.node_y.elbo
            N_sentence = np.sum(np.array([self.node_y.data[n].shape[0] for n in range(self.N)]))
            N_word = np.sum(np.array([np.sum(self.node_y.data[n]) for n in range(self.N)]))
            return -2*loglikelihood + par_ctm*np.log(np.array([self.N, N_sentence, N_word]))
        
        def CTM_new_data(self, new_data, new_N, ctm, K):
            self.ctm_new_data = ctm
            self.ctm_new_data.N = new_N
            self.ctm_new_data.data = new_data

            I = []
            J = []
            init_y_data = []
            init_xi_par = []

            for n in range(new_N):
                data_n = new_data[n]

                I_n = data_n.shape[0]
                J_n = np.sum(data_n, axis=1)
                
                I.append(I_n)
                J.append(J_n)

                init_y_data.append(data_n)
                # init_xi_par.append(np.ones((I_n, self.L))/self.L)
                init_xi_par.append(np.random.dirichlet(np.ones(self.L), size=(I_n)))
            
            self.ctm_new_data.I = I
            self.ctm_new_data.J = J

            if self.FA:
                pass
                #self.CTM_new_data.node_w_z = nodeCTM_w_z(np.ones((self.L, self.K)), np.ones((self.L, self.K)), np.ones((new_N, self.K)), np.ones((new_N, self.K)))
            else:
                self.ctm_new_data.node_w_z.E_w = np.zeros((self.L, K))
                self.ctm_new_data.node_w_z._w_squared = np.zeros((self.L, K))
                self.ctm_new_data.node_w_z.E_z = np.zeros((new_N, K))
                self.ctm_new_data.node_w_z.E_z_squared = np.zeros((new_N, K))
                self.ctm_new_data.node_w_z.E_w_z_squared = np.zeros((self.L, K))
            if self.FA:
                pass
                #self.node_muFA.vi_mu = np.zeros((new_N, self.L))
                #self.node_muFA.vi_Sigma = np.eye(self.L)
            else:
                self.ctm_new_data.node_muFA.vi_mu = np.zeros((new_N, self.L))
                self.ctm_new_data.node_muFA.vi_Sigma = np.zeros((self.L,self.L))
            self.ctm_new_data.node_eta.vi_mu = np.random.normal(size=(new_N, self.L))/10
            self.ctm_new_data.node_eta.vi_var = np.ones((new_N, self.L))
            self.ctm_new_data.node_eta.vi_zeta = np.ones(new_N)
            self.ctm_new_data.E_exp_eta = np.exp(self.ctm_new_data.node_eta.vi_mu + self.ctm_new_data.node_eta.vi_var/2)

            self.ctm_new_data.node_xi.vi_par = init_xi_par
            self.ctm_new_data.node_xi.vi_log_par = [np.log(init_xi_par[i]) for i in range(len(init_xi_par))]
            self.ctm_new_data.node_y.data = init_y_data

            ### i dont want to do it
            self.ctm_new_data.node_xi.N = new_N
            self.ctm_new_data.node_xi.I = I

            self.ctm_new_data.node_eta.N = new_N
            self.ctm_new_data.node_eta.I = I
            self.ctm_new_data.node_eta.J = J

            self.ctm_new_data.node_muFA.N = new_N
            self.ctm_new_data.node_muFA.I = I

            self.ctm_new_data.node_y.N = new_N
            self.ctm_new_data.node_y.I = I

    return CTM





