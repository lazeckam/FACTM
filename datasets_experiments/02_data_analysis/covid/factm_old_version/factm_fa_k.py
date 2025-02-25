import numpy as np
from scipy.special import beta, gamma, digamma, gammaln, betaln
from utils import *
EPS = 1e-20

class nodeFA_general():
    """
    Class to store const.
    """
    N = None
    K = None
    D = None
    M = None
    O = None
    S = None

    def __init__(self, N, K, D, M, O, S):
        nodeFA_general.N = N
        nodeFA_general.K = K
        nodeFA_general.D = D
        nodeFA_general.M = M
        nodeFA_general.O = O
        nodeFA_general.S = S

# BELOW:
# MB - Markov Blanket
# update - updates all the vi params
# update_k - updates vi params of an index k
# update_params - update all auxiliary params, e.g. E_X, E_log_X etc
# update_params_w_z - update auxiliary params that depend on W and Z (and are needed for updates of vi params and useful elsewhere)
# update_all_params - update_params + some const
# ELBO - computes ELBO
class nodeFA_z(nodeFA_general):
    """
    Class to define Z node (n times k)
    """
    def __init__(self, mu0, var0, vi_mu, vi_var):
        self.mu = mu0
        self.var0 = var0

        self.vi_mu = vi_mu
        self.vi_var = vi_var

        self.update_params()
        self.elbo = 0

    def MB(self, y_list, w_list, tau_list):
        self.y_node = y_list
        self.w_node = w_list
        self.tau_node = tau_list

    def update_k(self, k):
        vi_mu_new = np.zeros((self.N, self.K))
        vi_var_new = np.zeros((self.K))

        for m in range(self.M):
            vi_var_new += np.dot(self.tau_node[m].E_tau, self.w_node[m].E_w_squared)
            
            resid = self.y_node[m].data - np.dot(self.w_node[m].E_w, self.E_z.T).T
            partial_resid = (resid + np.outer(self.w_node[m].E_w[:,k], self.E_z[:,k]).T)
            vi_mu_new[:,k] += np.sum(self.tau_node[m].E_tau*self.w_node[m].E_w[:,k]*partial_resid, axis=1)

        vi_var_new = 1/(vi_var_new + 1)
        vi_var_new = np.outer(np.ones(self.N), vi_var_new)
        vi_mu_new = vi_mu_new * vi_var_new

        self.vi_mu[:,k] = vi_mu_new[:,k]
        self.vi_var[:,k] = vi_var_new[:,k]
        self.update_params()

    def update_params(self):
        self.E_z = self.vi_mu
        self.E_z_squared = self.vi_var + self.vi_mu**2
    
    def ELBO(self):
        elbo = self.N*self.K/2 - np.sum(self.E_z_squared)/2 + np.sum(log_eps(self.vi_var))/2
        self.elbo = elbo


class nodeFA_hat_w_m(nodeFA_general):
    """
    Class to define W_hat node (d times k, for one m)
    """
    def __init__(self, vi_mu, vi_var):
        self.vi_mu = vi_mu
        self.vi_var = vi_var

    def MB(self):
        pass

    def update_k(self, k, nominator, denominator, E_tau):
        self.vi_mu[:,k] = nominator/(denominator)
        self.vi_var[:,k] = 1/(E_tau*denominator)


class nodeFA_s_m(nodeFA_general):
    """
    Class to define S node (d times k, for one m)
    """
    def __init__(self, vi_lambda):
        self.vi_gamma = 1/(1 + np.exp(-vi_lambda))
        self.vi_lambda = vi_lambda

    def MB(self):
        pass

    def update_k(self, k, nominator, denominator, E_tau, E_alpha_div_E_Tau, E_log_LR_theta):

        lambda_k = E_log_LR_theta + log_eps(E_alpha_div_E_Tau, EPS)/2 - log_eps(denominator, EPS)/2 \
              + (E_tau*nominator**2)/(2*denominator)
        
        if np.any(lambda_k > -np.log(EPS)):
            lambda_k[lambda_k > -np.log(EPS)] = -np.log(EPS)
        
        self.vi_lambda[:,k] = lambda_k
        self.vi_gamma[:,k] = 1/(1 + np.exp(-lambda_k))


class nodeFA_w_m(nodeFA_general):  
    """
    Class to define W node (d times k, for one m; gathers together the nodes W_hat and S)
    """
    def __init__(self, m):
        # E(hat_w_s) (if S=True) or E(w) (if S=False)
        self.E_w = np.zeros((self.D[m], self.K))
        # E(hat_w_s)^2 (if S=True) or E(w)^2 (if S=False)
        self.E_w_squared = np.ones((self.D[m], self.K))
        # E(hat_w)^2 (if S=True) or E(hat_w)^2 = E(w)^2 (if S=False)
        self.E_hat_w_squared = np.ones((self.D[m], self.K))

        self.m = m

        self.E_w_z = np.zeros((self.N, self.D[m]))
        self.E_w_z_squared = np.zeros((self.N, self.D[m]))

        self.elbo = 0

    def MB(self, hat_w_m_node, s_m_node, alpha_m_node, theta_m_node, z_node, y_m_node, tau_m_node):
        self.hat_w_m_node = hat_w_m_node
        self.s_m_node = s_m_node
        self.alpha_m_node = alpha_m_node
        self.theta_m_node = theta_m_node
        self.z_node = z_node
        self.y_m_node = y_m_node
        self.tau_m_node = tau_m_node

        self.update_params()
        self.update_params_z() # is it ok here?

    def update_k(self, k):
        nominator_second_term_tmp = np.dot(self.E_w, np.dot(self.z_node.E_z.T, self.z_node.E_z[:,k]))
        nominator_second_term = nominator_second_term_tmp - self.E_w[:,k]*np.sum(self.z_node.E_z[:,k]**2)         
        nominator = np.dot(self.z_node.E_z[:,k], self.y_m_node.data) - nominator_second_term

        E_tau = self.tau_m_node.E_tau + 0.0
        E_alpha_div_E_Tau = self.alpha_m_node.E_alpha[k]/E_tau

        denominator = np.sum(self.z_node.E_z_squared[:,k]) + E_alpha_div_E_Tau

        self.hat_w_m_node.update_k(k, nominator, denominator, E_tau)

        if self.O[self.m] and self.S[self.m]: 
            E_log_LR_theta = self.theta_m_node.E_log_LR[k] + 0.0
            self.s_m_node.update_k(k, nominator, denominator, E_tau, E_alpha_div_E_Tau, E_log_LR_theta)

        self.update_params()
        self.update_params_z()

    def update_params(self):
        if self.S[self.m]:
            self.E_w = self.s_m_node.vi_gamma*self.hat_w_m_node.vi_mu
            self.E_w_squared = self.s_m_node.vi_gamma*(self.hat_w_m_node.vi_var + self.hat_w_m_node.vi_mu**2)
            self.E_hat_w_squared = self.s_m_node.vi_gamma*(self.hat_w_m_node.vi_var + self.hat_w_m_node.vi_mu**2) \
                + (1 - self.s_m_node.vi_gamma)*(np.outer(np.ones(self.D[self.m]), self.alpha_m_node.E_inv_alpha))
        else:
            self.E_w = self.hat_w_m_node.vi_mu
            self.E_w_squared = self.hat_w_m_node.vi_var + self.hat_w_m_node.vi_mu**2
            self.E_hat_w_squared = self.E_w_squared + 0.0

    def update_params_z(self):
        self.E_w_z = np.dot(self.E_w, self.z_node.E_z.T).T
        
        term_tmp = np.dot(self.E_w, self.z_node.E_z.T)
        first_term = (term_tmp)**2
        second_term = np.dot(self.E_w_squared, self.z_node.E_z_squared.T)
        third_term = np.dot(self.E_w**2, self.z_node.E_z.T**2)
        self.E_w_z_squared = (first_term + second_term - third_term).T

    def ELBO(self):
        if self.S[self.m]:
            kl = self.D[self.m]*np.sum(self.alpha_m_node.E_log_alpha)/2 - np.sum(np.dot(self.E_hat_w_squared, self.alpha_m_node.E_alpha))/2 \
                + np.sum(np.dot(self.theta_m_node.E_log_theta, self.s_m_node.vi_gamma.T)) + np.sum(np.dot(self.theta_m_node.E_log_1minustheta, (1 - self.s_m_node.vi_gamma).T))
            entropy = self.D[self.m]*self.K/2 + np.sum(self.s_m_node.vi_gamma*log_eps(self.hat_w_m_node.vi_var))/2 - np.sum(np.dot((1-self.s_m_node.vi_gamma), self.alpha_m_node.E_log_alpha))/2 \
                - np.sum(xlogx(self.s_m_node.vi_gamma)) - np.sum(xlogx(1-self.s_m_node.vi_gamma))
        else:
            kl = self.D[self.m]*np.sum(self.alpha_m_node.E_log_alpha)/2 - np.sum(self.alpha_m_node.E_alpha * self.E_w_squared)/2
            entropy = self.D[self.m]*self.K/2 + np.sum(log_eps(self.hat_w_m_node.vi_var))/2
        self.elbo = kl + entropy


class nodeFA_alpha_m(nodeFA_general):
    """
    Class to define alpha node (k, for one m)
    """
    def __init__(self, a0, b0, m):
        self.m = m    

        self.a0 = a0
        self.b0 = b0

        # start from E_alpha = 1
        # update of vi_a:
        self.vi_a = a0 + self.D[m]*np.ones(self.K)/2
        self.vi_b = self.vi_a + 0.0

        self.update_all_params()

        self.elbo = 0

    def MB(self, hat_w_m_node, s_m_node, w_m_node):
        self.hat_w_m_node = hat_w_m_node
        self.s_m_node = s_m_node
        self.w_m_node = w_m_node

    def update_k(self, k):

        self.vi_b[k] = self.b0 + np.sum(self.w_m_node.E_hat_w_squared[:,k])/2

        if self.vi_a[k]/self.vi_b[k] < EPS:
            self.vi_b[k] = self.vi_a[k]/EPS

        self.update_params()

    def update_params(self):
        self.E_alpha = self.vi_a/self.vi_b
        self.E_inv_alpha = self.vi_b/(self.vi_a - 1)
        self.E_log_alpha = -log_eps(self.vi_b) + self.digamma_vi_a

    def update_all_params(self):
        # including params which are const
        self.log_gamma_a0 = gammaln(self.a0)
        self.log_gamma_vi_a = gammaln(self.vi_a)
        self.digamma_vi_a = digamma(self.vi_a)

        self.update_params()

        self.kl_const = -self.K*(self.log_gamma_a0) + self.K*(self.a0*log_eps(self.b0))
        self.entropy_cons = np.sum(self.vi_a) + np.sum(self.log_gamma_vi_a) + np.sum((1 - self.vi_a)*self.digamma_vi_a)

    def ELBO(self):
        kl = self.kl_const + (self.a0 - 1)*np.sum(self.E_log_alpha) - self.b0*np.sum(self.E_alpha)
        entropy = self.entropy_cons - np.sum(log_eps(self.vi_b))
        self.elbo = kl + entropy


class nodeFA_theta_m(nodeFA_general):
    """
    Class to define theta node (k, for one m)
    """
    def __init__(self, a0, b0, vi_a, vi_b, m):
        self.m = m

        self.a0 = a0
        self.b0 = b0

        self.vi_a = vi_a
        self.vi_b = vi_b      

        self.update_params()

        self.elbo = 0
    
    def MB(self, s_m_node):
        self.s_m_node = s_m_node

    def update_k(self, k):
        sum_sdk = np.sum(self.s_m_node.vi_gamma[:,k])
        self.vi_a[k] = self.a0 + sum_sdk
        self.vi_b[k] = self.b0 - sum_sdk + self.D[self.m]

        self.update_params()

    def update_params(self):
        self.E_log_theta = digamma(self.vi_a) - digamma(self.vi_a + self.vi_b)
        self.E_log_1minustheta = digamma(self.vi_b) - digamma(self.vi_a + self.vi_b)
        self.E_log_LR = self.E_log_theta - self.E_log_1minustheta

    def ELBO(self):
        kl = np.sum((self.a0 - 1)*self.E_log_theta) + np.sum((self.b0 - 1)*self.E_log_1minustheta) - np.sum(betaln(self.a0, self.b0))
        entropy = - np.sum((self.vi_a - 1)*self.E_log_theta) - np.sum((self.vi_b - 1)*self.E_log_1minustheta) + np.sum(betaln(self.vi_a, self.vi_b))
        self.elbo = kl + entropy



class nodeFA_tau_m(nodeFA_general):
    """
    Class to define tau node (dim d, for one m)
    """
    def __init__(self, a0, b0, m):
        self.a0 = a0
        self.b0 = b0

        self.m = m

        # same as in alpha
        self.vi_a = a0 + self.N*np.ones(self.D[self.m])/2
        self.vi_b = self.vi_a + 0.0

        self.update_all_params()

        self.E_resid_squared_half = 0

        self.elbo = 0

    def MB(self, y_m_node, w_m_node, z_node):
        self.w_m_node = w_m_node
        self.y_m_node = y_m_node
        self.z_node = z_node

    def update(self):
        self.update_params_w_z()

        self.vi_b = self.b0 + self.E_resid_squared_half

        self.update_params()
    
    def update_all_params(self):
        self.log_gamma_a0 = gammaln(self.a0)
        self.log_gamma_vi_a = gammaln(self.vi_a)
        self.digamma_vi_a = digamma(self.vi_a)

        self.update_params()

        self.kl_const = -self.D[self.m]*(self.log_gamma_a0) + self.D[self.m]*(self.a0*log_eps(self.b0))
        self.entropy_cons = np.sum(self.vi_a) + np.sum(self.log_gamma_vi_a) + np.sum((1 - self.vi_a)*self.digamma_vi_a)
    
    def update_params(self):
        
        self.E_tau = self.vi_a/self.vi_b

        self.E_log_tau = -log_eps(self.vi_b) + self.digamma_vi_a

    def update_params_w_z(self):
        # d x n, sum over n
        third_term_of_tau = np.sum(self.w_m_node.E_w_z_squared, axis=0)/2
        first_term_of_tau = np.sum(self.y_m_node.data**2, axis=0)/2
        second_term_of_tau = - np.sum(self.y_m_node.data*self.w_m_node.E_w_z, axis=0)

        self.E_resid_squared_half = first_term_of_tau + second_term_of_tau + third_term_of_tau
    
    def ELBO(self):
        kl = self.kl_const + (self.a0 - 1)*np.sum(self.E_log_tau) - self.b0*np.sum(self.E_tau)
        entropy = self.entropy_cons - np.sum(log_eps(self.vi_b))
        self.elbo = kl + entropy


class nodeFA_y_m(nodeFA_general):
    def __init__(self, data_n, m):
        self.m = m

        self.data = data_n

        self.elbo = 0

    def MB(self, w_m_node, tau_m_node):
        self.w_m_node = w_m_node
        self.tau_m_node = tau_m_node

    def ELBO(self):
        elbo = -self.N*self.D[self.m]*log_eps(2*np.pi)/2 + self.N*np.sum(self.tau_m_node.E_log_tau)/2 \
             - np.sum(self.tau_m_node.E_tau*self.tau_m_node.E_resid_squared_half)
        self.elbo = elbo


def starting_params_z(starting_params, N, K):

    if 'z_mu' in starting_params.keys():
        z_mean = 1*starting_params['z_mu']
    else:
        z_mean = np.random.normal(size=(N, K))

    if 'z_var' in starting_params.keys():
        z_var = starting_params['z_var']
    else:
        z_var = np.ones((N, K))

    return z_mean, z_var


def starting_params_hat_w_m(starting_params, key_M, D, K):
    starting_params_m = starting_params[key_M]

    if 'w_mu' in starting_params_m.keys():
        w_mean = 1*starting_params_m['w_mu']
    else:
        w_mean = np.random.normal(size=(D, K))

    if 'w_var' in starting_params_m.keys():
        w_var = 1*starting_params_m['w_var']
    else:
        w_var = np.ones((D, K))

    return w_mean, w_var


def starting_params_s_m(starting_params, key_M, D, K):
    starting_params_m = starting_params[key_M]

    if 's_lambda' in starting_params_m.keys():
        s_lambda = 1.0*starting_params_m['s_lambda']
    else:
        # start with p(s=1) > 0.999
        s_lambda = 10.0*np.ones((D, K))

    return s_lambda


class FA():
    def __init__(self, data, N, M, K, D, O, S, center_data=None, starting_params=None, *args, **kwargs):
        self.N = N  # number of observations (samples)
        self.M = M  # number of views
        self.K = K  # number of hidden factors
        self.D = D  # a list containing number of features in each view
        self.O = O  # a list containing information if Y is observed (True) or if it needs CTM (False)
        self.S = S  # feature-wise sparsity per modality (spike and slab)

        # if modality is not observed turn off the spike and slab prior
        S = S and O

        # starting options
        if starting_params is None:
            starting_params = dict()
        for m in range(M):
            if not 'M'+str(m) in starting_params.keys():
                starting_params.update({'M'+str(m): dict()})
        if 'centering_data' in starting_params.keys():
            center_data = starting_params['centering_data']
        else:
            center_data = [True for m in range(M)]       

        # initialize the generel node
        nodeFA_general(N, K, D, M, O, self.S)

        # initialize all of the nodes
        z_mean, z_var = starting_params_z(starting_params, self.N, self.K)
        self.node_z = nodeFA_z(mu0 = 0, var0 = 1, vi_mu = z_mean, vi_var = z_var)
        self.nodelist_y = []
        self.nodelist_hat_w = []
        self.nodelist_s = []
        self.nodelist_w = []
        self.nodelist_alpha = []
        self.nodelist_theta = []
        self.nodelist_tau = []

        for m in range(self.M):
            key_tmp = 'M'+str(m)
            data_m = data[key_tmp]

            if self.O[m]:
                if center_data[m]:
                    feature_mean_m = np.mean(data_m, axis=0)
                    node_y_m = nodeFA_y_m(data_m - feature_mean_m, m)
                    node_y_m.data_mean = feature_mean_m
                else:
                    node_y_m = nodeFA_y_m(data_m + 0.0, m)
                    node_y_m.data_mean = None
            else:
                node_y_m = nodeFA_y_m(None, m)
            self.nodelist_y.append(node_y_m)

            w_mu, w_var = starting_params_hat_w_m(starting_params, key_tmp, D[m], K)
            node_hat_w_m = nodeFA_hat_w_m(w_mu, w_var) 
            self.nodelist_hat_w.append(node_hat_w_m)

            if self.S[m]:
                s_lambda = starting_params_s_m(starting_params, key_tmp, D[m], K)
                s = nodeFA_s_m(s_lambda)
            else:
                s = None
            self.nodelist_s.append(s)

            node_w_m = nodeFA_w_m(m)
            self.nodelist_w.append(node_w_m)

            node_alpha_m = nodeFA_alpha_m(1e-3, 1e-3, m)
            self.nodelist_alpha.append(node_alpha_m)

            if self.S[m]:
                node_theta_m = nodeFA_theta_m(1, 1, 99*np.ones(K), np.ones(K), m)
            else:
                node_theta_m = None
            self.nodelist_theta.append(node_theta_m)

            if self.O[m]:
                node_tau_m = nodeFA_tau_m(1e-3, 1e-3, m)
            else:
                node_tau_m = nodeFA_tau_m(1e-3, 1e-3, m)
            self.nodelist_tau.append(node_tau_m)

        self.elbo = 0

    def MB(self):
        self.node_z.MB(self.nodelist_y, self.nodelist_w, self.nodelist_tau)

        for m in range(self.M):
            self.nodelist_y[m].MB(self.nodelist_w[m], self.nodelist_tau[m])

            self.nodelist_hat_w[m].MB()
            if self.S[m]:
                self.nodelist_s[m].MB()

            self.nodelist_w[m].MB(self.nodelist_hat_w[m], self.nodelist_s[m], self.nodelist_alpha[m], self.nodelist_theta[m], 
                                  self.node_z, self.nodelist_y[m], self.nodelist_tau[m])

            self.nodelist_alpha[m].MB(self.nodelist_hat_w[m], self.nodelist_s[m], self.nodelist_w[m])
            if self.S[m]:
                self.nodelist_theta[m].MB(self.nodelist_s[m])

            self.nodelist_tau[m].MB(self.nodelist_y[m], self.nodelist_w[m], self.node_z)

       
    def update(self, update_tau=True, update_alpha=True):

        # update all Z by k
        for k in range(self.K):
            self.node_z.update_k(k)
        
        # update parameters in W that depend on Z
        for m in range(self.M):
            self.nodelist_w[m].update_params_z()

        # update W_hat, S (W = W_hat * S) by k and m
        #  - it depends on tau params, but not tau_w_z
        for k in range(self.K):
            for m in range(self.M):
                self.nodelist_w[m].update_k(k)

        # update params of tau which depend on W and Z  by k and m
        for m in range(self.M):
            self.nodelist_tau[m].update_params_w_z()

        # update alpha by k and m
        for k in range(self.K):
            for m in range(self.M):
                if update_alpha:
                    self.nodelist_alpha[m].update_k(k)

        # update theta by k and m
        for k in range(self.K):
            for m in range(self.M):
                if self.O[m] and self.S[m]:
                    self.nodelist_theta[m].update_k(k)

        # update tau by m
        for m in range(self.M):
            if self.O[m]:
                if update_tau:
                    self.nodelist_tau[m].update()
    
                    
    def delete_inactive(self, tres = 0.001):
        are_deleted = False
        var_exp = self.variance_explained_per_factor()

        set_of_active = var_exp >= tres
        number_of_inactive = self.K - np.sum(set_of_active).astype('int')        

        if number_of_inactive > 0:

            # to delete just 1 per iteration:
            # set_of_active = np.ones(self.K).astype('bool')
            # set_of_active[np.where(var_exp == np.min(var_exp))[0][0]] = False
            # # set_of_active = var_exp > np.min(var_exp)
            # number_of_inactive = self.K - np.sum(set_of_active).astype('int')

            self.K = np.sum(set_of_active).astype('int')
            nodeFA_general.K = np.sum(set_of_active).astype('int')

            self.node_z.vi_mu = self.node_z.vi_mu[:,set_of_active]
            self.node_z.vi_var = self.node_z.vi_var[:,set_of_active]
            self.node_z.update_params()

            for m in range(self.M):
                self.nodelist_hat_w[m].vi_mu = self.nodelist_hat_w[m].vi_mu[:,set_of_active]
                self.nodelist_hat_w[m].vi_var = self.nodelist_hat_w[m].vi_var[:,set_of_active]

                if self.O[m] and self.S[m]:

                    self.nodelist_s[m].vi_gamma = self.nodelist_s[m].vi_gamma[:,set_of_active]
                    self.nodelist_s[m].vi_lambda = self.nodelist_s[m].vi_lambda[:,set_of_active]

                self.nodelist_alpha[m].vi_a = self.nodelist_alpha[m].vi_a[set_of_active]
                self.nodelist_alpha[m].vi_b = self.nodelist_alpha[m].vi_b[set_of_active]
                self.nodelist_alpha[m].update_all_params()

                if self.O[m] and self.S[m]:
                    self.nodelist_theta[m].vi_a = self.nodelist_theta[m].vi_a[set_of_active]
                    self.nodelist_theta[m].vi_b = self.nodelist_theta[m].vi_b[set_of_active]
                    self.nodelist_theta[m].update_params()


                self.nodelist_w[m].update_params()
                self.nodelist_w[m].update_params_z()
            
            print("Deleted " + str(number_of_inactive)+" inactive factors")
            if number_of_inactive > 0:
                are_deleted = True
        
        return are_deleted
    
    def ELBO(self):
        # compute elbo
        self.node_z.ELBO()
        for m in range(self.M):
            self.nodelist_w[m].ELBO()
            self.nodelist_alpha[m].ELBO()
            self.nodelist_y[m].ELBO()
            if self.O[m] and self.S[m]:
                self.nodelist_theta[m].ELBO()
            if self.O[m]:
                self.nodelist_tau[m].ELBO()

        # update self.elbo
        elbo = 0
        elbo += self.node_z.elbo
        for m in range(self.M):
            elbo += self.nodelist_w[m].elbo
            elbo += self.nodelist_alpha[m].elbo
            elbo += self.nodelist_y[m].elbo
            if self.S[m]:
                elbo += self.nodelist_theta[m].elbo
            if self.O[m]:
                elbo += self.nodelist_tau[m].elbo
        self.elbo = elbo

    def get_elbo(self):
        return self.elbo
    
    def get_elbo_per_node(self):

        return (self.node_z.elbo, 
                [self.nodelist_w[m].elbo for m in range(self.M)],
                [self.nodelist_alpha[m].elbo for m in range(self.M)],
                [self.nodelist_tau[m].elbo for m in range(self.M) if self.O[m]],
                [self.nodelist_theta[m].elbo for m in range(self.M) if self.O[m] and self.S[m]],
                [self.nodelist_y[m].elbo for m in range(self.M)])
    
    def variance_explained_per_factor(self):

        var_exp_nominator = np.zeros(self.K)
        var_exp_denominator = np.zeros(self.K)

        for k in range(self.K):
            for m in range(self.M):
                var_exp_nominator[k] += np.sum((self.nodelist_y[m].data - np.outer(self.node_z.E_z[:,k], self.nodelist_w[m].E_w[:,k]))**2)
                var_exp_denominator[k] += np.sum(self.nodelist_y[m].data**2)
        
        return 1 - var_exp_nominator/var_exp_denominator

    def variance_explained_per_view(self):

        var_exp_nominator = np.zeros(self.M)
        var_exp_denominator = np.zeros(self.M)

        for m in range(self.M):
            var_exp_nominator[m] += np.sum((self.nodelist_y[m].data - np.dot(self.node_z.E_z, self.nodelist_w[m].E_w.T))**2)
            var_exp_denominator[m] += np.sum(self.nodelist_y[m].data**2)
        
        return 1 - var_exp_nominator/var_exp_denominator
    
    def variance_explained_per_factor_view(self):

        var_exp_nominator = np.zeros((self.K, self.M))
        var_exp_denominator = np.zeros((self.K, self.M))

        for k in range(self.K):
            for m in range(self.M):
                var_exp_nominator[k, m] = np.sum((self.nodelist_y[m].data - np.outer(self.node_z.E_z[:,k], self.nodelist_w[m].E_w[:,k]))**2)
                var_exp_denominator[k, m] = np.sum(self.nodelist_y[m].data**2)
        
        return 1 - var_exp_nominator/var_exp_denominator
    
    def variance_explained_per_factor_feature(self):

        explained_variance_list = []

        for m in range(self.M):

            D_m = self.nodelist_y[m].data.shape[1]

            var_exp_nominator = np.zeros((self.K, D_m))
            var_exp_denominator = np.zeros((self.K, D_m))

            for k in range(self.K):
                for d_m in range(D_m):
                    var_exp_nominator[k, d_m] = np.sum((self.nodelist_y[m].data[:,d_m] - self.node_z.E_z[:,k]*self.nodelist_w[m].E_w[d_m,k])**2)
                    var_exp_denominator[k, d_m] = np.sum(self.nodelist_y[m].data[:,d_m]**2)
            explained_variance_list.append(1 - var_exp_nominator/var_exp_denominator)
        
        return explained_variance_list