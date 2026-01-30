import math

import gin
import numpy as np


def dB_to_lin(x):
    return 10**(x/10)


@gin.configurable
class LorenzSSM(object):
    """ 
    This class defines the state space model for the Lorenz-63 attractor (LorenzSSM) for `alpha=0` or 
    for the Chen attractor (ChenSSM) using `alpha=1`. The simulation in discrete-time was done using a Taylor series 
    approximation of the matrix exponential (ref. to equation in the manuscript). 

    The function described in `A_fn` is in accordance with the generalized Lorenz form introduced in
    Čelikovský, Sergej, and Guanrong Chen. "On the generalized Lorenz canonical form." Chaos, Solitons & Fractals 26.5 (2005): 1271-1276.
    """
    def __init__(self, n_states=3, n_obs=3, J=5, delta=0.02, delta_d=0.02, alpha=0.0, decimate=False, mu_e=None, mu_w=None, H=None, use_Taylor=True, nonlinear_H=None) -> None:
        
        self.n_states = n_states
        self.J = J
        self.delta = delta
        self.alpha = alpha # alpha = 0 -- Lorenz attractor, alpha = 1 -- Chen attractor
        self.delta_d = delta_d
        self.n_obs = n_obs
        self.decimate = decimate
        self.mu_e = mu_e if mu_e is not None else np.zeros((n_states,))
        self.H = nonlinear_H if nonlinear_H is not None else lambda x: x
        self.mu_w = mu_w if mu_w is not None else np.zeros((n_obs,))
        self.use_Taylor = use_Taylor
    
    def A_fn(self, z): 
        return np.array([
        [-(10 + 25*self.alpha), (10 + 25*self.alpha), 0],
        [(28 -  35*self.alpha), (29*self.alpha - 1), -z],
        [0, z, -(8.0 + self.alpha)/3]
    ])
    
    #def A_fn(self, z):
    #    return np.array([
    #                [-10, 10, 0],
    #                [28, -1, -z],
    #                [0, z, -8.0/3]
    #            ])
    
    def h_fn(self, x):
        """ 
        Linear measurement setup y = x + w
        """
        return self.H(x)
            
    def f_linearize(self, x):

        self.F = np.eye(self.n_states)
        for j in range(1, self.J+1):
            #self.F += np.linalg.matrix_power(self.A_fn(x)*self.delta, j) / np.math.factorial(j)
            self.F += np.linalg.matrix_power(self.A_fn(x[0])*self.delta, j) / math.factorial(j)

        return self.F @ x
    
    def setStateCov(self, sigma_e2=0.1):
        self.Ce = sigma_e2 * np.eye(self.n_states)

    def setMeasurementCov(self, sigma_w2=1.0):
        self.Cw = sigma_w2 * np.eye(self.n_obs)
    
    @property
    def observation_cov(self) -> np.ndarray:
        return self.Cw

    @property
    def state_cov(self) -> np.ndarray:
        return self.Ce

    def generate_state_sequence(self, T, sigma_e2_dB):

        self.sigma_e2 = dB_to_lin(sigma_e2_dB)
        self.setStateCov(sigma_e2=self.sigma_e2)
        K = int(self.delta // self.delta_d)
        x_lorenz = np.zeros((T * K, self.n_states))
        e_k_arr = np.random.multivariate_normal(self.mu_e, self.Ce, size=(K * T,))

        for t in range(0, K * T - 1):
            x_lorenz[t+1] = self.f_linearize(x_lorenz[t]) + e_k_arr[t]
        
        if self.decimate == True:
            K = int(self.delta // self.delta_d)
            x_lorenz_d = x_lorenz[::K, :]
        else:        
            x_lorenz_d = np.copy(x_lorenz)

        return x_lorenz_d
    
    def generate_measurement_sequence(self, x_lorenz, T, smnr_dB=10.0):
        
        #signal_p = ((self.h_fn(x_lorenz) - np.zeros_like(x_lorenz))**2).mean()
        signal_p = np.var(self.h_fn(x_lorenz))
        self.sigma_w2 = signal_p / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        w_k_arr = np.random.multivariate_normal(self.mu_w, self.Cw, size=(T,))
        y_lorenz = np.zeros((T, self.n_obs))
        
        #print("smnr: {}, signal power: {}, sigma_w: {}".format(smnr_dB, signal_p, self.sigma_w2))
        
        #print(self.H.shape, x_lorenz.shape, y_lorenz.shape)
        for t in range(0,T):
            y_lorenz[t] = self.h_fn(x_lorenz[t]) + w_k_arr[t]
        
        #if self.decimate == True:
        #    K = self.delta_d // self.delta
        #    y_lorenz_d = y_lorenz[0:T:K,:] 
        #else:
        y_lorenz_d = np.copy(y_lorenz)

        return y_lorenz_d
    
    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB, num=None):
        del num  # Unused as a new sequence is generated each time.

        x_lorenz = self.generate_state_sequence(T=T, sigma_e2_dB=sigma_e2_dB)
        y_lorenz = self.generate_measurement_sequence(x_lorenz=x_lorenz, T=T, smnr_dB=smnr_dB)

        return x_lorenz, y_lorenz


def main():
    lorenz_model = LorenzSSM()
    x_lorenz, y_lorenz = lorenz_model.generate_single_sequence(T=500, sigma_e2_dB=-10, smnr_dB=10)
    print(y_lorenz.shape)
    print(x_lorenz.shape)
    assert x_lorenz.shape == y_lorenz.shape


if __name__ == '__main__':
    main()