###############################
# KernelSHAP4K For Regression #
###############################

import torch
from scipy.special import binom
import numpy as np
from gpytorch.kernels import RBFKernel
from gpytorch.lazy import lazify
from sklearn.linear_model import Ridge
from numpy import sum
from tqdm import tqdm
from src.sampling import large_scale_sample_alternative, generate_full_Z, subsetting_full_Z



class RKHSSHAP(object):
    """[summary]
    Implement the exact RKHS SHAP algorithm with no kernel approximation
    """

    def __init__(self, X, y, lambda_krr, lambda_cme, lengthscale):
        """[summary]
        
        Args:
            X ([torch tensor]): [data]
            y ([torch tensor]): [label]
            lambda_krr ([torch float]): [RKHS norm reg]
            lambda_cme ([type]): [Lambda CME reg]
            lengthscale ([type]): [lengthscale for kernel]
        """

        # Store data
        self.n, self.m = X.shape
        self.X, self.y = X, y
        self.X_scaled = (X/lengthscale).float()
        self.lengthscale = lengthscale

        self.lambda_cme, self.lambda_krr = lambda_cme, lambda_krr

        # Set up kernel
        rbf = RBFKernel()
        rbf.raw_lengthscale.requires_grad = False
        self.k = rbf

        # Run Kernel Ridge Regression (need alphas!)
        Kxx = rbf(self.X_scaled)
        alphas = Kxx.add_diag(lambda_krr).inv_matmul(self.y)
        self.alphas = alphas.float().reshape(-1, 1)

        self.ypred = Kxx@alphas
        self.rmse = torch.sqrt(torch.mean(self.ypred - self.y)**2)

    def _value_intervention(self, z, X_new):

        X_new_scaled = X_new/self.lengthscale

        n_ = X_new.shape[0]
        zc = (z == False)

        reference = (self.ypred.mean() * torch.ones((1, n_))).float()
        self.reference = reference

        if z.sum() == self.m:
            # features all active
            new_ypred = self.alphas.T @ self.k(self.X_scaled, X_new_scaled).evaluate()
            return new_ypred - reference
        
        elif z.sum() == 0:
            return 0
        
        else:
            X_S, X_Sc = self.X_scaled[:, z], self.X_scaled[:, zc]
            Xp_S = X_new_scaled[:, z]
            K_SSp = self.k(X_S, Xp_S).evaluate().float()
            K_Sc = self.k(X_Sc, X_Sc)

            KME_mat = K_Sc.evaluate().mean(axis=1)[:, np.newaxis] * torch.ones((self.n, n_))

            return self.alphas.T @ (K_SSp * KME_mat) - reference

    def _value_observation(self, z, X_new):

        X_new_scaled = X_new/self.lengthscale

        n_ = X_new.shape[0]
        zc = (z == False)

        reference = (self.ypred.mean() * torch.ones((1, n_))).float()
        self.reference = reference

        if z.sum() == self.m:
            new_ypred = self.alphas.T @ self.k(self.X_scaled, X_new_scaled).evaluate()

            return new_ypred - reference
        
        elif z.sum() == 0:
            return 0

        else:
            X_S, X_Sc = self.X_scaled[:, z], self.X_scaled[:, zc]
            Xp_S = X_new_scaled[:, z]
            K_SSp = self.k(X_S, Xp_S).evaluate().float()
            K_Sc = self.k(X_Sc, X_Sc)
            K_SS = self.k(X_S, X_S)

            Xi_S = (K_SS.add_diag(self.n * self.lambda_cme).inv_matmul(K_Sc.evaluate())).T

            return self.alphas.T @ (K_SSp * (Xi_S @ K_SSp)) - reference


    def fit(self, X_new, method, sample_method, num_samples=100, wls_reg=1e-10):

        n_ = X_new.shape[0]

        if sample_method=="MC":
            Z = large_scale_sample_alternative(self.m, num_samples)
        elif sample_method=="MC2":
            Z = generate_full_Z(self.m)
            Z = subsetting_full_Z(Z, samples=num_samples)
        else:
            Z = generate_full_Z(self.m)


        # Set up containers
        epoch = Z.shape[0]
        Y_target = np.zeros((epoch, n_))

        count = 0
        weights = []

        for row in tqdm(Z):
            if np.sum(row) == 0 or np.sum(row) == self.m:
                weights.append(1e+5)
                
            else:
                z = row
                weights.append((self.m - 1) / (binom(self.m, sum(z)) * sum(z) * (self.m - sum(z))))
            
            if method == "O":
                Y_target[count, :] = self._value_observation(row, X_new)
            elif method == "I":
                Y_target[count, :] = self._value_intervention(row, X_new)
            else:
                raise ValueError("Must be either interventional or observational")

            count += 1        

        clf = Ridge(wls_reg)
        clf.fit(Z, Y_target, sample_weight=weights)

        self.full_shapley_values_ = np.concatenate([clf.intercept_.reshape(-1, 1), clf.coef_], axis=1)
        self.SHAP_LM = clf

        return clf.coef_




