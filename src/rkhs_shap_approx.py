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
from src.kernel_approx import Nystroem_gpytorch


class RKHSSHAP_Approx(object):
    """
    Instantiate this class to run the RKHS-SHAP algorithm. Nystroem approximation is used by default.
    """

    def __init__(self,
                 X: np.array,
                 y: np.array,
                 lambda_krr: np.float = 1e-2,
                 lambda_cme: np.float = 1e-3,
                 lengthscale: np.array = None,
                 n_components: np.int = 100
                 ):

        """[summary]

        Args:
            X (np.array): [training instances]
            y (np.array): [training label]
            lambda_krr (np.float, optional): [regularisation for kernel ridge regression]. Defaults to 1e-2.
            lambda_cme (np.float, optional): [regularisation for conditional mean embedding]. Defaults to 1e-3.
            lengthscale (np.array, optional): [lengthscale for kernel]. Defaults to None.
            n_components (np.int, optional): [number of landmark points for kernel approximation]. Defaults to 100.

        """

        # Storing init data
        self.n, self.m = X.shape
        self.X = X
        self.X_scaled = X / lengthscale
        self.ls = lengthscale
        self.y = y
 
        self.lambda_cme = lambda_cme
        self.lambda_krr = lambda_krr
        self.num_components = n_components

        # Set up kernel
        rbf = RBFKernel()
        rbf.raw_lengthscale.requires_grad = False

        # Run Kernel Ridge Regression
        if n_components is None:
            raise ValueError("Input number of landmark points")

        ny = Nystroem_gpytorch(kernel=rbf,
                                lengthscale=self.ls,
                                n_components=n_components
                                )
        ny.fit(self.X)
        Z = ny.transform(self.X)
        Kx = Z @ Z.T

        alphas = lazify(Kx).add_diag(torch.tensor(self.lambda_krr)).inv_matmul(torch.tensor(y.reshape(-1, 1)).float())

        self.ypred = Kx @ alphas
        self.y_ten = torch.tensor(y).reshape(-1, 1).float()
        self.rmse = torch.sqrt(torch.mean((self.ypred - self.y_ten)**2))

        # Store alphas and nystroem object
        self.Z = Z
        self.alphas = alphas
        self.nystroem = ny
        self.Kx = Kx

    def _value_intervention(self, z, X_new: np.array, substract_ref=True):
        """
        Compute the marginal expectation of E_zc[f(X_new)]

        :param z: giving you the index for subsetting
        :param X_new: evaluating at these new points (could be old points)

        :return: marginal expectation of E_zc[f(X_new)]
        """

        n_ = X_new.shape[0]
        zc = (z == False)

        # compute the reference value - using previously trained data
        reference = (self.ypred.mean() * torch.ones((1, n_))).float()
        self.reference = reference

        if z.sum() == self.m:
            new_ypred = self.alphas.T @ self.Z @ self.nystroem.transform(X_new).T
            if substract_ref:
                return new_ypred - reference
            else:
                return new_ypred
        elif z.sum() == 0:
            if substract_ref:
                return 0
            else:
                return reference
        else:
            Z_S = self.nystroem.transform(self.X, active_dims=z)
            Z_S_new = self.nystroem.transform(X_new, active_dims=z)
            K_SSn = Z_S @ Z_S_new.T

            # Only using marginal measure from data, not new points
            Z_Sc = self.nystroem.transform(self.X, active_dims=zc)
            K_Sc = Z_Sc @ Z_Sc.T

            KME_mat = K_Sc.mean(axis=1)[:, np.newaxis] * torch.ones((self.n, n_))

            if substract_ref:
                return self.alphas.T @ (K_SSn * KME_mat) - reference
            else:
                return self.alphas.T @ (K_SSn * KME_mat)

    def _value_observation(self, z, X_new, substract_ref=True):
        """
        Compute the conditional expectation E_{Sc|S=s}[f(X_new)|S=s]

        :param z:
        :param X_new:
        :return:
        """
        zc = z == False
        n_ = X_new.shape[0]

        # compute the reference value - using previously trained data
        reference = self.ypred.mean() * torch.ones((1, n_))

        # Compute Reference

        if z.sum() == self.m:
            new_ypred = self.alphas.T @ self.Z @ self.nystroem.transform(X_new).T
            if substract_ref:
                return new_ypred - reference
            else:
                return new_ypred
        elif z.sum() == 0:
            if substract_ref:
                return 0
            else:
                return reference
        else:
            Z_S = self.nystroem.transform(self.X, active_dims=z)
            Z_S_new = self.nystroem.transform(X_new, active_dims=z)
            K_SSn = Z_S @ Z_S_new.T

            # Only using marginal measure from data, not new points
            Z_Sc = self.nystroem.transform(self.X, active_dims=zc)
            cme_latter_part = lazify(Z_S.T @ Z_S).add_diag(torch.tensor(self.lambda_cme).float()).inv_matmul(Z_S_new.T)
            holder = self.alphas.T @ (K_SSn * (Z_Sc @ Z_Sc.T @ Z_S @ cme_latter_part))

            if substract_ref:
                return holder - reference
            else:
                return holder

    def fit(self, X_new: np.array, method: str="O", sample_method: str="MC", num_samples: int=1000, substract_ref: bool=True, verbose: str=False):
        """[Running the RKHS SHAP Algorithm to explain kernel ridge regression]

        Args:
            X_new (np.array): [New X data]
            method (str, optional): [Interventional Shapley values (I) or Observational Shapley Values (O)]. Defaults to "O".
            sample_method (str, optional): [What sampling methods to use for the permutations
                                                if "MC" then you do sampling.
                                                if None, you look at all potential 2**M permutation ]. Defaults to "MC".
            num_samples (int, optional): [number of samples to use]. Defaults to None.
            verbose (str, optional): [description]. Defaults to False.
        """

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

        if verbose:
            for row in tqdm(Z):
                if np.sum(row) == 0 or np.sum(row) == self.m:
                    weights.append(1e+5)
                else:
                    z = row
                    weights.append((self.m - 1) / (binom(self.m, sum(z)) * sum(z) * (self.m - sum(z))))
                
                if method == "O":
                    Y_target[count, :] = self._value_intervention(row, X_new, substract_ref)
                elif method == "I":
                    Y_target[count, :] = self._value_observation(row, X_new, substract_ref)
                else:
                    raise ValueError("Must be either interventional or observational")

                count += 1
        
        else:
            for row in Z:
                if np.sum(row) == 0 or np.sum(row) == self.m:
                    weights.append(1e+5)
                else:
                    z = row
                    weights.append(((self.m - 1) / (binom(self.m, sum(z)) * sum(z) * (self.m - sum(z)))))
                
                if method == "O":
                    Y_target[count, :] = self._value_intervention(row, X_new, substract_ref)
                elif method == "I":
                    Y_target[count, :] = self._value_observation(row, X_new, substract_ref)
                else:
                    raise ValueError("Must be either interventional or observational")

                count += 1

        clf = Ridge(1e-5)
        clf.fit(Z, Y_target, sample_weight=weights)

        self.full_shapley_values_ = np.concatenate([clf.intercept_.reshape(-1, 1), clf.coef_], axis=1)
        self.SHAP_LM = clf

        return clf.coef_
