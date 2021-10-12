#################################################
# KernelSHAP4K For Fair Learning Regularisation #
#################################################

import copy
import torch
from scipy.special import binom
import numpy as np
from gpytorch.kernels import RBFKernel
from gpytorch.lazy import lazify
from tqdm import tqdm
from src.sampling import large_scale_sample_alternative, generate_full_Z, subsetting_full_Z
from src.kernel_approx import Nystroem_gpytorch


def insert_i(ls, feature_to_exclude):
    """
    A helping function for computing Shapley functional in ShapleyRegulariser
    """
    subset_vec = np.array([True for i in range(len(ls) + 1)])
    subset_vec[feature_to_exclude] = False

    holder = np.zeros(len(ls) + 1)
    holder[subset_vec] = ls

    Sui = copy.deepcopy(holder)
    Sui[feature_to_exclude] = True

    S = copy.deepcopy(holder)
    S[feature_to_exclude] = False

    return [Sui == 1, S == 1]


class ShapleyRegulariser(object):

    def __init__(self, lambda_sv: float, lambda_krr: float, lambda_cme: float, n_components: int):
        """[Initialisation]

        Args:
            lambda_sv (float): [regularisation parameter for Shapley Regulariser]
            lambda_krr (float): [regularisation parameter for kernel ridge regression]
            lambda_cme (float): [regularisation parameter for conditional mean embedding]
            n_components (int): [number of landmark points for kernel approximation]
        """

        self.lambda_cme = lambda_cme
        self.lambda_krr = lambda_krr
        self.lambda_sv = lambda_sv
        self.n_components = n_components
        
        self.Z = None
        self.nystroem = None
    
    def _get_int_embedding(self, z, X):
        """
        Compute the interventional embedding
        """
        
        zc = z == False
        n, m = X.shape
        
        if z.sum() == m:
            return self.Z @ self.Z.T
        elif z.sum() == 0:
            return (self.Kx.mean(axis=1) * torch.ones((n, n))).T
        else:
            Z_S = self.nystroem.transform(X, active_dims=z)
            K_SS = Z_S @ Z_S.T 

            Z_Sc = self.nystroem.transform(X, active_dims=zc)
            K_Sc = Z_Sc @ Z_Sc.T

            KME_mat = K_Sc.mean(axis=1)[:, np.newaxis] * torch.ones((n, n))

            return K_SS * KME_mat
    
    def _get_obsv_embedding(self, z, X):
        """
        Compute the observational embedding
        """

        zc = z == False
        n, m = X.shape

        if z.sum() == 0:
            return self.Kx
        elif z.sum() == 0:
            return self.Kx.mean(axis=1) * torch.ones((n, n)).T
        else:
            Z_S = self.nystroem.transform(X, active_dims=z)
            K_SS = Z_S @ Z_S.T

            Z_Sc = self.nystroem.transform(X, active_dims=zc)
            cme_latter_part = lazify(Z_S.T @ Z_S).add_diag(torch.tensor(self.lambda_cme).float()).inv_matmul(Z_S.T)
            holder = (K_SS * (Z_Sc @ Z_Sc.T @ Z_S @ cme_latter_part))

            return holder
        
    def fit(self, X: float, y: float, ls: float, features_index: list, method: str="O", num_samples: int=300, sample_method: str="MC"):
        """[summary]

        Args:
            X (float): [training data]
            y (float): [training labels]
            ls (float): [lengthscales for the kernel]
            features_index (list): [list containing which feature to minimise]
            method (str, optional): ["O" stands for OSV-Reg and "I" stands for ISV-Reg]. Defaults to "O".
            num_samples (int, optional): [number of samples to estimate the shapley functionals]. Defaults to 300.
            sample_method (str, optional): [sampling method to compute shapley functionals]. Defaults to "MC".
        """


        self.X = X
        self.n, self.m = X.shape
        self.ls = ls

        rbf = RBFKernel()
        rbf.raw_lengthscale.requires_grad = False

        ny = Nystroem_gpytorch(kernel=rbf,
                                lengthscale=self.ls,
                                n_components=self.n_components
                                )
        ny.fit(self.X)
        Phi = ny.transform(self.X)
        self.Z = Phi
        Kx = Phi@Phi.T

        self.nystroem = ny
        self.Kx = Kx

        m_exclude_i = self.m - len(features_index)
        if sample_method == "MC":
            Z_exclude_i = large_scale_sample_alternative(m_exclude_i, num_samples)
        else:
            Z_exclude_i = generate_full_Z(m_exclude_i)
        A = np.zeros((self.n, self.n))

        for row in tqdm(Z_exclude_i):
            Sui, S = insert_i(row, feature_to_exclude=features_index)
            
            if method == "O":
                sui = self._get_obsv_embedding(Sui, X)
                s = self._get_obsv_embedding(S, X)

                A += (sui - s).numpy()

            elif method == "I":
                sui = self._get_int_embedding(Sui, X)
                s = self._get_int_embedding(S, X)

                A += (sui - s).numpy()
            
            else:
                raise ValueError("Method must either be Observational or Interventional")
            
        A = A / Z_exclude_i.shape[0]

        self.A = A

        # Formulate the regression
        y_ten = torch.tensor(y).reshape(-1, 1).float()
        A = lazify(torch.tensor(A).float())
        self.AAt = A@A.t()
        K = rbf(torch.tensor(X/ls)).float()
        alphas = (K@K + self.lambda_krr * K + self.lambda_sv*self.AAt).add_jitter().inv_matmul(K.matmul(y_ten))

        self.alphas = alphas
        ypred = K.matmul(alphas)

        print("RMSE: %.2f" %torch.sqrt(torch.mean((y_ten - ypred)**2)))

        self.RMSE = torch.sqrt(torch.mean((y_ten - ypred)**2)).numpy()
        self.ypred = ypred

    def predict(self, X_test):

        # obtain kernel K_{test, train} first
        Z_test = self.nystroem.transform(X_test)
        K_xpx = Z_test @ self.Z.T

        return K_xpx@self.alphas
