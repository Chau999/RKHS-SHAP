import torch
import numpy as np
from sklearn.cluster import KMeans


class Nystroem_gpytorch(object):
    """[summary]

    Fit and run the nystroem approximation for large scale kernel operation

    """

    def __init__(self, kernel, lengthscale, n_components):
        """

        :param kernel: gpytorch basekernel
        :param lengthscale: lengthscale of the kernel
        :param n_componenets:
        """

        self.ls = lengthscale
        self.n_components = n_components
        self.fitted = None
        self.kernel = kernel

    def fit(self, X):
        X = X/self.ls

        km = KMeans(n_clusters=self.n_components)
        km.fit(X)

        self.landmarks = km.cluster_centers_

    def transform(self, X, active_dims=None):
        if active_dims is None:
            active_dims = np.array([True for i in range(X.shape[1])])

        X = X/self.ls

        sub_X = torch.tensor(X[:, active_dims]).float()
        
        sub_landmarks = torch.tensor(self.landmarks[:, active_dims]).float()

        ZT = self.kernel(sub_landmarks).add_jitter().cholesky().inv_matmul(self.kernel(sub_landmarks, sub_X).evaluate())

        #return (ZT.T).detach().numpy()
        return ZT.T

    def compute_kernel(self, X, active_dims=None):

        Z = self.transform(X=X, active_dims=active_dims)
        return Z @ Z.T


