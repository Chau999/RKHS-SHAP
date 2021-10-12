import copy
import numpy as np

class Banana2d(object):

    def __init__(self, n: int, b: float, v: float, noise=0.1, outlier_quantile=3):

        self.n = n
        self.b = b
        self.v = v

        # Simulate data
        cov_matrix = np.eye(2)
        cov_matrix[0, 0] = v
        mean_vec = np.zeros(2)

        Z = np.random.multivariate_normal(mean=mean_vec, cov=cov_matrix, size=n)
        X = copy.deepcopy(Z)
        X[:, 1] += b * (X[:, 0]**2 - v)

        # f(x) or ys
        y = X[:, 1] + b * (X[:, 0]**2 - v) + noise*np.random.normal(size=n)

        # Store result
        
        # Remove extreme outlier
        keep = X[:, 1] < outlier_quantile*X[:, 1].std()
        X = X[keep, :]
        y = y[keep]
        
        self.X, self.y = X, y        

        # Compute Observational Shapley
        self.phi_1 = 0.5 * (3*b*(X[:, 0]**2 - v) - X[:, 1])
        self.phi_2 = 0.5 * (3*X[:, 1] - b * (X[:, 0]**2 - v))

        # Compute Interventional Shapley
        self.phi_1_I = b*(X[:, 0]**2 - v)
        self.phi_2_I = X[:, 1]
        
