"""[Fitting a Kernel Ridge Regressor]
"""
import torch, gpytorch
import numpy as np
from sklearn.metrics import pairwise_distances


def compute_median_heuristic(X):
    median_heuristic = [np.median(pairwise_distances(X[:, [i]].reshape(-1,1))) for i in range(X.shape[1])]
    return torch.tensor(median_heuristic)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        self.covar_module.lengthscale = lengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class KernelRidgeRegressor(object):
    
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.fit_switch = 0
        #self.lengthscale = compute_median_heuristic(X)

        
    def fit(self, epoch=500, lr=1e-1, verbose=True, lengthscale=None):

        # define a GP model for optimisation
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(self.X, self.y, likelihood, lengthscale)

        model.train()
        likelihood.train()

        # optimiser and loss
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(epoch):
            optim.zero_grad()
            output = model(self.X)
            loss = -mll(output, self.y)
            loss.backward()

            if verbose:
                if i % 200 == 0:
                    print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                        i + 1, epoch, loss.item(),
                        model.likelihood.noise.item()
                    ))
            
            optim.step()
        
        # Store optimal parameters
        self.k = model.covar_module
        self.k.raw_lengthscale.require_grad = False
        self.lmda = torch.tensor(model.likelihood.noise.item())
        

        # Create alphas
        Kxx = self.k(self.X)
        self.alpha = Kxx.add_diag(self.lmda).inv_matmul(self.y)

        self.flip_switch = 1
    
    def predict(self, X_test):

        X_test = torch.tensor(X_test).float()
        Kxnx = self.k(X_test, self.X)

        return (Kxnx.evaluate() @ self.alpha).detach().numpy()









