"""
Use a similar logic to SHAP, fit a multivariate Gaussian and do data imputation
"""
import numpy as np
from tqdm import tqdm

class Observation2dBanana(object):

    def __init__(self, model, data):

        if data.shape[1] > 2:
            raise ValueError

        # Fit a multivariate Gaussian
        mean = np.mean(data, axis=1)
        cov = np.cov(data.T)

        self.mean, self.cov = mean, cov
        self.f = model

    def _value_function(self, z, x, sample=100):

        if z == 0:
            zc = 1
            new_mean = self.mean[1] + self.cov[1, 0]*(x[0] - self.mean[0])/self.cov[0,0]
            new_cov = self.cov[1,1] - self.cov[0,1] * self.cov[1,0]/self.cov[0,0]

            new_sample = np.random.normal(loc=new_mean, scale=new_cov, size=sample)

            hold = np.zeros((sample, 2))
            hold[:, z] = x[z]
            hold[:, zc] = new_sample

            samples_of_f = self.f(hold)

            return samples_of_f.mean()

        if z == 1:
            zc = 0
            new_mean = self.mean[1] + self.cov[1, 0]*(x[1] - self.mean[1])/self.cov[1,1]
            new_cov = self.cov[0, 0] - self.cov[0,1] * self.cov[1,0]/self.cov[1,1]

            new_sample = np.random.normal(loc=new_mean, scale=new_cov, size=sample)

            hold = np.zeros((sample, 2))
            hold[:, z] = x[z]
            hold[:, zc] = new_sample

            samples_of_f = self.f(hold)

            return samples_of_f.mean()

    def fit(self, data, num_samples):

        E_f_X = self.f(data).mean() * np.ones((data.shape[0]))
        E_f_X_conditioned_both = self.f(data)

        EfX_condition_1 = []
        EfX_condition_0 = []

        for i in tqdm(range(data.shape[0])):
            x = data[i, :]

            EfX_condition_1.append(self._value_function(z=1, x=x, sample=num_samples))
            EfX_condition_0.append(self._value_function(z=0, x=x, sample=num_samples))


        EfX_condition_1 = np.array(EfX_condition_1)
        EfX_condition_0 = np.array(EfX_condition_0)


        # compute svs
        PHI_0 = 0.5*(EfX_condition_0 - E_f_X + E_f_X_conditioned_both - EfX_condition_1)
        PHI_1 = 0.5*(EfX_condition_1 - E_f_X + E_f_X_conditioned_both - EfX_condition_0)

        return PHI_0, PHI_1
        





