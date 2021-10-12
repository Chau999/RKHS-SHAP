import warnings
warnings.filterwarnings("ignore")

import os, sys, warnings
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
sys.path.append(base_dir)

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import torch
from src.shapley_regulariser import ShapleyRegulariser
from src.rkhs_shap_approx import RKHSSHAP_Approx
from experiments.Regularisation.correlated_linear_model import CorrelatedLinearModel
from sklearn.metrics import pairwise_distances
import copy

MYDIR = "experiments/Regularisation/results"
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    os.makedirs(MYDIR)


# Basic Set up
n, m = 1000, 5
beta = np.array([(i+1) for i in range(m)])
beta[-1] = beta[-1]*3
mean_vec = np.zeros(m)
Sigma = np.eye(m)
Sigma[4, 3] = 0.9
Sigma[3, 4] = 0.9

# Linear model
clm_obj = CorrelatedLinearModel(n=n, noise=.1, mean_vec=mean_vec, Sigma=Sigma, beta=beta, random_state=1234)
X, y = clm_obj.X, clm_obj.y

B_true = clm_obj.kernelSHAP(X)

compute_mh = lambda X: np.array([np.median(pairwise_distances(X[:, [i]])) for i in range(X.shape[1])])

BO_ls = []
lambda_sv_ls = [0, 2, 5, 10, 20]

for lambda_sv in lambda_sv_ls:

    krr_reg = ShapleyRegulariser(lambda_sv=lambda_sv,
                                lambda_krr=1e-2,
                                lambda_cme=1e-3,
                                n_components=600)

    krr_reg.fit(X=X, y=y, ls=compute_mh(X), method="O",num_samples=100, sample_method="Not MC",features_index=[4])

    rkhs_shap = RKHSSHAP_Approx(X=X, 
                             y=krr_reg.ypred,
                             lambda_krr=1e-2,
                             lambda_cme=1e-3,
                             lengthscale=compute_mh(X),
                             n_components=600)

    B = rkhs_shap.fit(X_new=X, method="O", sample_method="full", verbose=True)
    BO_ls.append(B)


BI_ls = []
lambda_sv_ls = [0,1, 1.5, 5, 20]

for lambda_sv in lambda_sv_ls:

    krr_reg = ShapleyRegulariser(lambda_sv=lambda_sv,
                                lambda_krr=1e-2,
                                lambda_cme=1e-3,
                                n_components=600)

    krr_reg.fit(X=X, y=y, ls=compute_mh(X), method="I",num_samples=100, sample_method="Not MC",features_index=[4])

    rkhs_shap = RKHSSHAP_Approx(X=X, 
                             y=krr_reg.ypred,
                             lambda_krr=1e-2,
                             lambda_cme=1e-3,
                             lengthscale=compute_mh(X),
                             n_components=600)

    B = rkhs_shap.fit(X_new=X, method="I", sample_method="full", verbose=True)
    BI_ls.append(B)


f, ax = plt.subplots(1, 4, figsize=(8, 3), sharey=True, sharex=True)
new_color_palette = sns.color_palette("RdPu", len(BI_ls))
new_color_palette2 = sns.color_palette("Blues", len(BI_ls))

for i in range(len(BI_ls)):

    sns.distplot(BO_ls[i][:, 3], color=new_color_palette[i], ax=ax[2], hist=False,kde_kws=dict(linewidth=2.5))
    sns.distplot(BO_ls[i][:, 4], color=new_color_palette[i], ax=ax[3], hist=False,kde_kws=dict(linewidth=2.5))    
    
    sns.distplot(BI_ls[i][:, 3], color=new_color_palette2[i], ax=ax[0], hist=False,kde_kws=dict(linewidth=2.5))
    sns.distplot(BI_ls[i][:, 4], color=new_color_palette2[i], ax=ax[1], hist=False,kde_kws=dict(linewidth=2.5))
    
    ax[0].set_xlabel("$\phi^{(I)}_{X,4}(f_{reg})$", size=20)
    ax[1].set_xlabel("$\phi^{(I)}_{X,5}(f_{reg})$", size=20)
    ax[2].set_xlabel("$\phi^{(O)}_{X,4}(f_{reg})$", size=20)
    ax[3].set_xlabel("$\phi^{(O)}_{X,5}(f_{reg})$", size=20)    
    
    ax[0].set_title("ISV-Reg", size=20)
    ax[1].set_title("ISV-Reg", size=20)
    ax[2].set_title("OSV-Reg", size=20)
    ax[3].set_title("OSV-Reg", size=20)
    
    f.tight_layout()

f.savefig("experiments/Regularisation/results/reg_trend.pdf")