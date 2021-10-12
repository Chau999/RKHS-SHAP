###############
# Experiment 2#
###############

import os, sys, torch, shap, time, pickle, warnings
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
sys.path.append(base_dir)

import numpy as np
from experiments.BananaShapley.banana_distribution import Banana2d

warnings.filterwarnings("ignore")
from experiments.BananaShapley.gshap_banana import Observation2dBanana
from src.rkhs_shap_exact import RKHSSHAP as RKHS_SHAP
from sklearn.metrics import pairwise_distances


# For experiment 2

n_ls = [100, 500, 1000, 1500, 3000, 5000]
v = 10
b = 1.0
iterations = 10
result = []

for iter in range(iterations):
    for n in n_ls:
        
        banana2d = Banana2d(n=n, v=v, b=b, noise=0, outlier_quantile=3)
        
        scale = 1
        y = banana2d.y/scale
        X = banana2d.X

        compute_mh = lambda X: np.array([np.median(pairwise_distances(X[:, [i]])) for i in range(X.shape[1])])
        lengthscale = torch.tensor(compute_mh(X)).float()
        lengthscale[1] *= 1
        print("Lengthscale:", lengthscale)

        # True OSVs:
        phi1 = banana2d.phi_1/scale
        phi2 = banana2d.phi_2/scale
        PHI = np.array([phi1, phi2]).T

        # Compute RKHSSHAP
        X_ten, y_ten = torch.tensor(X).float(), torch.tensor(y).float().reshape(-1,1)
        lambda_krr, lambda_cme = torch.tensor(1e-3), torch.tensor(1e-3)

        rkhs_shap = RKHS_SHAP(X=X_ten,
                            y=y_ten,
                            lambda_krr=lambda_krr,
                            lambda_cme=lambda_cme,
                            lengthscale=lengthscale)
        print("RMSE: ", rkhs_shap.rmse)

        # Set up the model for Model Agnostic SHAP
        def predict_for_shap(X_new):
            pred = rkhs_shap.k(torch.tensor(X_new).float(), rkhs_shap.X_scaled).evaluate()@rkhs_shap.alphas
            return pred.detach().numpy().reshape(-1)
        
        # Evaluate Shapley VALUES

        # RKHS SHAP
        start_time = time.time()
        B_I = rkhs_shap.fit(X_new=X_ten, method="I", sample_method="full", num_samples=100, wls_reg=0)
        end_time = time.time()
        B_I_time = end_time - start_time

        start_time = time.time()
        B_O = rkhs_shap.fit(X_new=X_ten, method="O", sample_method="full", num_samples=100, wls_reg=0)
        end_time = time.time()
        B_O_time = end_time - start_time

        # KernelSHAP
        X_scaled = X/compute_mh(X)

        start_time = time.time()
        explainer = shap.KernelExplainer(predict_for_shap, X_scaled)
        shap_values = explainer.shap_values(X_scaled, nsamples=n)
        end_time = time.time()
        KSHAP_time = end_time - start_time


        # Multivariate Guassian
        start_time = time.time()
        ogshap = Observation2dBanana(predict_for_shap, X_scaled)
        ophi1, ophi2 = ogshap.fit(X_scaled, num_samples=n)    
        end_time = time.time()
        OGSHAP_time = end_time - start_time
        OPHI = np.array([ophi1,ophi2]).T

        # Collect Result
        print("At iteration %i" %iter)
        print("At n=%f" %n)
        print("ISV takes: %.2f"%B_I_time)
        print("OSV takes: %.2f"%B_O_time)
        print("KSHAP takes: %.2f"%KSHAP_time)
        print('OGSHAP takes: %.2f'%OGSHAP_time)
        print("\n")


        result.append([B_I_time, B_O_time, KSHAP_time, OGSHAP_time, n, iter])


MYDIR = "experiments/BananaShapley/results"
CHECK_FOLDER = os.path.isdir(MYDIR)

# If folder doesn't exist, then create it.
if not CHECK_FOLDER:
    os.makedirs(MYDIR)

with open("experiments/BananaShapley/results/exp2.pkl", "wb") as f:
    pickle.dump(result, f)
