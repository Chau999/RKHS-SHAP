###############
# Experiment  #
###############


import os, sys, pickle, warnings, torch, shap
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
sys.path.append(base_dir)

import numpy as np
from experiments.BananaShapley.banana_distribution import Banana2d
from experiments.BananaShapley.gshap_banana import Observation2dBanana

warnings.filterwarnings("ignore")
from src.rkhs_shap_exact import RKHSSHAP as RKHS_SHAP
from sklearn.metrics import mean_squared_error, pairwise_distances, r2_score
import copy


# For experiment 1

n = 1000
v = 10
b_ls = [0.01, 0.02, 0.05, 0.1, 1]
iterations = 10
result = []

for iter in range(iterations):
    for b in b_ls:
        
        banana2d = Banana2d(n=n, v=v, b=b, noise=0, outlier_quantile=2.)
        
        # Scaled the output so that can we compare accuracy across
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

        # True ISVs:
        phi1_I = banana2d.phi_1_I
        phi2_I = banana2d.phi_2_I
        PHI_I = np.array([phi1_I, phi2_I]).T

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
        B_I = rkhs_shap.fit(X_new=X_ten, method="I", sample_method="full", num_samples=100, wls_reg=0)
        B_O = rkhs_shap.fit(X_new=X_ten, method="O", sample_method="full", num_samples=100, wls_reg=0)

        # KernelSHAP
        X_scaled = X/compute_mh(X)
        explainer = shap.KernelExplainer(predict_for_shap, shap.kmeans(X_scaled, n))
        shap_values = explainer.shap_values(X_scaled, nsamples=n)

        # Multivariate Guassian
        ogshap = Observation2dBanana(predict_for_shap, X_scaled)
        ophi1, ophi2 = ogshap.fit(X_scaled, num_samples=n)    
        OPHI = np.array([ophi1,ophi2]).T

        # Collect Result for OSV
        ISV_error = np.sqrt(mean_squared_error(PHI, B_I))
        OSV_error = np.sqrt(mean_squared_error(PHI, B_O))
        KSHAP_error = np.sqrt(mean_squared_error(PHI, shap_values))
        Gaussian_error = np.sqrt(mean_squared_error(PHI, OPHI))

        # Collect Result for OSV in R2
        ISV_r2 = r2_score(PHI, B_I)
        OSV_r2 = r2_score(PHI, B_O)
        KSHAP_r2 = r2_score(PHI, shap_values)
        Gaussian_r2 = r2_score(PHI, OPHI)

        # Collect Result for ISV
        ISV_error_I = np.sqrt(mean_squared_error(PHI_I, B_I))
        OSV_error_I = np.sqrt(mean_squared_error(PHI_I, B_O))
        KSHAP_error_I = np.sqrt(mean_squared_error(PHI_I, shap_values))
        Gaussian_error_I = np.sqrt(mean_squared_error(PHI_I, OPHI))

        # Collect Result for ISV in R2
        ISV_r2_I = r2_score(PHI_I, B_I)
        OSV_r2_I = r2_score(PHI_I, B_O)
        KSHAP_r2_I = r2_score(PHI_I, shap_values)
        Gaussian_r2_I = r2_score(PHI_I, OPHI)


        # Printing Metric
        print("At iteration %i" %iter)
        print("At b: %f"%b)
        print("In estimating ISV: \n")
        # print("RKHS-OSV RMSE: %.2f, R2: %.2f"%(OSV_error_I, OSV_r2_I))
        print("RKHS-ISV RMSE: %.2f, R2: %.2f"%(ISV_error_I, ISV_r2_I))
        print("KSHAP-ISV RMSE: %.2f, R2: %.2f"%(KSHAP_error_I, KSHAP_r2_I))
        print("\n")
        print("In estimating OSV: \n")
        print("RKHS-OSV RMSE: %.2f, R2: %.2f"%(OSV_error, OSV_r2))
        print("GSHAP-OSV RMSE: %.2f, R2: %.2f"%(Gaussian_error, Gaussian_r2))
        print("\n")


        result.append([ISV_r2_I, KSHAP_r2_I, OSV_r2, Gaussian_r2, b, iter])


# You should change 'test' to your preferred folder.
MYDIR = "experiments/BananaShapley/results"
CHECK_FOLDER = os.path.isdir(MYDIR)

# If folder doesn't exist, then create it.
if not CHECK_FOLDER:
    os.makedirs(MYDIR)

with open("experiments/BananaShapley/results/exp1.pkl", "wb") as f:
    pickle.dump(result, f)