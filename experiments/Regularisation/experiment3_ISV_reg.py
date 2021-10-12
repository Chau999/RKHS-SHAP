import warnings
warnings.filterwarnings("ignore")

import os, sys, pickle, warnings, torch, copy
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
sys.path.append(base_dir)

import numpy as np

from src.shapley_regulariser import ShapleyRegulariser
from src.rkhs_shap_approx import RKHSSHAP_Approx
from src.kernel_ridge_regression import KernelRidgeRegressor
from experiments.Regularisation.correlated_linear_model import CorrelatedLinearModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def generate_simulation(n: int, m: int, beta: np.array):
    """[Generate simple simulation]

    Args:
        n (int): [number of data]
        m (int): [number of features]
        beta (np.array): [the contribution of each feature]
    """

    X = np.random.normal(loc=0, size=(n, m))
    y = X@beta + 1e-2 * np.random.normal(size=n)

    return X, y


def centering_matrix_torch(n):
    one_vec = torch.ones(n).reshape(-1, 1)
    return torch.eye(n) - (one_vec @ one_vec.T)/n


def fit_rkhs_shap(X, y, lr=1e-2, lengthscale=None, epoch=1000, lambda_cme=1e-3, method="O", substract_ref=True):
    # Step 1: Fit my model
    krr = KernelRidgeRegressor(X, y)
    if lengthscale is None:
        lengthscale = torch.tensor(torch.ones(X.shape[1]).float())
    krr.fit(epoch=epoch, verbose=True, lr=lr, lengthscale=lengthscale)
    print("Kernel Lengthscale:", krr.k.lengthscale)

    # Step 2: RUN RKHS-SHAP and plot the distribution of the Shapley Values
    rkhs_shap = RKHSSHAP_Approx(X=X,
                             y=y,
                             lambda_krr=krr.lmda,
                             lambda_cme=lambda_cme,
                             lengthscale=krr.k.lengthscale.detach().numpy(),
                             n_components=200
                            )

    B = rkhs_shap.fit(X_new=X, 
                       method=method,
                       sample_method="full", 
                       substract_ref=substract_ref,
                       num_samples=500, 
                       verbose=True
                      )
    
    return krr, B, rkhs_shap

def fair_learning_with_rkhs_shap(X, y, lambda_sv, lambda_krr, lambda_cme, ls, features_index, sv_method_to_train, sv_method_to_evaluate):
    
    krr_reg = ShapleyRegulariser(lambda_sv=lambda_sv,
                                       lambda_krr=lambda_krr, 
                                       lambda_cme=lambda_cme, 
                                       n_components=400
                                      )

    krr_reg.fit(X=X,
                y=y, 
                ls=ls,
                features_index=features_index, 
                method=sv_method_to_train,
                num_samples=100,
                sample_method="Not MC"
               )

    rkhs_shap = RKHSSHAP_Approx(X=X,
                                y=krr_reg.ypred,
                                lambda_krr=lambda_krr,
                                lambda_cme=lambda_cme,
                                lengthscale=ls,
                                n_components=400
                                )

    B_reg = rkhs_shap.fit(X_new=X, 
                       method=sv_method_to_evaluate,
                       sample_method="full", 
                       num_samples=400, 
                       verbose=True
                      )
    
    return B_reg, krr_reg.RMSE, krr_reg

def run_attack_experiment(n=1300, m=5, noise_ls=[0], sv_reg_ls=[0], verbose=True):
    """[Use Interventional Regularisation to defend against noisy test points.]

    Args:
        n ([int]): [Number of data]
        m ([int]): [Number of features]
        noise ([float]): [noise denoting attacking power on test feature]
        sv_reg ([float]): [regularisation strength on SV of attacked feature]
        verbose ([bool]): [Whether to print statements]

    """

    beta = np.array([(i+1) for i in range(m)])
    beta[-1] = beta[-1]
    mean_vec = np.zeros(m)
    Sigma = np.eye(m)
    Sigma[4, 3] = 0.99
    Sigma[3, 4] = 0.99

    # Linear model
    clm_obj = CorrelatedLinearModel(n=n, noise=3, mean_vec=mean_vec, Sigma=Sigma, beta=beta)
    X, y = clm_obj.X, clm_obj.y    

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # train a krr model
    krr = KernelRidgeRegressor(X, y)
    lengthscale = torch.tensor(torch.ones(X.shape[1]).float())
    krr.fit(epoch=500, verbose=True, lr=1e-1, lengthscale=lengthscale)
    print("Kernel Lengthscale:", krr.k.lengthscale)

    # Run regularisation
    feature_to_ignore = m - 1

    test_rmse_ls, test_rmse_on_attack_ls, rmse_ratio_ls = [], [], []

    for noise in noise_ls:
        for sv_reg in sv_reg_ls:

            X_test_noise = copy.deepcopy(X_test)
            X_test_noise[:, [-1]] += noise * np.random.normal(size=(X_test.shape[0], 1))

            B_reg, train_rmse, new_krr = fair_learning_with_rkhs_shap(X=X_train,
                y=y_train,
                lambda_sv=sv_reg, 
                lambda_krr=krr.lmda, 
                lambda_cme=1e-3, 
                ls=krr.k.lengthscale.detach().numpy(),
                features_index=[feature_to_ignore],
                sv_method_to_train="I",
                sv_method_to_evaluate="I"
                )

            # Compute metric time :D 
            ypred, ypred_attack = new_krr.predict(X_test), new_krr.predict(X_test_noise)
            test_rmse = np.sqrt(mean_squared_error(ypred, y_test))
            test_rmse_on_attack = np.sqrt(mean_squared_error(ypred_attack, y_test))
            rmse_ratio = test_rmse_on_attack / test_rmse

            if verbose:
                print("-----------------------")
                print("With noise %.2f and regularisation %f" %(noise, sv_reg))
                print("RMSE on clean test data: %.2f" %test_rmse)
                print("RMSE on attacked test data: %.2f" %test_rmse_on_attack)
                print("RMSE Ratios: %.2f" %rmse_ratio)
                print("-----------------------")
                print("\n")

            test_rmse_ls.append(test_rmse)
            test_rmse_on_attack_ls.append(test_rmse_on_attack)
            rmse_ratio_ls.append(rmse_ratio)

    return test_rmse_ls, test_rmse_on_attack_ls, rmse_ratio_ls


########################
# Run the experiments  #
########################

MYDIR = "experiments/Regularisation/results"
CHECK_FOLDER = os.path.isdir(MYDIR)

# If folder doesn't exist, then create it.
if not CHECK_FOLDER:
    os.makedirs(MYDIR)

repetition = 10
noise_ls = [0.1, 0.5, 1, 1.5, 2, 2.5]
sv_reg_ls= [0, 1e-1, 0.5, 1, 1.5, 2, 2.5, 5.]

for round in range(repetition):

    test_rmse_ls, test_rmse_on_attack_ls, rmse_ratio_ls = run_attack_experiment(n=1300, m=5, noise_ls=noise_ls, sv_reg_ls=sv_reg_ls)
    result = [noise_ls, sv_reg_ls, test_rmse_ls, test_rmse_on_attack_ls, rmse_ratio_ls, round]

    directory = "experiments/Regularisation/results/attack_exp_%i"%round

    with open(directory, "wb") as f:
        pickle.dump(result, f)