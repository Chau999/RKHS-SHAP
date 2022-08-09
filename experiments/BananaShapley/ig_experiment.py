import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys

from sklearn.datasets import fetch_california_housing


import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from gpytorch.kernels import RBFKernel
from sklearn.datasets import load_boston, load_diabetes

sys.path.append("RKHS-SHAP/src/")
# from
from src.rkhs_shap_exact import RKHSSHAP
# from .. import
from sklearn.model_selection import train_test_split

# import xgboost
import shap
import pandas as pd
import matplotlib.pylab as plt


class KRR(nn.Module):

    def __init__(self, train_X, train_y):
        super().__init__()
        self.k = RBFKernel()
        self.register_buffer('train_X',train_X)
        init_alphas = self.k(train_X).add_diag(torch.tensor(1e-1)).inv_matmul(train_y)
        self.alphas = nn.Parameter(init_alphas)

    def forward(self, test_X):
        K_newXX = self.k(test_X, self.train_X).evaluate()

        return K_newXX @ self.alphas


def extracting_order(attribution, feature_names):
    mean_attribution = pd.Series(np.abs(attribution).mean(axis=0))
    mean_attribution.index = feature_names

    return mean_attribution.sort_values(ascending=False)


def training_krr(features_to_include,
                 k,
                 X_train,
                 X_test,
                 y_train,
                 y_test,
                 feature_names
                 ):
    X_train = pd.DataFrame(X_train)
    X_train.columns = feature_names
    X_test = pd.DataFrame(X_test)
    X_test.columns = feature_names

    X_train = torch.tensor(X_train.loc[:, features_to_include].values).float()
    X_test = torch.tensor(X_test.loc[:, features_to_include].values).float()

    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    ypred = k(X_test, X_train).evaluate() @ k(X_train).add_diag(torch.tensor(1e-1)).inv_matmul(y_train)

    return torch.sqrt(torch.mean((ypred - y_test) ** 2)).detach().numpy()

EXPERIMENT = "housing"


if __name__ == '__main__':

    # Diabetes
    if EXPERIMENT == "housing":
        data_dict = fetch_california_housing()
        epoch=250

    elif EXPERIMENT == "diabetes":
        data_dict = load_diabetes()
        epoch=500
    elif EXPERIMENT == "boston":
        data_dict = load_boston()
        epoch=500

    # Train test split
    X, y = data_dict["data"], data_dict["target"]
    X = X[:2500,:]
    y = y[:2500]
    X_tensor, y_tensor = torch.tensor(X).float(), torch.tensor(y).float()

    ig_ls_ls = []
    rkhs_ls_ls = []
    rkhsO_ls_ls = []

    for i in range(2):

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.33
                                                            )

        X_train_tensor = torch.tensor(X_train).float()
        X_test_tensor = torch.tensor(X_test).float()
        y_train_tensor = torch.tensor(y_train).float()
        y_test_tensor = torch.tensor(y_test).float()

        baselines_train = X_train_tensor.mean(axis=0) * torch.ones_like(X_train_tensor)
        baselines_test = X_test_tensor.mean(axis=0) * torch.ones_like(X_test_tensor)

        krr = KRR(X_train_tensor, y_train_tensor.reshape(-1, 1))
        krr = krr.to('cuda:0')
        optimizer = torch.optim.Adam(krr.parameters(), lr=1e-1, )
        loss_fn = torch.nn.MSELoss()
        for i in range(epoch):
            ypred = krr(X_train_tensor.cuda())
            loss = loss_fn(ypred, y_train_tensor.cuda())
            reg = torch.sum(krr.alphas**2)*1e-2
            print(loss)
            tot_loss = loss+reg
            tot_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # krr = krr.to('cpu')
        krr = krr.eval()
        ig = IntegratedGradients(krr.eval())
        attributions = ig.attribute(inputs=X_test_tensor.cuda(),
                                    baselines=baselines_test.cuda(),
                                    )

        rkhs_shap = RKHSSHAP(X_train_tensor.cuda(),
                             y_train_tensor.cuda(),
                             torch.tensor(1e-1).cuda(),
                             torch.tensor(1e-1).cuda(),
                             lengthscale=krr.k.lengthscale.detach()
                             )

        rkhs_shap_attributions = rkhs_shap.fit(X_test_tensor.cuda(),
                                               method="I",
                                               sample_method="meh",
                                               )

        rkhs_shap_attributions_O = rkhs_shap.fit(X_test_tensor.cuda(),
                                                 method="O",
                                                 sample_method="meh",
                                                 )

        X_df = pd.DataFrame(X)
        X_df.columns = data_dict["feature_names"]

        X_test_df = pd.DataFrame(X_test)
        X_test_df.columns = data_dict["feature_names"]

        # train XGBoost model
        # model = xgboost.XGBRegressor().fit(X_df, y)

        # compute SHAP values
        # explainer = shap.Explainer(model, X_test_df)
        # shap_values = explainer(X_test_df)

        # shap.plots.beeswarm(shap_values, show=False)
        # plt.xlabel("Integrated Gradients with KRR")

        # shap_values.values = attributions
        # shap.plots.beeswarm(shap_values, show=False)
        # plt.xlabel("Integrate Gradients with KRR")
        # plt.show()

        # shap_values.values = rkhs_shap_attributions
        # shap.plots.beeswarm(shap_values, show=False)
        # plt.xlabel("RKHSSHAP")
        # plt.show()

        # shap_values.values = rkhs_shap_attributions_O
        # shap.plots.beeswarm(shap_values, show=False)
        # plt.xlabel("RKHSSHAP Observational")
        # plt.show()

        ig_orders = extracting_order(attributions.cpu().numpy(), data_dict["feature_names"]).index

        rkhs_shap_I_orders = extracting_order(rkhs_shap_attributions, data_dict["feature_names"]).index

        rkhs_shap_O_orders = extracting_order(rkhs_shap_attributions_O, data_dict["feature_names"]).index

        ig_ls, rkhs_ls, rkhsO_ls = [], [], []
        krr = krr.cpu()
        for i in range(len(ig_orders)):
            ig_ls.append(training_krr(ig_orders[i:],
                                      krr.k,
                                      X_train=X_train,
                                      y_train=y_train,
                                      X_test=X_test,
                                      y_test=y_test,
                                      feature_names=data_dict["feature_names"]
                                      ))
            rkhs_ls.append(training_krr(rkhs_shap_I_orders[i:],
                                        krr.k,
                                        X_train=X_train,
                                        y_train=y_train,
                                        X_test=X_test,
                                        y_test=y_test,
                                        feature_names=data_dict["feature_names"]
                                        ))
            rkhsO_ls.append(training_krr(rkhs_shap_O_orders[i:],
                                         krr.k,
                                         X_train=X_train,
                                         y_train=y_train,
                                         X_test=X_test,
                                         y_test=y_test,
                                         feature_names=data_dict["feature_names"]
                                         ))

        ig_ls = np.array(ig_ls)
        rkhs_ls = np.array(rkhs_ls)
        rkhsO_ls = np.array(rkhsO_ls)

        plt.plot(ig_ls, "x--", label="IG")
        plt.plot(rkhs_ls, "o--", label="RKHS-I", alpha=0.5)
        plt.plot(rkhsO_ls, ".--", label="RKHS-O", alpha=0.5)

        plt.title("Affect on test accuracies if we remove features sequentially from IG, RKHS-I, RKHS-O")
        plt.ylabel("RMSE")
        plt.xlabel("Feature ordering")
        plt.legend()
        plt.show()

        ig_ls_ls.append(ig_ls)
        rkhs_ls_ls.append(rkhs_ls)
        rkhsO_ls_ls.append(rkhsO_ls)

plt.errorbar(x=range(len(ig_ls)),
             y=np.mean(ig_ls_ls, axis=0),
             yerr=np.std(ig_ls_ls, axis=0),
             label="IG",
             alpha=0.5
             )

plt.errorbar(x=range(len(ig_ls)),
             y=np.mean(rkhs_ls_ls, axis=0),
             yerr=np.std(rkhs_ls_ls, axis=0),
             label="RKHS-I",
             alpha=0.5
             )

plt.errorbar(x=range(len(ig_ls)),
             y=np.mean(rkhsO_ls_ls, axis=0),
             yerr=np.std(rkhsO_ls_ls, axis=0),
             label="RKHS-O",
             alpha=0.5
             )

plt.xlabel("Feature Ordered by Importance")
plt.ylabel("RMSE on test data")
plt.title("Sequential removal effect based on different attribution methods")
plt.legend()
if not os.path.exists('figures'):
    os.makedirs('figures')
plt.savefig(f"figures/{EXPERIMENT}.pdf")
# plt.show()