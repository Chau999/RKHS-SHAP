# RKHS-SHAP: Shapley Values for Kernel Methods
The repository contains code for the project "RKHS-SHAP: Shapley Values for Kernel Methods"

To rerun the large scale experiments:

1. Download the LOL datset at https://www.kaggle.com/datasets/paololol/league-of-legends-ranked-matches
2. Then run RKHS_SHAP_preprocess.py to preprocess the data
3. Run debug_train_model_RKHS.py to train the KRR model
4. Run rkhs_shap.py to generate the shapley values for the model and a random subset of data
5. Run post_process_plots_RKHS.py to generate the plots in the appendix



