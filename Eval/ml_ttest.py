import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import json

from counterfactual_explanation.models.classifier import train_predictive_model, Net
from counterfactual_explanation.utils.data_catalog import (DataCatalog, EncoderNormalizeDataCatalog,
                                                           LabelEncoderNormalizeDataCatalog,
                                                           TargetEncoderNormalizingDataCatalog,
                                                           TensorDatasetTraning)
from counterfactual_explanation.utils.helpers import load_configuration_from_yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from config.config import Config
import config.setup as setup
from scipy import stats

if __name__ == '__main__':
    args = setup.setup(Config)
    DATA_NAME = args.data_name
    CONFIG_PATH = "/workspace/Eval/configuration/data_catalog.yaml"
    CONFIG_FOR_PROJECT = "/workspace/Eval/configuration/project_configurations.yaml"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    with open(configuration_for_proj['accuracy_per_fold_onehotenc_' + DATA_NAME], 'r') as f:
        accuracy_ohe = json.load(f)
    with open(configuration_for_proj['auc_per_fold_onehotenc_' + DATA_NAME], 'r') as f:
        auc_ohe = json.load(f)
    with open(configuration_for_proj['accuracy_per_fold_targetenc_' + DATA_NAME], 'r') as f:
        accuracy_te = json.load(f)
    with open(configuration_for_proj['auc_per_fold_targetenc_' + DATA_NAME], 'r') as f:
        auc_te = json.load(f)
    print(f"Accuracy onehotenc: {np.mean(accuracy_ohe)}")
    print(f"Accuracy onehotenc std: {np.std(accuracy_ohe)}")
    print(f"AUC onehotenc: {np.mean(auc_ohe)}")
    print(f"AUC onehotenc std: {np.std(auc_ohe)}")
    print(f"Accuracy targetenc: {np.mean(accuracy_te)}")
    print(f"Accuracy targetenc std: {np.std(accuracy_te)}")
    print(f"AUC targetenc: {np.mean(auc_te)}")
    print(f"AUC targetenc std: {np.std(auc_te)}")
    print(stats.ttest_ind(accuracy_ohe, accuracy_te))
    print(stats.ttest_ind(auc_ohe, auc_te))
    