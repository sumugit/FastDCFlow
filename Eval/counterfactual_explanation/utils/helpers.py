import os
from typing import Dict

import numpy as np
import torch
import yaml

import sys
sys.path.append('/workspace/Eval/')
from counterfactual_explanation.utils.data_catalog import (DataCatalog, LabelEncoderNormalizeDataCatalog, 
EncoderNormalizeDataCatalog, TargetEncoderNormalizingDataCatalog, load_target_features_name)
from counterfactual_explanation.utils.mlcatalog import load_pytorch_prediction_model_from_model_path


def load_configuration_from_yaml(config_path):
    with open(config_path, 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return conf


def load_hyperparameter_for_method(path, method, data_name) -> Dict:
    setup_catalog = load_configuration_from_yaml(path)
    hyperparameter = setup_catalog['recourse_methods'][method]["hyperparams"]
    hyperparameter["data_name"] = data_name
    return hyperparameter


def load_all_configuration_with_data_name(DATA_NAME, encoding=None):
    CONFIG_PATH = '/workspace/Eval/configuration/data_catalog.yaml'
    CONFIG_FOR_PROJECT = '/workspace/Eval/configuration/project_configurations.yaml'
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_train_input']

    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)

    if encoding == "targetenc":
        encoder_normalize_data_catalog = TargetEncoderNormalizingDataCatalog(data_catalog)
    elif encoding == "onehotenc":
        encoder_normalize_data_catalog = EncoderNormalizeDataCatalog(data_catalog)


    if encoding == "targetenc":
        predictive_model_path = configuration_for_proj['trained_models_targetenc_' + DATA_NAME]
    elif encoding == "onehotenc":
        predictive_model_path = configuration_for_proj['trained_models_onehotenc_' + DATA_NAME]
        
    predictive_model = load_pytorch_prediction_model_from_model_path(predictive_model_path)
    predictive_model = predictive_model.cuda()

    return predictive_model, encoder_normalize_data_catalog, configuration_for_proj