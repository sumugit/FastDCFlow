import argparse
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/workspace/Eval/')
from config.config import Config
import config.setup as setup

from counterfactual_explanation.flow_ce.flow_method import (
    CounterfactualSimpleBn, FindCounterfactualSample)
from counterfactual_explanation.flow_ssl.realnvp.coupling_layer import (
    Dequantization, DequantizationOriginal)
from counterfactual_explanation.models.classifier import Net
from counterfactual_explanation.utils.data_catalog import (
    DataCatalog, EncoderNormalizeDataCatalog, LabelEncoderNormalizeDataCatalog,
    TensorDatasetTraning)
from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name, load_configuration_from_yaml)
from counterfactual_explanation.utils.mlcatalog import (
    find_latent_mean_two_classes, model_prediction, negative_prediction_index,
    positive_prediction_index, prediction_instances)

def trans_to_device(variable):
    if torch.cuda.is_available() and args.device == 'cuda':
        return variable.cuda()
    else:
        return variable.cpu()

if __name__ == '__main__':
    """Parsing argument"""
    args = setup.setup(Config)
    DATA_NAME = args.data_name
    CONFIG_PATH = '/workspace/Eval/configuration/data_catalog.yaml'
    CONFIG_FOR_PROJECT = '/workspace/Eval/configuration/project_configurations.yaml'
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_train_input']
    predictive_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
        DATA_NAME, encoding='targetenc')
    predictive_model = trans_to_device(predictive_model)
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    feature_names = encoder_normalize_data_catalog.feature_names
    flow_model = torch.load(configuration_for_proj['ceflow_model_targetenc_' + DATA_NAME])

    LR_INIT = args.lr_init
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    PRINT_FREQ = args.print_freq

    """ test input """
    TOTAL_CFS = args.total_cfs
    query_instances = pd.read_csv(configuration_for_proj[DATA_NAME + '_raw_test_input'])
    # TargetEncoding
    for feature in encoder_normalize_data_catalog.categoricals:
        query_instances[feature] = query_instances[feature].map(encoder_normalize_data_catalog.cat_dict[feature])
    # Normalization
    query_features = query_instances.drop(columns=[target], axis=1)
    query_labels = query_instances[target].values.astype(np.float32)
    query_features = encoder_normalize_data_catalog.scaler.transform(query_features[feature_names])
    
    
    result_dict = {}
    negative_cnt = 0
    weight = 0.5
    start = time.time()
    for query_instance in tqdm(query_features):
        if negative_cnt >= args.num_inputs:
            break
        query_instance = trans_to_device(torch.Tensor(query_instance))
        test_preds = model_prediction(predictive_model, query_instance).detach().cpu()
        if test_preds.item() >= args.pred_thrsh:
            continue
        negative_cnt += 1
        counterfactual_instance = CounterfactualSimpleBn(predictive_model, flow_model, weight)
        query_instance = query_instance.repeat(args.total_cfs, 1)
        x_cf = counterfactual_instance.find_counterfactual_via_optimizer(query_instance)
        y_cf = model_prediction(predictive_model, x_cf).detach().cpu()
        # print(query_instance.detach().cpu().shape, test_preds.unsqueeze(1).shape, x_cf.shape, y_cf.shape)
        # x_origin = torch.hstack((query_instance.detach().cpu(), test_preds.unsqueeze(1)))
        # x_origin_df = pd.DataFrame(x_origin.numpy(), columns=feature_names + [target])
        # x_origin_df = encoder_normalize_data_catalog.denormalize_continuous_feature(x_origin_df)
        # x_origin_df = encoder_normalize_data_catalog.convert_from_targetenc_to_original_forms(x_origin_df)
        x_cf = torch.hstack((x_cf.detach().cpu(), y_cf))
        x_cf_df = pd.DataFrame(x_cf.numpy(), columns=feature_names + [target])
        # x_cf_df.drop(target, axis=1).to_csv(configuration_for_proj["cfs_raw_ceflow_targetenc_" + DATA_NAME] + f"neg_{negative_cnt}.csv", index=False)
        x_cf_df = encoder_normalize_data_catalog.denormalize_continuous_feature(x_cf_df)
        x_cf_df = encoder_normalize_data_catalog.convert_from_targetenc_to_original_forms(x_cf_df)
        # x_cf_df.to_csv(configuration_for_proj["cfs_ceflow_targetenc_" + DATA_NAME] + f"neg_{negative_cnt}.csv", index=False)
    end = time.time()
    elapsed_time = end - start
    print(f'{args.total_cfs} cfs generation time: {elapsed_time:.5f}')
