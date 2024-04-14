import argparse
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

import sys
sys.path.append('/workspace/Eval/')
from config.config import Config
import config.setup as setup
from counterfactual_explanation.utils.mlcatalog import (save_pytorch_model_to_model_path,
                                                        train_one_epoch_batch_data)
from counterfactual_explanation.flow_ssl.flow_loss import FlowLoss
from counterfactual_explanation.flow_ssl.realnvp.realnvp import RealNVPTabular
from counterfactual_explanation.flow_ssl.distributions import SSLGaussMixture
from counterfactual_explanation.utils.data_catalog import (
    DataCatalog, EncoderNormalizeDataCatalog, LabelEncoderNormalizeDataCatalog, TargetEncoderNormalizingDataCatalog,
    TensorDatasetTraning)
from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name, load_configuration_from_yaml)
from counterfactual_explanation.utils.mlcatalog import (
    find_latent_mean_two_classes, model_prediction, negative_prediction_index,
    positive_prediction_index, prediction_instances, make_perturbation,
    get_latent_representation_from_flow_mixed_type,
    original_space_value_from_latent_representation_mixed_type,
    get_latent_representation_from_flow, original_space_value_from_latent_representation)

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

    LR_INIT = args.lr_init
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    PRINT_FREQ = args.print_freq

    means = [data_frame[feature].mean() for feature in feature_names]
    means = torch.tensor([np.array(means).astype(np.float32)])

    features = data_frame[feature_names].values.astype(np.float32)
    features = torch.Tensor(features)
    features_dev = trans_to_device(features)
    labels = model_prediction(predictive_model, features_dev).detach().cpu()
    
    negative_index = negative_prediction_index(labels, 1.0) # pred < 1.0 idx
    negative_instance_features = prediction_instances(features, negative_index) # get negative instances
    negative_labels = prediction_instances(labels, negative_index)
    negative_data = torch.hstack((negative_instance_features, negative_labels))
    negative_data = TensorDatasetTraning(negative_data)
    negative_loader = DataLoader(negative_data, batch_size=BATCH_SIZE, shuffle=True)
    
    
    """Flow Model"""
    PRETRAINED = args.pretrain
    if PRETRAINED:
        flow_model = torch.load(configuration_for_proj['fastdcflow_model_targetenc_' + DATA_NAME])
        flow_model = trans_to_device(flow_model)
        elapsed_time1 = 7.28194
    else:
        print('You need to pretrain flow model')
        exit()
    
    """ test input """
    TOTAL_CFS = args.total_cfs
    
    ts = range(1, 11)
    for t in ts:
        query_instances = pd.read_csv(configuration_for_proj[DATA_NAME + '_raw_test_input'])
        # TargetEncoding
        for feature in encoder_normalize_data_catalog.categoricals:
            query_instances[feature] = query_instances[feature].map(encoder_normalize_data_catalog.cat_dict[feature])
        # Normalization
        query_features = query_instances.drop(columns=[target], axis=1)
        query_labels = query_instances[target].values.astype(np.float32)
        query_features = encoder_normalize_data_catalog.scaler.transform(query_features[feature_names])
        start2 = time.time()
        negative_cnt = 0
        args.temperature = args.temperature
        for query_instance in tqdm(query_features):
            if negative_cnt >= args.num_inputs:
                break
            query_instance = trans_to_device(torch.Tensor(query_instance))
            test_preds = model_prediction(predictive_model, query_instance).detach().cpu()
            if test_preds.item() >= args.pred_thrsh:
                continue
            negative_cnt += 1
            # generate counterfactuals
            query_instance = query_instance.repeat(TOTAL_CFS, 1)
            test_preds = test_preds.repeat(TOTAL_CFS, 1).squeeze(1)
            
            z_value = get_latent_representation_from_flow(flow_model, query_instance)
            # add purturbation
            delta = torch.randn_like(z_value) * t
            perturbed_z_value = make_perturbation(z_value, delta)
            x_cf = original_space_value_from_latent_representation(flow_model, perturbed_z_value)
            y_cf = model_prediction(predictive_model, x_cf).detach().cpu()
            x_cf = torch.hstack((x_cf.detach().cpu(), y_cf))
            x_cf_df = pd.DataFrame(x_cf.numpy(), columns=feature_names + [target])
            x_cf_df.drop(target, axis=1).to_csv(configuration_for_proj["cfs_t_fastdcflow_targetenc_" + DATA_NAME] + f"t_{int(t)}_neg_{negative_cnt}.csv", index=False)
        
        assert negative_cnt == args.num_inputs
        end2 = time.time()
        elapsed_time2 = end2 - start2
        print(f'{args.total_cfs} cfs generation time: {elapsed_time2:.5f}')
        print(f'total time: {elapsed_time1 + elapsed_time2:.5f}')