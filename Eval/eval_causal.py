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
import glob
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
from counterfactual_explanation.flow_ssl.causal_loss import CausalLoss
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
    
    """Flow Model"""
    flow_model_origin = torch.load(configuration_for_proj['fastdcflow_model_targetenc_' + DATA_NAME])
    flow_model_origin = trans_to_device(flow_model_origin)
    flow_model_causal = torch.load(configuration_for_proj['fastdcflow_model_targetenc_causal_' + DATA_NAME])
    flow_model_causal = trans_to_device(flow_model_causal)
    
    fix_dims = [0, 1]
    monotonous_dims = [6]

    """ test input """
    TOTAL_CFS = args.total_cfs
    query_instances = pd.read_csv(configuration_for_proj[DATA_NAME + '_raw_test_input'])
    query_instances_original = query_instances.copy()
    # TargetEncoding
    for feature in encoder_normalize_data_catalog.categoricals:
        query_instances[feature] = query_instances[feature].map(encoder_normalize_data_catalog.cat_dict[feature])
    # Normalization
    query_features = query_instances.drop(columns=[target], axis=1)
    query_labels = query_instances[target].values.astype(np.float32)
    query_features = encoder_normalize_data_catalog.scaler.transform(query_features[feature_names])
    
    args.temperature = args.temperature
    
    # 1. fastdcflow, 2. fastdcflow_causal
    fix_acc_models = []
    mon_acc_models = []
    
    for mdx, flow_name in enumerate(['fastdcflow_model_targetenc_', 'fastdcflow_model_targetenc_causal_']):
        flow_model = torch.load(configuration_for_proj[flow_name + DATA_NAME])
        flow_model = trans_to_device(flow_model)
        fix_accuracy = 0
        fix_acc_lst = []
        monotonous_accuracy = 0
        mon_acc_lst = []
        negative_cnt = 0
        for idx, query_instance in tqdm(enumerate(query_features), total=len(query_features)):
            if negative_cnt >= args.num_inputs:
                break
            query_instance = trans_to_device(torch.Tensor(query_instance))
            test_preds = model_prediction(predictive_model, query_instance).detach().cpu()
            if test_preds.item() >= args.pred_thrsh:
                continue        
            start3 = time.time()
            # generate counterfactuals
            query_instance = query_instance.repeat(TOTAL_CFS, 1)
            test_preds = test_preds.repeat(TOTAL_CFS, 1).squeeze(1)

            x_origin = torch.hstack((query_instance.detach().cpu(), test_preds.unsqueeze(1)))
            x_origin_df = pd.DataFrame(x_origin.numpy(), columns=feature_names + [target])
            x_origin_df = encoder_normalize_data_catalog.denormalize_continuous_feature(x_origin_df)
            x_origin_df = encoder_normalize_data_catalog.convert_from_targetenc_to_original_forms(x_origin_df)
            if mdx == 0:
                CFS_DATA_PATH = configuration_for_proj['cfs_fastdcflow_targetenc_' + DATA_NAME]
            else:
                CFS_DATA_PATH = configuration_for_proj['cfs_fastdcflow_targetenc_causal_' + DATA_NAME]
            csv_files = glob.glob(f'{CFS_DATA_PATH}*.csv')
            x_cf_df = pd.read_csv(csv_files[negative_cnt])[:args.total_cfs]
            # fix_dim については、カテゴリの水準 (文字列) が入力と異なる割合
            fix_temp = 0
            for dim in fix_dims:
                # 各行について、文字列が一致しているかどうかを判定
                fix_temp += sum(x_origin_df.iloc[:, dim] == x_cf_df.iloc[:, dim])
            fix_temp /= len(fix_dims)
            fix_temp /= args.total_cfs
            
            # monotonous_dim については、入力よりCFが大きい割合
            monotonous_temp = 0            
            for dim in monotonous_dims:
                monotonous_temp += sum(x_origin_df.iloc[:, dim] < x_cf_df.iloc[:, dim])
            monotonous_temp /= len(monotonous_dims)
            monotonous_temp /= args.total_cfs
            
            
            fix_accuracy += fix_temp
            fix_acc_lst.append(fix_temp)
            monotonous_accuracy += monotonous_temp
            mon_acc_lst.append(monotonous_temp)
            negative_cnt += 1
            
        fix_accuracy /= negative_cnt
        monotonous_accuracy /= negative_cnt
        fix_acc_models.append(fix_acc_lst)
        mon_acc_models.append(mon_acc_lst)
        print(fix_accuracy, monotonous_accuracy)

    print(np.std(fix_acc_models[0]), np.std(fix_acc_models[1]))
    print(np.std(mon_acc_models[0]), np.std(mon_acc_models[1]))
    np.save(configuration_for_proj['cfs_fastdcflow_npy'] + DATA_NAME + '_fix_accuracy.npy', np.array(fix_acc_models))
    np.save(configuration_for_proj['cfs_fastdcflow_npy'] + DATA_NAME + '_mon_accuracy.npy', np.array(mon_acc_models))