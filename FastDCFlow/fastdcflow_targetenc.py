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
    positive_prediction_index, prediction_instances, make_perturbation, generate_orthogonal_noise,
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
    stds = [data_frame[feature].std() for feature in feature_names]
    stds = torch.tensor([np.array(stds).astype(np.float32)])

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
    # set up flow model and loss function
    flow_model = RealNVPTabular(num_coupling_layers=3, in_dim=features.shape[1], num_layers=5, hidden_dim=12)
    prior = SSLGaussMixture(means=means, inv_cov_stds=stds, device=args.device)
    loss_fn = FlowLoss(prior) # flow loss
    loss_y = nn.BCELoss() # prediction loss
    loss_prox = nn.MSELoss() # distance loss
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=LR_INIT, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.01)
    best_model = None
    best_loss = 100000
    PRETRAINED = args.pretrain
    if PRETRAINED:
        flow_model = torch.load(configuration_for_proj['fastdcflow_model_targetenc_' + DATA_NAME])
        flow_model = trans_to_device(flow_model)
        elapsed_time1 = 0
    else:
        # histroy = defaultdict(lambda: [])

        """train flow model"""
        start1 = time.time()
        for epoch in tqdm(range(1, EPOCHS+1)):
            temp_flow_loss = 0
            temp_y_loss = 0
            temp_prox_loss = 0
            temp_total_loss = 0
            for local_batch, local_labels in (negative_loader):
                local_batch = trans_to_device(local_batch)
                local_labels = trans_to_device(local_labels)
                z_value = get_latent_representation_from_flow(flow_model, local_batch)
                # add purturbation
                delta = torch.randn_like(z_value)
                perturbed_z_value = make_perturbation(z_value, delta)
                x_cf = original_space_value_from_latent_representation(flow_model, perturbed_z_value)
                y_cf = model_prediction(predictive_model, x_cf).reshape(-1)
                sldj = flow_model.logdet()
                y_expected = trans_to_device(torch.ones(local_labels.shape, dtype=torch.float).reshape(-1))
                # loss func
                flow_loss = loss_fn(z_value, sldj)
                y_loss = loss_y(y_cf, y_expected)
                prox_loss = loss_prox(local_batch, x_cf)
                total_loss = args.lambda_1*flow_loss + y_loss + prox_loss
                # print('flow_loss: {}, y_loss: {}, prox_loss: {}'.format(flow_loss, y_loss, prox_loss))
                temp_flow_loss += flow_loss.item()
                temp_y_loss += y_loss.item()
                temp_prox_loss += prox_loss.item()
                temp_total_loss += total_loss.item()
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            # histroy['flow_loss'].append(temp_flow_loss / len(negative_loader))
            # histroy['y_loss'].append(temp_y_loss / len(negative_loader))
            # histroy['prox_loss'].append(temp_prox_loss / len(negative_loader))
            # histroy['total_loss'].append(temp_total_loss / len(negative_loader))
            scheduler.step()
            cur_lr = scheduler.optimizer.param_groups[0]['lr']
            if total_loss < best_loss:
                best_loss = total_loss
                best_model = flow_model
            # if epoch % PRINT_FREQ == 0:
            #     # print("\n Epoch {}, Loss {:.4f}, Learning rate {:.4f}".format(epoch, total_loss, cur_lr))
            

        end1 = time.time()
        elapsed_time1 = end1 - start1
        print(f'pretrain time: {elapsed_time1:.5f}s')
        flow_model = best_model
        save_pytorch_model_to_model_path(flow_model, configuration_for_proj['fastdcflow_model_targetenc_' + DATA_NAME])
        # with open(configuration_for_proj['fastdcflow_model_targetenc_' + DATA_NAME + '_history'], 'w') as f:
        #     json.dump(histroy, f, indent=2)

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
    start2 = time.time()
    negative_cnt = 0
    times = []
    args.temperature = args.temperature
    for query_instance in tqdm(query_features):
        if negative_cnt >= args.num_inputs:
            break
        query_instance = trans_to_device(torch.Tensor(query_instance))
        test_preds = model_prediction(predictive_model, query_instance).detach().cpu()
        if test_preds.item() >= args.pred_thrsh:
            continue        
        negative_cnt += 1
        start3 = time.time()
        # generate counterfactuals
        query_instance = query_instance.repeat(TOTAL_CFS, 1)
        test_preds = test_preds.repeat(TOTAL_CFS, 1).squeeze(1)
        
        z_value = get_latent_representation_from_flow(flow_model, query_instance)
        # add purturbation
        delta = torch.randn_like(z_value) * args.temperature
        perturbed_z_value = make_perturbation(z_value, delta)
        x_cf = original_space_value_from_latent_representation(flow_model, perturbed_z_value)
        y_cf = model_prediction(predictive_model, x_cf).detach().cpu()
        y_expected = trans_to_device(torch.ones(y_cf.shape, dtype=torch.float).reshape(-1))
        y_loss = loss_y(y_cf.reshape(-1), y_expected)
        prox_loss = loss_prox(query_instance, x_cf)
        eval_loss = y_loss + prox_loss
        x_origin = torch.hstack((query_instance.detach().cpu(), test_preds.unsqueeze(1)))
        x_origin_df = pd.DataFrame(x_origin.numpy(), columns=feature_names + [target])
        x_origin_df = encoder_normalize_data_catalog.denormalize_continuous_feature(x_origin_df)
        x_origin_df = encoder_normalize_data_catalog.convert_from_targetenc_to_original_forms(x_origin_df)
        x_cf = torch.hstack((x_cf.detach().cpu(), y_cf))
        x_cf_df = pd.DataFrame(x_cf.numpy(), columns=feature_names + [target])
        x_cf_df["eval_loss"] = eval_loss.detach().cpu().numpy()
        x_cf_df = x_cf_df.sort_values(by="eval_loss", ascending=True)
        x_cf_df = x_cf_df.drop("eval_loss", axis=1)
        x_cf_df.drop(target, axis=1).to_csv(configuration_for_proj["cfs_raw_fastdcflow_targetenc_" + DATA_NAME] + f"neg_{negative_cnt}.csv", index=False)        
        x_cf_df = encoder_normalize_data_catalog.denormalize_continuous_feature(x_cf_df)
        x_cf_df = encoder_normalize_data_catalog.convert_from_targetenc_to_original_forms(x_cf_df)
        x_cf_df.to_csv(configuration_for_proj["cfs_fastdcflow_targetenc_" + DATA_NAME] + f"neg_{negative_cnt}.csv", index=False)
        end3 = time.time()
        times.append(end3 - start3)
        # print(f'negative_cnt: {negative_cnt}, time: {end3 - start3:.5f}')
    
    
    end2 = time.time()
    elapsed_time2 = end2 - start2
    print(f'{args.total_cfs} cfs generation time: {elapsed_time2:.5f}')
    print(f'total time: {(elapsed_time1 + elapsed_time2)/args.num_inputs:.5f}')
    # with open(configuration_for_proj['fastdcflow_model_targetenc_' + DATA_NAME + '_times'], 'w') as f:
    #     json.dump(times, f)