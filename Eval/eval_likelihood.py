from counterfactual_explanation.utils.helpers import (
load_all_configuration_with_data_name, load_configuration_from_yaml)
from counterfactual_explanation.utils.mlcatalog import load_pytorch_prediction_model_from_model_path
from counterfactual_explanation.flow_ssl.likelihood import Likelihood
from counterfactual_explanation.flow_ssl.distributions import SSLGaussMixture

from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name, load_configuration_from_yaml)
from counterfactual_explanation.utils.mlcatalog import (
    model_prediction, get_latent_representation_from_flow)

import pandas as pd
import numpy as np
import torch
import json
from config.config import Config
import config.setup as setup
from tqdm import tqdm
import glob
import warnings
warnings.filterwarnings('ignore')

def trans_to_device(variable):
    if torch.cuda.is_available() and args.device == 'cuda':
        return variable.cuda()
    else:
        return variable.cpu()

if __name__ == '__main__':
    args = setup.setup(Config)
    DATA_NAME = args.data_name
    CONFIG_PATH = '/workspace/Eval/configuration/data_catalog.yaml'
    CONFIG_FOR_PROJECT = '/workspace/Eval/configuration/project_configurations.yaml'
    TEST_INPUT = '/workspace/Eval/configuration/test_input.yaml'
    data_config = load_configuration_from_yaml(CONFIG_PATH)
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    target_encoded_feature_names = data_config[DATA_NAME]['categorical']
    predictive_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
    DATA_NAME, encoding='targetenc')
    predictive_model.eval()
    predictive_model = predictive_model.cpu()
    
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    feature_names = encoder_normalize_data_catalog.feature_names
    means = [data_frame[feature].mean() for feature in feature_names]
    means = torch.tensor([np.array(means).astype(np.float32)])
    
    fastdcflow = load_pytorch_prediction_model_from_model_path(configuration_for_proj['fastdcflow_model_targetenc_causal_' + DATA_NAME])
    fastdcflow.eval()
    fastdcflow = fastdcflow.cpu()
    prior = SSLGaussMixture(means=means, device=args.device)
    loss_fn = Likelihood(prior)
    loss_fn.eval()
    loss_fn = loss_fn.cpu()

    CFS_DATA_PATH = configuration_for_proj['cfs_raw_fastdcflow_targetenc_' + DATA_NAME]
    csv_files = glob.glob(f'{CFS_DATA_PATH}*.csv')
    query_instances = pd.read_csv(configuration_for_proj[DATA_NAME + '_raw_test_input'])
    original_query_instances = query_instances.copy()
    query_instances = query_instances.drop(columns=encoder_normalize_data_catalog.target)
    # preprocessing
    category_dict = json.load(open(configuration_for_proj[DATA_NAME + '_dataset_targetenc'], 'r'))
    for feature in target_encoded_feature_names:
        if feature == "SeniorCitizen":
            query_instances[feature] = query_instances[feature].astype('str')
        query_instances[feature] = query_instances[feature].map(category_dict[feature])
    query_instances = encoder_normalize_data_catalog.scaler.transform(query_instances[encoder_normalize_data_catalog.feature_names])
    negative_cnt = 0
    for idx, query in tqdm(enumerate(query_instances), total=len(query_instances)):
        if negative_cnt >= args.num_inputs:
            break
        query = torch.tensor(query, dtype=torch.float32)
        test_pred = predictive_model(query).detach().numpy()[0]
        if test_pred >= args.pred_thrsh:
            continue
        # load cfs
        cfs = pd.read_csv(csv_files[negative_cnt])[:args.total_cfs]
        cfs = np.array(cfs.values, dtype=np.float32)
        cfs = torch.tensor(cfs, dtype=torch.float32)

        local_batch = trans_to_device(cfs)
        z_value = get_latent_representation_from_flow(fastdcflow, local_batch)
        sldj = fastdcflow.logdet()
        flow_loss = loss_fn(z_value, sldj)
        score_index = torch.argsort(flow_loss)
        cfs = cfs[score_index]
        negative_cnt += 1
        
        if 0.1 < test_pred < 0.2 and negative_cnt == 109:
            x_origin = torch.hstack((query.detach().cpu(), torch.tensor(test_pred).detach().cpu()))
            x_origin_df = pd.DataFrame([x_origin.numpy()], columns=feature_names + [target])
            x_origin_df = encoder_normalize_data_catalog.denormalize_continuous_feature(x_origin_df)
            x_origin_df = encoder_normalize_data_catalog.convert_from_targetenc_to_original_forms(x_origin_df)
            print("--- query instance ---")
            print(negative_cnt)
            print(x_origin_df)
            # concat (cfs, y_cf)
            y_cf = model_prediction(predictive_model, cfs).detach().cpu()
            cfs = torch.cat([cfs, y_cf], dim=1)
            x_cf_df = pd.DataFrame(cfs.numpy(), columns=feature_names + [target])
            x_cf_df = encoder_normalize_data_catalog.denormalize_continuous_feature(x_cf_df)
            x_cf_df = encoder_normalize_data_catalog.convert_from_targetenc_to_original_forms(x_cf_df)
            print("--- counterfactuals ---")
            print(x_cf_df.head())
            exit() 