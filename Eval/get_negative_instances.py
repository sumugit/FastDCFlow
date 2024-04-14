from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name, load_configuration_from_yaml)
from torch import nn
import pandas as pd
import numpy as np
import torch
import json
from config.config import Config
import config.setup as setup
from tqdm import tqdm
import glob

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

    query_instances = pd.read_csv(configuration_for_proj[DATA_NAME + '_raw_test_input'])
    query_instances = query_instances.drop(columns=encoder_normalize_data_catalog.target)
    query_instances2 = query_instances.copy()
    # preprocessing
    category_dict = json.load(open(configuration_for_proj[DATA_NAME + '_dataset_targetenc'], 'r'))
    for feature in target_encoded_feature_names:
        if feature == "SeniorCitizen":
            query_instances2[feature] = query_instances2[feature].astype('str')
        query_instances2[feature] = query_instances2[feature].map(category_dict[feature])
    query_instances2 = encoder_normalize_data_catalog.scaler.transform(query_instances2[encoder_normalize_data_catalog.feature_names])
    negative_cnt = 0
    for idx, query in tqdm(enumerate(query_instances2), total=len(query_instances2)):
        if negative_cnt >= args.num_inputs:
            break
        query = torch.tensor(query, dtype=torch.float32)
        test_pred = predictive_model(query).detach().numpy()[0]
        if test_pred >= args.pred_thrsh:
            continue
        if 0.2 <= test_pred < 0.3:
            print(query_instances.iloc[negative_cnt])
            print(test_pred)
            print(f'negative_cnt: {negative_cnt}')
            exit()
        negative_cnt += 1
  