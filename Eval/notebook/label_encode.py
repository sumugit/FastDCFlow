import sys
sys.path.append('/workspace/Eval/')
from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name, load_configuration_from_yaml)
from sklearn.preprocessing import LabelEncoder
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
    negative_cnt = 6
    CONFIG_PATH = '/workspace/Eval/configuration/data_catalog.yaml'
    CONFIG_FOR_PROJECT = '/workspace/Eval/configuration/project_configurations.yaml'
    data_config = load_configuration_from_yaml(CONFIG_PATH)
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    target_encoded_feature_names = data_config[DATA_NAME]['categorical']
    fit_frame = pd.read_csv(configuration_for_proj[DATA_NAME + "_dataset"])
    target_frame = pd.read_csv(configuration_for_proj["cfs_cf_flow_targetenc_" + DATA_NAME] + f"neg_{negative_cnt}.csv")
    le = LabelEncoder()
    target_dict = {}
    for feature in target_encoded_feature_names:
        fit_frame[feature] = le.fit_transform(fit_frame[feature])
        target_frame[feature] = le.transform(target_frame[feature])
        target_frame[feature] = target_frame[feature] + 1
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        target_dict[feature] = mapping
    
    print(target_dict)
    json.dump(target_dict, open(configuration_for_proj["labelenc_dict_" + DATA_NAME], 'w'))
    target_frame.to_csv(configuration_for_proj["cfs_cf_flow_labelenc_" + DATA_NAME] + f"neg_{negative_cnt}.csv", index=False)
    
    