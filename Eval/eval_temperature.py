import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/workspace/Eval/')
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
args = setup.setup(Config)
CONFIG_PATH = '/workspace/Eval/configuration/data_catalog.yaml'
CONFIG_FOR_PROJECT = '/workspace/Eval/configuration/project_configurations.yaml'
TEST_INPUT = '/workspace/Eval/configuration/test_input.yaml'
data_config = load_configuration_from_yaml(CONFIG_PATH)
configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)


data_names = ['adult', 'bank', 'churn']
models = ['cf_vae', 'cf_cvae', 'cf_duvae', 'fastdcflow']
ts = range(1, 11)
data_dict = {}

for DATA_NAME in data_names:
    target_encoded_feature_names = data_config[DATA_NAME]['categorical']
    predictive_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
    DATA_NAME, encoding='targetenc')
    predictive_model.eval()
    predictive_model = predictive_model.cpu()

    model_proximity = []
    model_validity = []
    model_inner_diversity = []
    model_outer_diversity = []
    for model in models:
        proximity_loss_lst = []
        validity_lst = []
        inner_diversity_lst = []
        outer_diversity_lst = []
        cfs_lst = []
        for t in ts:
            proximity_loss = 0.0
            inner_diversity = 0.0
            outer_diversity = 0.0
            validity = 0.0
            cfs_lst = []
            CFS_DATA_PATH = configuration_for_proj['cfs_t_' + model + '_targetenc_' + DATA_NAME] + f't_{t}_'
            csv_files = glob.glob(f'{CFS_DATA_PATH}*.csv')
            query_instances = pd.read_csv(configuration_for_proj[DATA_NAME + '_raw_test_input'])
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
                # get proximity loss
                cfs = np.array(cfs.values, dtype=np.float32)
                cfs = torch.tensor(cfs, dtype=torch.float32)
                temp = torch.cdist(cfs, query.unsqueeze(0), p=2).mean().detach().numpy()
                proximity_loss += -1*temp
                # get inner diversity
                temp = 0
                cnt = 0
                for i in range(len(cfs)):
                    for j in range(i+1, len(cfs)):
                        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                        temp += -1*cos(cfs[i], cfs[j]).item()
                        cnt += 1
                temp /= cnt
                inner_diversity += temp
                # get outer diversity
                cfs_lst.append(cfs)
                # get validity
                y_cfs = predictive_model(cfs).detach().numpy().squeeze()
                temp = (y_cfs > test_pred).sum()/len(y_cfs)
                validity += temp
                negative_cnt += 1
            
            assert negative_cnt == args.num_inputs
            cnt = 0
            for i in range(len(cfs_lst)):
                for j in range(i+1, len(cfs_lst)):
                    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                    temp = -1*cos(torch.mean(cfs_lst[i], axis=0), torch.mean(cfs_lst[j], axis=0)).item()
                    outer_diversity += temp
                    cnt += 1
            
            proximity_loss /= args.num_inputs
            inner_diversity /= args.num_inputs
            outer_diversity /= cnt
            validity /= args.num_inputs
            proximity_loss_lst.append(proximity_loss)
            inner_diversity_lst.append(inner_diversity)
            outer_diversity_lst.append(outer_diversity)
            validity_lst.append(validity)
        
        assert len(proximity_loss_lst) == len(ts)
        model_proximity.append(proximity_loss_lst)
        model_inner_diversity.append(inner_diversity_lst)
        model_outer_diversity.append(outer_diversity_lst)
        model_validity.append(validity_lst)
    
    assert len(model_proximity) == len(models)
    data_dict[DATA_NAME] = {
        'proximity': model_proximity,
        'validity': model_validity,
        'inner_diversity': model_inner_diversity,
        'outer_diversity': model_outer_diversity,
    }

json.dump(data_dict, open(configuration_for_proj['cfs_fastdcflow_npy'] + 'temperature_dict.json', 'w'))