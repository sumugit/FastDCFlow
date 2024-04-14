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
    # models = ['dice', 'geco', 'moc', 'ceflow', 'cf_vae', 'cf_cvae', 'cf_duvae', 'fastdcflow']
    models = ['fastdcflow']
    # models = ['fastdcflow', 'cf_vae', 'cf_cvae', 'cf_duvae']
    target_encoded_feature_names = data_config[DATA_NAME]['categorical']
    predictive_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
    DATA_NAME, encoding='targetenc')
    predictive_model.eval()
    predictive_model = predictive_model.cpu()

    np_proximity_lst = []
    np_inner_diversity_lst = []
    np_outer_diversity_lst = []
    np_validity_lst = []
    
    np_val_i_lst = []
    np_dev_i_lst = []
    np_std_i_lst = []
    for model_name in models:
        proximity_loss = 0.0
        inner_diversity = 0.0
        outer_diversity = 0.0
        validity = 0.0
        
        val_i = 0.0
        dev_i = 0.0
        std_i = 0.0
        proximity_lst = []
        inner_diversity_lst = []
        outer_diversity_lst = []
        validity_lst = []
        
        val_i_lst = []
        dev_i_lst = []
        std_i_lst = []
        cfs_lst = []
        CFS_DATA_PATH = configuration_for_proj['cfs_raw_' + model_name + '_targetenc_ortho_' + DATA_NAME]
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
            proximity_lst.append(temp)
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
            inner_diversity_lst.append(temp)
            # get outer diversity
            cfs_lst.append(cfs)
            # get validity
            y_cfs = predictive_model(cfs).detach().numpy().squeeze()
            temp = (y_cfs > test_pred).sum()/len(y_cfs)
            validity += temp
            validity_lst.append(temp)
            negative_cnt += 1
        
        cnt = 0
        for i in range(len(cfs_lst)):
            for j in range(i+1, len(cfs_lst)):
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                temp = -1*cos(torch.mean(cfs_lst[i], axis=0), torch.mean(cfs_lst[j], axis=0)).item()
                outer_diversity += temp
                cnt += 1
                outer_diversity_lst.append(temp)
        
        proximity_loss /= args.num_inputs
        inner_diversity /= args.num_inputs
        outer_diversity /= cnt
        validity /= args.num_inputs
        
        np_proximity_lst.append(proximity_lst)
        np_inner_diversity_lst.append(inner_diversity_lst)
        np_outer_diversity_lst.append(outer_diversity_lst)
        np_validity_lst.append(validity_lst)
        print(f'------------------{model_name}------------------')
        print(f'proximity: {round(proximity_loss, 5)}')
        print(f'proximity_std: {round(np.std(np.array(proximity_lst)), 5)}')
        print(f'validity: {round(validity, 5)}')
        print(f'validity_std: {round(np.std(np.array(validity_lst)), 5)}')
        print(f'inner_diviersity: {round(inner_diversity, 5)}')
        print(f'inner_diviersity_std: {round(np.std(np.array(inner_diversity_lst)), 5)}')
        print(f'outer_diviersity: {round(outer_diversity, 5)}')
        print(f'outer_diviersity_std: {round(np.std(np.array(outer_diversity_lst)), 5)}')

    
    # np.save(configuration_for_proj['cfs_fastdcflow_npy'] + DATA_NAME + '_proximity.npy', np.array(np_proximity_lst))
    # np.save(configuration_for_proj['cfs_fastdcflow_npy'] + DATA_NAME + '_validity.npy', np.array(np_validity_lst))    
    # np.save(configuration_for_proj['cfs_fastdcflow_npy'] + DATA_NAME + '_inner_diversity.npy', np.array(np_inner_diversity_lst))
    # np.save(configuration_for_proj['cfs_fastdcflow_npy'] + DATA_NAME + '_outer_diversity.npy', np.array(np_outer_diversity_lst))