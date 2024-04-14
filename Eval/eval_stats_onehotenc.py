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
    models = ['fastdcflow']
    target_encoded_feature_names = data_config[DATA_NAME]['categorical']
    predictive_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
    DATA_NAME, encoding='onehotenc')
    predictive_model.eval()
    predictive_model = predictive_model.cpu()

    np_proximity_lst = []
    np_inner_diversity_lst = []
    np_outer_diversity_lst = []
    np_success_rate_lst = []

    np_val_i_lst = []
    np_dev_i_lst = []
    np_std_i_lst = []
    for model_name in models:
        proximity_loss = 0.0
        inner_diversity = 0.0
        outer_diversity = 0.0
        success_rate = 0.0

        val_i = 0.0
        dev_i = 0.0
        std_i = 0.0
        proximity_lst = []
        inner_diversity_lst = []
        outer_diversity_lst = []
        success_rate_lst = []

        val_i_lst = []
        dev_i_lst = []
        std_i_lst = []
        cfs_lst = []
        CFS_DATA_PATH = configuration_for_proj['cfs_raw_' + model_name + '_onehotenc_' + DATA_NAME]
        csv_files = glob.glob(f'{CFS_DATA_PATH}*.csv')
        query_instances = pd.read_csv(configuration_for_proj[DATA_NAME + '_onehotenc_test_input'])
        query_instances = query_instances.drop(columns=encoder_normalize_data_catalog.target)
        query_instances = query_instances.values
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
            proximity_loss += temp
            proximity_lst.append(temp)
            # get inner diversity
            temp = 0
            # get outer diversity
            cfs_lst.append(cfs)
            # get success rate
            y_cfs = predictive_model(cfs).detach().numpy().squeeze()
            temp = (y_cfs > test_pred).sum()/len(y_cfs)
            success_rate += temp
            success_rate_lst.append(temp)

            val_i += temp
            val_i_lst.append(temp)
            dev_ij = (np.abs(y_cfs - test_pred))**3
            dev_i += dev_ij.sum()/len(y_cfs)
            dev_i_lst.append(dev_ij.sum()/len(y_cfs))
            std_i += np.std(y_cfs)
            std_i_lst.append(np.std(y_cfs))     
            negative_cnt += 1
        
        proximity_loss /= args.num_inputs
        inner_diversity /= args.num_inputs
        outer_diversity /= args.num_inputs * (args.num_inputs - 1)
        success_rate /= args.num_inputs

        val_i /= args.num_inputs
        dev_i /= args.num_inputs
        std_i /= args.num_inputs

        np_proximity_lst.append(proximity_lst)
        np_inner_diversity_lst.append(inner_diversity_lst)
        np_outer_diversity_lst.append(outer_diversity_lst)
        np_success_rate_lst.append(success_rate_lst)
        # np.save(configuration_for_proj['cfs_fastdcflow_npy'] + DATA_NAME + '_proximity.npy', np.array(np_proximity_lst))
        # np.save(configuration_for_proj['cfs_fastdcflow_npy'] + DATA_NAME + '_success_rate.npy', np.array(np_success_rate_lst))
        # np.save(configuration_for_proj['cfs_fastdcflow_npy'] + DATA_NAME + '_E.npy', np.array(np_E_i_lst))
        # np.save(configuration_for_proj['cfs_fastdcflow_npy'] + DATA_NAME + '_inner_diversity.npy', np.array(np_inner_diversity_lst))
        # np.save(configuration_for_proj['cfs_fastdcflow_npy'] + DATA_NAME + '_outer_diversity.npy', np.array(np_outer_diversity_lst))
        print(f'------------------{model_name}------------------')
        # print(f'proximity: {round(proximity_loss, 5)}')
        # print(f'success_rate: {round(success_rate, 5)}')
        # print(f'E_i: {round(E_i, 5)}')
        # print(f'inner_diviersity: {round(inner_diversity, 5)}')
        # print(f'outer_diviersity: {round(outer_diversity, 5)}')
        print(f'val_i: {round(val_i, 3)}')
        print(f'val_i_std: {round(np.std(np.array(val_i_lst)), 3)}')
        print(f'dev_i: {round(dev_i, 3)}')
        print(f'dev_i_std: {round(np.std(np.array(dev_i_lst)), 3)}')
        print(f'std_i: {round(std_i, 3)}')
        print(f'std_i_std: {round(np.std(np.array(std_i_lst)), 3)}')