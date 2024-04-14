import sys
import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm
sys.path.append('/workspace/Eval/')
from config.config import Config
import config.setup as setup
from counterfactual_explanation.utils.helpers import load_configuration_from_yaml
from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name, load_configuration_from_yaml)
from counterfactual_explanation.utils.mlcatalog import (
    find_latent_mean_two_classes, model_prediction, negative_prediction_index,
    positive_prediction_index, prediction_instances, make_perturbation)
from counterfactual_explanation.utils.data_catalog import (
    DataCatalog, EncoderNormalizeDataCatalog, LabelEncoderNormalizeDataCatalog, TargetEncoderNormalizingDataCatalog,
    TensorDatasetTraning)
sys.path.append('/workspace/GeCo/')
import warnings
warnings.filterwarnings("ignore")
from GeCo.geco import Genetic 


if __name__ == '__main__':
    args = setup.setup(Config)
    DATA_NAME = args.data_name
    CONFIG_PATH = '/workspace/Eval/configuration/data_catalog.yaml'
    CONFIG_FOR_PROJECT = '/workspace/Eval/configuration/project_configurations.yaml'
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_train_input']

    predictive_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
        DATA_NAME, encoding='onehotenc')
    predictive_model = predictive_model.cpu()
    
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    feature_names = encoder_normalize_data_catalog.feature_names
    
    """ test input """
    TOTAL_CFS = args.total_cfs
    POP_SIZE = args.pop_size
    MAXITERATIONS = args.maxiterations
    inputs = pd.read_csv(configuration_for_proj[DATA_NAME + '_onehotenc_test_input'])
    inputs = inputs.drop(target, axis=1)
    
    start = time.time()
    negative_cnt = 0
    for query_instance in tqdm(inputs.values):
        if negative_cnt >= args.num_inputs:
            break
        query_instance = torch.Tensor(query_instance)
        test_preds = model_prediction(predictive_model, query_instance)
        if test_preds.item() >= 0.5:
            continue
        
        negative_cnt += 1
        # mode
        exp_genetic = Genetic(encoder_normalize_data_catalog, predictive_model)
        x_cf = exp_genetic.generate_counterfactuals(query_instance, total_CFs=TOTAL_CFS, population_size=POP_SIZE, maxiterations=MAXITERATIONS)
        y_cf = model_prediction(predictive_model, x_cf)
        x_cf = torch.hstack((x_cf, y_cf))
        x_cf_df = pd.DataFrame(x_cf.detach().numpy(), columns=feature_names + [target])
        x_cf_df.drop(target, axis=1).to_csv(configuration_for_proj["cfs_raw_geco_onehotenc_" + DATA_NAME] + f"neg_{negative_cnt}.csv", index=False)
        x_cf_df = encoder_normalize_data_catalog.denormalize_continuous_feature(x_cf_df)
        x_cf_df = encoder_normalize_data_catalog.convert_from_one_hot_to_original_forms(x_cf_df)
        x_cf_df.to_csv(configuration_for_proj["cfs_geco_onehotenc_" + DATA_NAME] + f"neg_{negative_cnt}.csv", index=False)

    end = time.time()
    elapsed_time = end - start
    print(f'{args.total_cfs} cfs generation time: {elapsed_time:.5f}')