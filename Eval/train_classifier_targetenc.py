import numpy as np
import torch
from torch.utils.data import DataLoader
import json

from counterfactual_explanation.models.classifier import train_predictive_model, Net
from counterfactual_explanation.utils.data_catalog import (DataCatalog, EncoderNormalizeDataCatalog,
                                                           LabelEncoderNormalizeDataCatalog,
                                                           TargetEncoderNormalizingDataCatalog,
                                                           TensorDatasetTraning)
from counterfactual_explanation.utils.helpers import load_configuration_from_yaml
from counterfactual_explanation.utils.mlcatalog import save_pytorch_model_to_model_path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from config.config import Config
import config.setup as setup

if __name__ == '__main__':
    args = setup.setup(Config)
    DATA_NAME = args.data_name
    CONFIG_PATH = "/workspace/Eval/configuration/data_catalog.yaml"
    CONFIG_FOR_PROJECT = "/workspace/Eval/configuration/project_configurations.yaml"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_train_input']
    MODEL_PATH = configuration_for_proj['trained_models_targetenc_' + DATA_NAME]

    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)
    encoder_normalize_data_catalog = TargetEncoderNormalizingDataCatalog(data_catalog)
    JSON_PATH = configuration_for_proj[DATA_NAME + '_dataset_targetenc']
    json_file = open(JSON_PATH, mode="w")
    json.dump(encoder_normalize_data_catalog.cat_dict, json_file, indent=2, ensure_ascii=False)
    
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    feature_names = encoder_normalize_data_catalog.categoricals + \
        encoder_normalize_data_catalog.continous
    
    train_labels = data_frame[target].values.astype(np.float32)
    train_features = data_frame.drop(
        columns=[target], axis=1).values.astype(np.float32)

    train_features = torch.Tensor(train_features)
    train_labels = torch.Tensor(train_labels).reshape(-1, 1)
    train_data = torch.hstack((train_features, train_labels))
    train_data = TensorDatasetTraning(train_data)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = Net(train_features.shape[1])
    model.to(DEVICE)
    model = train_predictive_model(train_loader, model, flag=False)
    save_pytorch_model_to_model_path(model, MODEL_PATH)
    
    
    query_instances = pd.read_csv(configuration_for_proj[DATA_NAME + '_raw_test_input'])
    # TargetEncoding
    for feature in encoder_normalize_data_catalog.categoricals:
        query_instances[feature] = query_instances[feature].map(encoder_normalize_data_catalog.cat_dict[feature])
    # Normalization
    query_features = query_instances.drop(columns=[target], axis=1)
    query_labels = query_instances[target].values.astype(np.float32)
    query_features = encoder_normalize_data_catalog.scaler.transform(query_features[feature_names])
    query_instances = np.hstack((query_features, query_labels.reshape(-1, 1)))
    query_instances = torch.Tensor(query_instances).cuda()
    query_instances = TensorDatasetTraning(query_instances)
    test_loader = DataLoader(query_instances, batch_size=64, shuffle=True)
    
    model.eval()
    acc_results = []
    auc_results = []
    for local_batch, local_labels in test_loader:
        with torch.no_grad():
            inputs = local_batch.to(DEVICE)
            outputs = model(inputs)
            acc_batch = accuracy_score(local_labels.cpu().numpy(), np.round(outputs.cpu().numpy()))
            auc_batch = roc_auc_score(local_labels.cpu().numpy(), outputs.cpu().numpy())
            acc_results.append(acc_batch)
            auc_results.append(auc_batch)

    print(f'Accuracy over test data: {round(np.mean(acc_results), 5)}')
    print(f'AUC over test data: {round(np.mean(auc_results), 5)}')