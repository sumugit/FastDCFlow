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
    MODEL_PATH = configuration_for_proj['trained_models_onehotenc_' + DATA_NAME]

    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)
    encoder_normalize_data_catalog = EncoderNormalizeDataCatalog(data_catalog)
        
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target

    labels = data_frame[target].values.astype(np.float32)
    features = data_frame.drop(
        columns=[target], axis=1).values.astype(np.float32)

    # split dataset
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.1, random_state=configuration_for_proj['seed'])
    test_dataset = np.hstack((test_features, test_labels.reshape(-1, 1)))
    test_df = pd.DataFrame(test_dataset, columns=data_frame.drop(columns=[target], axis=1).columns.tolist() + [target])
    test_df.to_csv(configuration_for_proj[DATA_NAME + '_onehotenc_test_input'], index=False)
    
    # test_raw_df = encoder_normalize_data_catalog.denormalize_continuous_feature(test_df)
    # test_raw_df = encoder_normalize_data_catalog.convert_from_targetenc_to_original_forms(test_raw_df)
    # test_raw_df.to_csv(configuration_for_proj[DATA_NAME + '_raw_test_input'], index=False)

    train_features = torch.Tensor(train_features)
    train_labels = torch.Tensor(train_labels).reshape(-1, 1)
    train_data = torch.hstack((train_features, train_labels))
    train_data = TensorDatasetTraning(train_data)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = Net(features.shape[1])
    model.to(DEVICE)
    model = train_predictive_model(train_loader, model, flag=False)
    save_pytorch_model_to_model_path(model, MODEL_PATH)
    
    test_features = torch.Tensor(test_features)
    test_labels = torch.Tensor(test_labels).reshape(-1, 1)
    test_data = torch.hstack((test_features, test_labels))
    test_data = TensorDatasetTraning(test_data)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    
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
