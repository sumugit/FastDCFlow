import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pickle
import json

from counterfactual_explanation.models.classifier import train_predictive_model, Net
from counterfactual_explanation.utils.data_catalog import (DataCatalog, EncoderNormalizeDataCatalog,
                                                           LabelEncoderNormalizeDataCatalog,
                                                           TargetEncoderNormalizingDataCatalog,
                                                           TensorDatasetTraning)
from counterfactual_explanation.utils.helpers import load_configuration_from_yaml
from sklearn.metrics import accuracy_score, roc_auc_score
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

    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)
    encoder_normalize_data_catalog = TargetEncoderNormalizingDataCatalog(data_catalog)
    JSON_PATH = configuration_for_proj[DATA_NAME + '_dataset_targetenc']
    json_file = open(JSON_PATH, mode="w")
    json.dump(encoder_normalize_data_catalog.cat_dict, json_file, indent=2, ensure_ascii=False)
    
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target

    labels = data_frame[target].values.astype(np.float32)
    features = data_frame.drop(columns=[target], axis=1).values.astype(np.float32)

    model_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_index, val_index) in enumerate(kf.split(features, labels)):
            print(f"Running fold {i+1}...")
            train_features, train_labels = features[train_index], labels[train_index]
            val_features, val_labels = features[val_index], labels[val_index]

            train_features = torch.Tensor(train_features)
            train_labels = torch.Tensor(train_labels).reshape(-1, 1)
            val_features = torch.Tensor(val_features)
            val_labels = torch.Tensor(val_labels).reshape(-1, 1)

            train_data = torch.hstack((train_features, train_labels))
            val_data = torch.hstack((val_features, val_labels))

            train_data = TensorDatasetTraning(train_data)
            val_data = TensorDatasetTraning(val_data)

            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

            model = Net(features.shape[1])
            model.to(DEVICE)
            model.train()
            model = train_predictive_model(train_loader, model, flag=False)

            model.eval()
            acc_results = []
            auc_results = []
            for local_batch, local_labels in val_loader:
                with torch.no_grad():
                    inputs = local_batch.to(DEVICE)
                    outputs = model(inputs)
                    acc_batch = accuracy_score(local_labels.cpu().numpy(), np.round(outputs.cpu().numpy()))
                    auc_batch = roc_auc_score(local_labels.cpu().numpy(), outputs.cpu().numpy())
                    acc_results.append(acc_batch)
                    auc_results.append(auc_batch)

            print(f'Accuracy over all validation data: {np.mean(acc_results)}')
            print(f'AUC over all validation data: {np.mean(auc_results)}')
            model_scores.append((np.mean(acc_results), np.mean(auc_results)))

    accuracy = [score[0] for score in model_scores]
    auc = [score[1] for score in model_scores]
    avg_accuracy = np.mean(accuracy)
    avg_auc = np.mean(auc)
    print(f'Average validation Accuracy over all folds: {round(avg_accuracy, 5)}')
    print(f'Average validation AUC over all folds: {round(avg_auc, 5)}')
        
    with open(configuration_for_proj['accuracy_per_fold_targetenc_' + DATA_NAME], 'w') as f:
        json.dump(accuracy, f)
    with open(configuration_for_proj['auc_per_fold_targetenc_' + DATA_NAME], 'w') as f:
        json.dump(auc, f)
