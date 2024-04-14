import numpy as np
import torch
from counterfactual_explanation.utils.helpers import load_configuration_from_yaml
from sklearn.model_selection import train_test_split
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
    data_catalog = load_configuration_from_yaml(CONFIG_PATH)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_dataset']
    MODEL_PATH = configuration_for_proj['trained_models_targetenc_' + DATA_NAME]
    target = data_catalog[DATA_NAME]['target']

    dataset = pd.read_csv(DATA_PATH)
    features = dataset.drop(columns=[target], axis=1)
    # drop nan
    for feature in dataset.columns:
        dataset[feature] = dataset[feature].replace(' ', np.nan)
    dataset = dataset.dropna()
    features = dataset.drop(columns=[target], axis=1)
    feature_columns = features.columns.tolist()
    features = features.values
    # change labels from Yes, No to 0, 1
    dataset[target] = dataset[target].map({'Yes': 1, 'No': 0})
    labels = dataset[target].values.astype(np.float32)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.1, random_state=configuration_for_proj['seed'])
    train_df = pd.DataFrame(np.hstack((train_features, train_labels.reshape(-1, 1))), columns=feature_columns + [target])
    test_df = pd.DataFrame(np.hstack((test_features, test_labels.reshape(-1, 1))), columns=feature_columns + [target])
    train_df.to_csv(configuration_for_proj[DATA_NAME + '_train_input'], index=False)
    test_df.to_csv(configuration_for_proj[DATA_NAME + '_raw_test_input'], index=False)