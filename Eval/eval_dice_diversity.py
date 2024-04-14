import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append("/workspace/Eval/")
from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name,
    load_configuration_from_yaml,
)
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
CONFIG_PATH = "/workspace/Eval/configuration/data_catalog.yaml"
CONFIG_FOR_PROJECT = "/workspace/Eval/configuration/project_configurations.yaml"
TEST_INPUT = "/workspace/Eval/configuration/test_input.yaml"
data_config = load_configuration_from_yaml(CONFIG_PATH)
configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)


data_names = ["adult", "bank", "churn"]
model = "fastdcflow"
criterion = torch.nn.MSELoss()
ts = range(1, 11)
data_dict = {}

for DATA_NAME in data_names:
    target_encoded_feature_names = data_config[DATA_NAME]["categorical"]
    (
        predictive_model,
        encoder_normalize_data_catalog,
        configuration_for_proj,
    ) = load_all_configuration_with_data_name(DATA_NAME, encoding="targetenc")
    predictive_model.eval()
    predictive_model = predictive_model.cpu()
    det_t_lst = []

    for t in ts:
        det_lst = []
        CFS_DATA_PATH = (
            configuration_for_proj["cfs_t_" + model + "_targetenc_" + DATA_NAME]
            + f"t_{t}_"
        )
        csv_files = glob.glob(f"{CFS_DATA_PATH}*.csv")
        query_instances = pd.read_csv(
            configuration_for_proj[DATA_NAME + "_raw_test_input"]
        )
        query_instances = query_instances.drop(
            columns=encoder_normalize_data_catalog.target
        )
        # preprocessing
        category_dict = json.load(
            open(configuration_for_proj[DATA_NAME + "_dataset_targetenc"], "r")
        )
        for feature in target_encoded_feature_names:
            if feature == "SeniorCitizen":
                query_instances[feature] = query_instances[feature].astype("str")
            query_instances[feature] = query_instances[feature].map(
                category_dict[feature]
            )
        query_instances = encoder_normalize_data_catalog.scaler.transform(
            query_instances[encoder_normalize_data_catalog.feature_names]
        )
        negative_cnt = 0
        for idx, query in tqdm(
            enumerate(query_instances), total=len(query_instances)
        ):
            if negative_cnt >= args.num_inputs:
                break
            query = torch.tensor(query, dtype=torch.float32)
            test_pred = predictive_model(query).detach().numpy()[0]
            if test_pred >= args.pred_thrsh:
                continue
            # load cfs
            cfs = pd.read_csv(csv_files[negative_cnt])[: args.total_cfs]
            cfs = encoder_normalize_data_catalog.scaler.inverse_transform(
                cfs[encoder_normalize_data_catalog.feature_names]
            )
            cfs = np.array(cfs, dtype=np.float32)
            cfs = torch.tensor(cfs, dtype=torch.float32)
            det_entries = torch.ones((args.total_cfs, args.total_cfs))
            for i in range(args.total_cfs):
                for j in range(args.total_cfs):
                    det_entries[(i, j)] = (1.0 / (1.0 + criterion(cfs[i], cfs[j])))
                    if i == j:
                        det_entries[(i, j)] += 0.0001
            det_score = torch.logdet(det_entries)
            det_score = float(det_score.detach().numpy())
            # print(det_score)
            det_lst.append(det_score)

            negative_cnt += 1

        print(np.mean(det_lst))
        det_t_lst.append(det_lst)

    data_dict[DATA_NAME] = {
        "det_score": det_t_lst,
    }

json.dump(
    data_dict,
    open(
        configuration_for_proj["cfs_fastdcflow_npy"]
        + "dice_diversity_temperature_dict.json",
        "w",
    ),
)
