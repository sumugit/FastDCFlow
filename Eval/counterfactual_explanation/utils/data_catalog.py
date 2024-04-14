from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import os
import numpy as np
from collections import defaultdict
import bisect
import yaml
SEED = 42


def load_target_features_name(filename, dataset, keys):
    with open(os.path.join(filename), "r") as file_handle:
        catalog = yaml.safe_load(file_handle)

    for key in keys:
        if catalog[dataset][key] is None:
            catalog[dataset][key] = []

    return catalog[dataset]


class Data(ABC):
    @property
    @abstractmethod
    def categoricals(self):
        pass

    @property
    @abstractmethod
    def continous(self):
        pass

    @property
    @abstractmethod
    def immutables(self):
        pass

    @property
    @abstractmethod
    def target(self):
        pass

    @property
    @abstractmethod
    def raw(self):
        pass


class DataCatalog(Data):
    def __init__(self, data_name: str, data_path: str, configuration_path: str):
        self.name = data_name

        catalog_content = ["continous", "categorical", "immutable", "target"]
        self.catalog = load_target_features_name(
            configuration_path, data_name, catalog_content)

        features = []
        for key in ["continous", "categorical", "immutable"]:
            if self.catalog[key] is None:
                self.catalog[key] = []
            features.append(self.catalog[key])
        self._raw = pd.read_csv(data_path)
        print(self.catalog)

    @property
    def categoricals(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def continous(self) -> List[str]:
        return self.catalog["continous"]

    @property
    def immutables(self) -> List[str]:
        return self.catalog["immutable"]

    @property
    def target(self) -> str:
        return self.catalog["target"]

    @property
    def raw(self) -> pd.DataFrame:
        column_name = self.catalog["categorical"] + \
            self.catalog["continous"] + [self.catalog["target"]]
        return self._raw.copy()[column_name]


class EncoderNormalizeDataCatalog():
    def __init__(self, data: DataCatalog):
        self.data_raw = data.raw
        self.data_frame = data.raw
        self.continous = data.continous
        self.categoricals = data.categoricals
        self.scaler = StandardScaler()
        self.target = data.target
        self.data_catalog = data
        self.immutables = data.immutables
        self.original_features = self.data_frame.columns.tolist()
        self.encoder = OneHotEncoder(sparse=False)
        self.normalize_continuous_feature()
        self.convert_to_one_hot_encoding_form()

    def normalize_continuous_feature(self):
        self.data_frame[self.continous] = self.data_frame[self.continous].astype('float')
        self.data_frame[self.continous] = self.scaler.fit_transform(self.data_frame[self.continous])
    
    def denormalize_continuous_feature(self, df):
        df[self.continous] = self.scaler.inverse_transform(
            df[self.continous])
        return df

    def convert_to_one_hot_encoding_form(self):
        encoded_data_frame = self.encoder.fit_transform(
            self.data_frame[self.categoricals])
        column_name = self.encoder.get_feature_names_out(self.categoricals)
        self.data_frame[column_name] = encoded_data_frame
        self.data_frame = self.data_frame.drop(self.categoricals, axis=1)
        self.feature_names = self.data_frame.drop(self.target, axis=1).columns.tolist()
        self.encoded_feature_name = list(column_name)
    
    def convert_to_one_hot_encoding_form_test(self, query_instance):
        encoded_data_frame = self.encoder.transform(
            query_instance[self.categoricals])
        column_name = self.encoder.get_feature_names_out(self.categoricals)
        query_instance[column_name] = pd.DataFrame(
            encoded_data_frame, columns=column_name)
        query_instance = query_instance.drop(self.categoricals, axis=1)
        return query_instance
        
    def convert_from_one_hot_to_original_forms(self, data, prefix_sep="_"):
        out = data.copy()
        for feat in self.categoricals:
            # first, derive column names in the one-hot-encoded data from the original data
            cat_col_values = []
            for val in list(self.data_raw[feat].unique()):
                cat_col_values.append(feat + prefix_sep + str(val))
            match_cols = [c for c in data.columns if c in cat_col_values]

            # then, recreate original data by removing the suffixes
            cols, labs = [[c.replace(x, "") for c in match_cols] for x in ["", feat + prefix_sep]]
            out[feat] = pd.Categorical(
                np.array(labs)[np.argmax(data[cols].values, axis=1)])
            out.drop(cols, axis=1, inplace=True)
        return out

    def order_data(self, df, feature_order) -> pd.DataFrame:
        return df[feature_order]



class LabelEncoderNormalizeDataCatalog():
    def __init__(self, data: DataCatalog, gs = False):
        self.data_frame = data.raw
        self.continous = data.continous
        self.categoricals = data.categoricals
        self.scaler = StandardScaler()
        self.target = data.target
        self.data_catalog = data
        self.encoder = LabelEncoder()
        self.normalize_continuous_feature()
        self.convert_to_one_hot_encoding_form()
        self.encoded_feature_name = ""
        self.immutables = data.immutables

    def normalize_continuous_feature(self):
        self.data_frame[self.continous] = self.scaler.fit_transform(
            self.data_frame[self.continous])

    def denormalize_continuous_feature(self, df):
        df[self.continous] = self.scaler.inverse_transform(
            df[self.continous])
        return df

    def convert_to_one_hot_encoding_form(self):
        self.data_frame[self.categoricals] = self.data_frame[self.categoricals].apply(LabelEncoder().fit_transform)

    def convert_from_one_hot_to_original_forms(self, data, prefix_sep="_"):
        out = data.copy()
        for feat in self.categorical_feature_names:
            # first, derive column names in the one-hot-encoded data from the original data
            cat_col_values = []
            for val in list(self.data_df[feat].unique()):
                cat_col_values.append(feat + prefix_sep + str(val))
            match_cols = [c for c in data.columns if c in cat_col_values]

            # then, recreate original data by removing the suffixes
            cols, labs = [[c.replace(x, "") for c in match_cols] for x in ["", feat + prefix_sep]]
            out[feat] = pd.Categorical(
                np.array(labs)[np.argmax(data[cols].values, axis=1)])
            out.drop(cols, axis=1, inplace=True)
        return out


    def order_data(self, feature_order) -> pd.DataFrame:
        return self.data_frame[feature_order]



class TargetEncoderNormalizingDataCatalog():
    def __init__(self, data: DataCatalog):
        self.data_frame = data.raw
        self.continous = data.continous
        self.categoricals = data.categoricals
        self.feature_names = self.categoricals + self.continous
        self.scaler = StandardScaler()
        self.target = data.target
        self.data_catalog = data
        self.convert_to_target_encoding_form()
        self.normalize_feature()
        self.encoded_feature_name = ""
        self.immutables = data.immutables

    def convert_to_target_encoding_form(self):
        self.cat_dict = {}
        self.target_encoded_dict = {}
        for feature in self.categoricals:
            tmp_dict = defaultdict(lambda: 0)
            data_tmp = pd.DataFrame({feature: self.data_frame[feature], self.target: self.data_frame[self.target]})
            target_mean = data_tmp.groupby(feature)[self.target].mean()
            self.target_encoded_dict[feature] = target_mean
            for cat in target_mean.index.tolist():
                tmp_dict[cat] = target_mean[cat]
            self.cat_dict[feature] = dict(tmp_dict)           
            
            tmp = np.repeat(np.nan, self.data_frame.shape[0])
            kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
            for idx_1, idx_2 in kf.split(self.data_frame):
                target_mean = data_tmp.iloc[idx_1].groupby(feature)[self.target].mean()
                tmp[idx_2] = self.data_frame[feature].iloc[idx_2].map(target_mean)
            self.data_frame[feature] = tmp
        
        self.data_frame[self.categoricals] = self.data_frame[self.categoricals].astype('float')

    def normalize_feature(self):
        self.data_frame[self.feature_names] = self.scaler.fit_transform(self.data_frame[self.feature_names])

    def denormalize_continuous_feature(self, df):
        df[self.feature_names] = self.scaler.inverse_transform(df[self.feature_names])
        return df
    
    def convert_from_targetenc_to_original_forms(self, df):
        for cat in self.categoricals:
            d = self.cat_dict[cat]
            # ソート済みのキーと値のリストを作成
            sorted_keys = sorted(d.keys(), key=lambda k: d[k])
            sorted_values = [d[k] for k in sorted_keys]

            values = df[cat].values
            replace_values = []
            for val in values:
                # 二分探索でbに最も近い値のインデックスを見つける
                index = bisect.bisect_left(sorted_values, val)

                # 最も近い値のインデックスを範囲内に収める
                if index == len(sorted_values):
                    index -= 1
                elif index > 0 and abs(sorted_values[index] - val) > abs(sorted_values[index - 1] - val):
                    index -= 1

                # 最も絶対値の差が小さいキーを見つける
                closest_key = sorted_keys[index]
                replace_values.append(closest_key)
            df[cat] = replace_values
        return df



class TensorDatasetTraning(Dataset):

    def __init__(self, data, transform=None):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index, :-1]
        label = self.data[index, -1]
        return image, label


class TensorDatasetTraningCE(Dataset):

    def __init__(self, data, transform=None):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index, :-2]
        label = self.data[index, -2:]
        return image, label
