import torch
from .data_catalog import DataCatalog

def load_pytorch_prediction_model_from_model_path(model_path):
    # print("Load model")
    model = torch.load(model_path)
    model.eval()
    # print("End load")
    return model


def load_pytorch_prediction_model_with_state_dict(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_pytorch_model_to_model_path(model, model_path):
    torch.save(model, model_path)


def get_latent_representation_from_flow(flow, input_value):
    """潜在空間へのマッピングと同じ
    """
    return flow(input_value)


def original_space_value_from_latent_representation(flow, z_value):
    """逆変換と同じ
    """
    return flow.inverse(z_value)

def get_latent_representation_from_flow_mixed_type(flow, deq, input_value, index_):
    """離散値と連続値を分けて潜在空間へマッピング"""
    discrete_value = input_value[:,:index_]
    continuous_transformation, _ = deq.forward(discrete_value, ldj=None, reverse=False) # 離散値 → 連続値
    continuous_value = input_value[:, index_:]
    continuous_representation = torch.hstack([continuous_transformation, continuous_value])
    
    return flow(continuous_representation)

def original_space_value_from_latent_representation_mixed_type(flow, deq, z_value, index_):
    """離散値と連続値を分けて逆変換"""
    continuous_value = flow.inverse(z_value)
    discrete_value = continuous_value[:,:index_]
    continuous_value = continuous_value[:, index_:]
    discrete_value, _ = deq.forward(discrete_value, ldj=None, reverse=True) # 連続値 → 離散値
    input_value = torch.hstack([discrete_value, continuous_value])

    return input_value

def model_prediction(predictive_model, features):
    return predictive_model(features)

def negative_prediction_index(prediction, num):
    """get idx which is lower than 0.5
    """
    return torch.lt(prediction, num).reshape(-1)

def positive_prediction_index(prediction, num):
    """get idx which is greater than 0.5
    """
    return torch.gt(prediction, num).reshape(-1)

def prediction_instances(instances, indexes):
    """get dataset[idx]
    """
    return instances[indexes]


def find_latent_mean_two_classes(flow, x0, x1):
    z0 = flow(x0) 
    z1 = flow(x1)
    mean_z0 = torch.mean(z0)
    mean_z1 = torch.mean(z1)
    return mean_z0, mean_z1

class MLModelCatalog():
    def __init__(self,data: DataCatalog, predictive_model) -> None:
        self.model = predictive_model 
        self._continuous = data.continous
        self._categoricals = data.categoricals

    def predict(self):
        pass
    def predict_proba(self):
        pass


def train_one_epoch_batch_data(flow_model, optimizer, loss_fn, batch_x, batch_y):
    z_value = flow_model(batch_x)
    sldj = flow_model.logdet()
    loss = loss_fn(z_value, sldj, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def make_perturbation(z_value, delta_value):
    return z_value + delta_value


def generate_orthogonal_noise(z, scaling_factor=1.0):
    batch_size, dim = z.shape
    random_matrix = torch.randn(batch_size, batch_size)
    q, _ = torch.linalg.qr(random_matrix)
    scaled_noise = q[:, :dim] * scaling_factor
    return scaled_noise
