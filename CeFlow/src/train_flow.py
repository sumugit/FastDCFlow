import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/workspace/Eval/')
from config.config import Config
import config.setup as setup
from counterfactual_explanation.flow_ssl import FlowLoss
from counterfactual_explanation.flow_ssl.distributions import SSLGaussMixture
from counterfactual_explanation.flow_ssl.realnvp.coupling_layer import (
    DequantizationOriginal)
from counterfactual_explanation.flow_ssl.realnvp.realnvp import RealNVPTabular
from counterfactual_explanation.utils.data_catalog import (
    DataCatalog, EncoderNormalizeDataCatalog, LabelEncoderNormalizeDataCatalog,
    TensorDatasetTraning)
from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name, load_configuration_from_yaml)
from counterfactual_explanation.utils.mlcatalog import (
    find_latent_mean_two_classes, model_prediction, negative_prediction_index,
    positive_prediction_index, prediction_instances, make_perturbation,
    get_latent_representation_from_flow_mixed_type,
    original_space_value_from_latent_representation_mixed_type,
    get_latent_representation_from_flow, original_space_value_from_latent_representation,
    save_pytorch_model_to_model_path)

def trans_to_device(variable):
    if torch.cuda.is_available() and args.device == 'cuda':
        return variable.cuda()
    else:
        return variable.cpu()

if __name__ == "__main__":
    """Parsing argument"""
    args = setup.setup(Config)
    DATA_NAME = args.data_name
    CONFIG_PATH = '/workspace/Eval/configuration/data_catalog.yaml'
    CONFIG_FOR_PROJECT = '/workspace/Eval/configuration/project_configurations.yaml'
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_train_input']
    predictive_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
        DATA_NAME, encoding='targetenc')
    predictive_model = trans_to_device(predictive_model)
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    feature_names = encoder_normalize_data_catalog.feature_names

    LR_INIT = args.lr_init
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    PRINT_FREQ = args.print_freq

    means = [data_frame[feature].mean() for feature in feature_names]
    means = torch.tensor([np.array(means).astype(np.float32)])
    stds = [data_frame[feature].std() for feature in feature_names]
    stds = torch.tensor([np.array(stds).astype(np.float32)])

    features = data_frame[feature_names].values.astype(np.float32)
    features = torch.Tensor(features)
    features_dev = trans_to_device(features)
    labels = model_prediction(predictive_model, features_dev).detach().cpu()

    train_data = torch.hstack((features, labels))
    train_data = TensorDatasetTraning(train_data)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    
    """Flow Model"""
    # set up flow model and loss function
    flow_model = RealNVPTabular(num_coupling_layers=3, in_dim=features.shape[1], num_layers=5, hidden_dim=12)
    prior = SSLGaussMixture(means=means, inv_cov_stds=stds, device=args.device)
    loss_fn = FlowLoss(prior) # flow loss
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=LR_INIT, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.01)
    
    best_model = None
    best_loss = 100000

    start = time.time()
    for epoch in tqdm(range(1, EPOCHS+1)):
        for local_batch, local_labels in (train_loader):
            local_batch = trans_to_device(local_batch)
            local_labels = trans_to_device(local_labels)
            z_value = get_latent_representation_from_flow(flow_model, local_batch)
            sldj = flow_model.logdet() # get log det Jacobian
            flow_loss = loss_fn(z_value, sldj, local_labels) # get nll loss
            optimizer.zero_grad()
            flow_loss.backward()
            optimizer.step()
        
        
        if epoch % PRINT_FREQ == 0:
            scheduler.step()
            cur_lr = scheduler.optimizer.param_groups[0]['lr']
            if flow_loss < best_loss:
                best_loss = flow_loss
                best_model = flow_model
            print('iter %s:' % epoch, 'loss = %.3f' % flow_loss, 'learning rate: %s' % cur_lr)


end = time.time()
elapsed_time = end - start
print ("pretrain time:{0}".format(elapsed_time) + "[sec]")
# save_pytorch_model_to_model_path(best_model, configuration_for_proj['ceflow_model_targetenc_' + DATA_NAME])
