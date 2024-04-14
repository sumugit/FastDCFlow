import numpy as np
import torch

from counterfactual_explanation.flow_ssl.flow_loss import FlowLoss, FlowCrossEntropyCELoss
# from counterfactual_explanation.flow_ssl.data import make_moons_ssl
from counterfactual_explanation.flow_ssl.distributions import SSLGaussMixture
from counterfactual_explanation.flow_ssl.realnvp.realnvp import RealNVPTabular
from counterfactual_explanation.utils.data_catalog import (DataCatalog, LabelEncoderNormalizeDataCatalog,
                                                           TensorDatasetTraning, TensorDatasetTraningCE)
from counterfactual_explanation.utils.helpers import load_configuration_from_yaml
from counterfactual_explanation.utils.mlcatalog import (save_pytorch_model_to_model_path,
                                                        train_one_epoch_batch_data)
from torch.utils.data import DataLoader
from counterfactual_explanation.utils.helpers import \
    load_all_configuration_with_data_name
from counterfactual_explanation.utils.mlcatalog import (
    model_prediction, negative_prediction_index, positive_prediction_index, prediction_instances)

from counterfactual_explanation.utils.mlcatalog import (
    get_latent_representation_from_flow,
    original_space_value_from_latent_representation)

from tqdm import tqdm

if __name__ == "__main__":
    # loading data
    DATA_NAME = 'adult'
    CONFIG_PATH = '/workspace/NormalizingFlow/configuration/data_catalog.yaml'
    CONFIG_FOR_PROJECT = '/workspace/NormalizingFlow/configuration/project_configurations.yaml'
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_dataset']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)
    encoder_normalize_data_catalog = LabelEncoderNormalizeDataCatalog(data_catalog)
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    feature_names = encoder_normalize_data_catalog.categoricals + \
        encoder_normalize_data_catalog.continous
    predictive_model, _, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
        DATA_NAME)

    LR_INIT = 1e-6
    EPOCHS = 10
    BATCH_SIZE = 32
    PRINT_FREQ = 10
    MEAN_VALUE = 0.5

    if DATA_NAME == 'simple_bn':
        x1_mean = data_frame['x1'].median()
        x2_mean = data_frame['x2'].median()
        x3_mean = data_frame['x3'].median()
        means = torch.tensor([
            np.array([x1_mean, x2_mean, x3_mean]).astype(np.float32)
        ])
    elif DATA_NAME == 'adult':
        x1_mean = 0.05
        x2_mean = 0.05
        x3_mean = 0.05
        x4_mean = data_frame['age'].mean()
        x5_mean = data_frame['hours_per_week'].mean()
        means = torch.tensor([
            np.array([x1_mean, x2_mean, x3_mean, x4_mean, x5_mean]).astype(np.float32)
        ])

    prior = SSLGaussMixture(means=means, device='cuda')
    features = data_frame[['education', 'marital_status', 'occupation', 'age', 'hours_per_week']].values.astype(
        np.float32)
    features = data_frame[feature_names].values.astype(np.float32)
    features = torch.Tensor(features)
    features_cuda = features.cuda()
    labels = model_prediction(predictive_model, features_cuda).detach().cpu()

    # set up flow model and loss function
    flow = RealNVPTabular(num_coupling_layers=3, in_dim=5,
                          num_layers=5, hidden_dim=8).cuda()
    loss_fn = FlowLoss(prior) # flow loss
    loss_cefn = FlowCrossEntropyCELoss(margin=0.5)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=LR_INIT, weight_decay=1e-2)

    negative_index = negative_prediction_index(labels) # pred < 0.5 idx
    negative_instance_features = prediction_instances(features, negative_index) # get negative instances
    negative_labels = prediction_instances(labels, negative_index)
    negative_data = torch.hstack(
        (negative_instance_features, negative_labels))
    negative_data = TensorDatasetTraning(negative_data)
    negative_loader = DataLoader(negative_data, batch_size=64, shuffle=True)

    positive_index = positive_prediction_index(labels)
    positive_instance_features = prediction_instances(features, positive_index)
    positive_labels = prediction_instances(labels, positive_index)
    positive_data = torch.hstack(
        (positive_instance_features, positive_labels))
    positive_data = TensorDatasetTraning(positive_data)
    positive_loader = DataLoader(positive_data, batch_size=64, shuffle=True)

    # delta_value = nn.Parameter(torch.zeros(z_value.shape[1]).cuda())
    # for positive instances
    # for t in tqdm(range(EPOCHS)):
    #     for local_batch, local_labels in (positive_loader):
    #         local_batch = local_batch.cuda()
    #         local_labels = local_labels.cuda()
    #         z = flow(local_batch)              
    #         x = flow.inverse(z)
    #         local_prediction = model_prediction(predictive_model, x)
    #         sldj = flow.logdet()
    #         flow_loss = loss_fn(z, sldj)
    #         ce_loss = loss_cefn.forward(local_prediction, positive=True)
    #         total_loss = flow_loss + ce_loss
    #         optimizer.zero_grad()
    #         total_loss.backward()
    #         optimizer.step()
    #     if t % PRINT_FREQ == 0:
    #         print('iter %s:' % t, 'loss = %.3f' % total_loss)

    # for negative instances
    for t in tqdm(range(EPOCHS)):
        for local_batch, local_labels in (negative_loader):
            local_batch = local_batch.cuda()
            local_labels = local_labels.cuda()
            z = flow(local_batch)
            # 摂動を加える
            perturbation = torch.randn_like(z)
            perturbed_z_value = z + perturbation
            x = flow.inverse(perturbed_z_value)
            local_prediction = model_prediction(predictive_model, x)
            sldj = flow.logdet()
            flow_loss = loss_fn(z, sldj)
            ce_loss = loss_cefn.forward(local_prediction)
            total_loss = flow_loss + ce_loss * 50
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        if t % PRINT_FREQ == 0:
            print('iter %s:' % t, 'loss = %.3f' % total_loss)


    # test input
    for local_batch, local_labels in (negative_loader):
        local_batch = local_batch.cuda()
        z = flow(local_batch) # forward
        # generate samples
        # 少しノイズを加える
        perturbation = torch.randn_like(z)
        perturbed_z_value = z + perturbation
        x = flow.inverse(perturbed_z_value) # inverse
        labels = model_prediction(
            predictive_model, x).detach().cpu()

        x_origin = torch.hstack((local_batch.detach().cpu(), local_labels.unsqueeze(1)))
        x_cf = torch.hstack((x.detach().cpu(), labels))
        print(x_origin[:5])
        print(x_cf[:5])
        print((labels.squeeze(1) -local_labels > 0).sum() / len(labels))
        break


    # save_pytorch_model_to_model_path(flow, configuration_for_proj['flow_ce_' + DATA_NAME])
