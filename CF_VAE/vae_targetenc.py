import sys
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import time
import json
sys.path.append('/workspace/Eval/')
from config.config import Config
import config.setup as setup
from counterfactual_explanation.utils.mlcatalog import (save_pytorch_model_to_model_path,
                                                        train_one_epoch_batch_data)
from counterfactual_explanation.utils.helpers import load_configuration_from_yaml
from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name, load_configuration_from_yaml)
from counterfactual_explanation.utils.mlcatalog import (
    find_latent_mean_two_classes, model_prediction, negative_prediction_index,
    positive_prediction_index, prediction_instances, make_perturbation)
from counterfactual_explanation.utils.data_catalog import (
    DataCatalog, EncoderNormalizeDataCatalog, LabelEncoderNormalizeDataCatalog, TargetEncoderNormalizingDataCatalog,
    TensorDatasetTraning)
sys.path.append('/workspace/CF_VAE/')
from models.cf_vae import CF_VAE
from models.decoder.dec_nn import NNDecoder
from models.encoder.enc_nn import GaussianNNEncoder

def trans_to_device(variable):
    if torch.cuda.is_available() and args.device == 'cuda':
        return variable.cuda()
    else:
        return variable.cpu()

if __name__ == '__main__':
    args = setup.setup(Config)
    DATA_NAME = args.data_name
    CONFIG_PATH = '/workspace/Eval/configuration/data_catalog.yaml'
    CONFIG_FOR_PROJECT = '/workspace/Eval/configuration/project_configurations.yaml'
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_train_input']

    predictive_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
        DATA_NAME, encoding='targetenc')
    predictive_model = trans_to_device(predictive_model)
    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)
    
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    feature_names = encoder_normalize_data_catalog.feature_names
    

    LR_INIT = args.lr_init
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    PRINT_FREQ = args.print_freq
    DEVICE = args.device

    means = [data_frame[feature].mean() for feature in feature_names]
    means = torch.tensor([np.array(means).astype(np.float32)])

    features = data_frame[feature_names].values.astype(np.float32)
    features = torch.Tensor(features)
    features_dev = trans_to_device(features)
    labels = model_prediction(predictive_model, features_dev).detach().cpu()

    negative_index = negative_prediction_index(labels, 1.0) # pred < 1.0 idx
    negative_instance_features = prediction_instances(features, negative_index) # get negative instances
    negative_labels = prediction_instances(labels, negative_index)
    negative_data = torch.hstack((negative_instance_features, negative_labels))
    negative_data = TensorDatasetTraning(negative_data)
    negative_loader = DataLoader(negative_data, batch_size=BATCH_SIZE, shuffle=True)

    """CF-VAE Model"""
    PRETRAINED = args.pretrain
    if PRETRAINED:
        cf_vae = torch.load(configuration_for_proj['cf_vae_model_targetenc_' + DATA_NAME])
        cf_vae = trans_to_device(cf_vae)
        elapsed_time1 = 0
    else:
        encoder = GaussianNNEncoder(encoder_normalize_data_catalog, args)
        decoder = NNDecoder(encoder_normalize_data_catalog, args)
        cf_vae = CF_VAE(encoder, decoder)
        optimizer = optim.Adam([
            {'params': filter(lambda p: p.requires_grad, encoder.fc_mean.parameters()), 'weight_decay': args.wm1},
            {'params': filter(lambda p: p.requires_grad, encoder.fc_logvar.parameters()), 'weight_decay': args.wm1},
            ], lr=LR_INIT
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.01)
        best_model = None
        best_loss = 1000000

        # histroy = defaultdict(lambda: [])
        start1 = time.time()
        for epoch in tqdm(range(1, EPOCHS+1)):
            temp_kl_loss = 0
            temp_y_loss = 0
            temp_prox_loss = 0
            temp_total_loss = 0
            for local_batch, local_labels in (negative_loader):
                local_batch = trans_to_device(local_batch)
                local_labels = trans_to_device(local_labels)
                expected_outcome = trans_to_device(torch.ones(local_batch.shape[0]))
                cf_vae.train()
                optimizer.zero_grad()
                # compute_loss の中身修正
                total_loss, recon_err, kl_divergence, validity_loss = cf_vae.compute_loss(local_batch, expected_outcome, predictive_model, args.temperature)
                temp_kl_loss += kl_divergence.item()
                temp_y_loss += validity_loss.item()
                temp_prox_loss += recon_err.item()
                temp_total_loss += total_loss.item()
                total_loss.backward()
                optimizer.step()
            # print
            # histroy['kl_loss'].append(temp_kl_loss / len(negative_loader))
            # histroy['y_loss'].append(temp_y_loss / len(negative_loader))
            # histroy['prox_loss'].append(temp_prox_loss / len(negative_loader))
            # histroy['total_loss'].append(temp_total_loss / len(negative_loader))
            if epoch % PRINT_FREQ == 0:
                scheduler.step()
                cur_lr = scheduler.optimizer.param_groups[0]['lr']
                # print("\n Epoch {}, Loss {:.4f}, Learning rate {:.4f}".format(epoch, total_loss, cur_lr))
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_model = cf_vae
        
        end1 = time.time()
        elapsed_time1 = end1 - start1
        print(f'pretrain time: {elapsed_time1:.5f}s')
        cf_vae = best_model
        # save_pytorch_model_to_model_path(cf_vae, configuration_for_proj['cf_vae_model_targetenc_' + DATA_NAME])
        # with open(configuration_for_proj['cf_vae_model_targetenc_' + DATA_NAME + '_history'], 'w') as f:
        #     json.dump(histroy, f, indent=2)

    
    """ test input """
    TOTAL_CFS = args.total_cfs
    query_instances = pd.read_csv(configuration_for_proj[DATA_NAME + '_raw_test_input'])
    # TargetEncoding
    for feature in encoder_normalize_data_catalog.categoricals:
        query_instances[feature] = query_instances[feature].map(encoder_normalize_data_catalog.cat_dict[feature])
    # Normalization
    query_features = query_instances.drop(columns=[target], axis=1)
    query_labels = query_instances[target].values.astype(np.float32)
    query_features = encoder_normalize_data_catalog.scaler.transform(query_features[feature_names])

    start2 = time.time()
    negative_cnt = 0
    times = []
    for query_instance in tqdm(query_features):
        if negative_cnt >= args.num_inputs:
            break
        query_instance = trans_to_device(torch.Tensor(query_instance))
        test_preds = model_prediction(predictive_model, query_instance)
        if test_preds.item() >= 0.5:
            continue
        negative_cnt += 1
        start3 = time.time()
        # generate counterfactuals
        query_instance = query_instance.repeat(TOTAL_CFS, 1)
        expected_outcome = trans_to_device(torch.ones(TOTAL_CFS))
        total_loss, x_cf, y_cf = cf_vae.compute_elbo(query_instance, expected_outcome, predictive_model, args.temperature)
        x_cf = torch.hstack((x_cf, y_cf))
        x_cf_df = pd.DataFrame(x_cf.detach().cpu().numpy(), columns=feature_names + [target])
        x_cf_df["eval_loss"] = total_loss.detach().cpu().numpy()
        x_cf_df = x_cf_df.sort_values(by="eval_loss", ascending=True)
        x_cf_df = x_cf_df.drop("eval_loss", axis=1)
        x_cf_df.drop(target, axis=1).to_csv(configuration_for_proj["cfs_raw_cf_vae_targetenc_" + DATA_NAME] + f"neg_{negative_cnt}.csv", index=False)
        x_cf_df = encoder_normalize_data_catalog.denormalize_continuous_feature(x_cf_df)
        x_cf_df = encoder_normalize_data_catalog.convert_from_targetenc_to_original_forms(x_cf_df)
        x_cf_df.to_csv(configuration_for_proj["cfs_cf_vae_targetenc_" + DATA_NAME] + f"neg_{negative_cnt}.csv", index=False)
        end3 = time.time()
        times.append(end3 - start3)
        # print(f'negative_cnt: {negative_cnt}, time: {end3 - start3:.5f}')
    
    end2 = time.time()
    elapsed_time2 = end2 - start2
    print(f'{args.total_cfs} cfs generation time: {elapsed_time2:.5f}')
    print(f'total time: {(elapsed_time1 + elapsed_time2)/args.num_inputs:.5f}')
    with open(configuration_for_proj['cf_vae_model_targetenc_' + DATA_NAME + '_times'], 'w') as f:
        json.dump(times, f)