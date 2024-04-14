import sys
import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm
sys.path.append('/workspace/Eval/')
from config.config import Config
import config.setup as setup
from counterfactual_explanation.utils.helpers import load_configuration_from_yaml
from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name, load_configuration_from_yaml)
from counterfactual_explanation.utils.mlcatalog import (
    find_latent_mean_two_classes, model_prediction, negative_prediction_index,
    positive_prediction_index, prediction_instances, make_perturbation)
from counterfactual_explanation.utils.data_catalog import (
    DataCatalog, EncoderNormalizeDataCatalog, LabelEncoderNormalizeDataCatalog, TargetEncoderNormalizingDataCatalog,
    TensorDatasetTraning)
sys.path.append('/workspace/CF_MOGA/')
# dataset preprocess
import warnings
warnings.filterwarnings("ignore")
# Counterfactual library
from MOC.moc import MOGA
# MOCO
from MOC.moc_problem import MyProblem
# pymoo library
from pymoo.util.misc import stack
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.repair.rounding import RoundingRepair

if __name__ == '__main__':
    args = setup.setup(Config)
    DATA_NAME = args.data_name
    CONFIG_PATH = '/workspace/Eval/configuration/data_catalog.yaml'
    CONFIG_FOR_PROJECT = '/workspace/Eval/configuration/project_configurations.yaml'
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_train_input']

    predictive_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
        DATA_NAME, encoding='onehotenc')
    predictive_model = predictive_model.cpu()
    
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    feature_names = encoder_normalize_data_catalog.feature_names

    """ test input """
    TOTAL_CFS = args.total_cfs
    inputs = pd.read_csv(configuration_for_proj[DATA_NAME + '_onehotenc_test_input'])
    inputs = inputs.drop(target, axis=1)

    start = time.time()
    negative_cnt = 0
    for query_instance in tqdm(inputs.values):
        if negative_cnt >= args.num_inputs:
            break
        query_instance = torch.Tensor(query_instance)
        test_preds = model_prediction(predictive_model, query_instance)
        if test_preds.item() >= args.pred_thrsh:
            continue
        
        
        negative_cnt += 1
        # model
        exp_genetic = MOGA(encoder_normalize_data_catalog, predictive_model, query_instance)
        # MOCO problem setting
        problem = MyProblem(exp_genetic)
        # initialize the genetic algorithm
        algorithm = NSGA2(
            pop_size=args.pop_size,
            n_offsprings=args.pop_size - TOTAL_CFS,
            sampling=LHS(),
            crossover=UniformCrossover(prob=0.5, repair=RoundingRepair()),
            mutation=PolynomialMutation(eta=20),
            eliminate_duplicates=True
        )
        # finish condition
        termination = get_termination("n_gen", args.maxiterations)
        # compute optimazaition
        res = minimize(problem,
                    algorithm,
                    termination,
                    seed=42,
                    save_history=True,
                    verbose=False)
        # print('end!')
        losses = res.F # objective function value
        loss_df = pd.DataFrame(losses, columns=['yloss', 'prox_loss'])
        pop = res.pop.get("X")
        pop_pred = exp_genetic.predict_fn(pop)
        x_cf = np.hstack((pop, pop_pred))
        x_cf_df = pd.DataFrame(x_cf, columns=feature_names + [target])
        x_cf_df.drop(target, axis=1).to_csv(configuration_for_proj["cfs_raw_moc_onehotenc_" + DATA_NAME] + f"neg_{negative_cnt}.csv", index=False)
        x_cf_df = encoder_normalize_data_catalog.denormalize_continuous_feature(x_cf_df)
        x_cf_df = encoder_normalize_data_catalog.convert_from_one_hot_to_original_forms(x_cf_df)
        x_cf_df.to_csv(configuration_for_proj["cfs_moc_onehotenc_" + DATA_NAME] + f"neg_{negative_cnt}.csv", index=False)
        loss_df.to_csv(configuration_for_proj["cfs_moc_onehotenc_loss_" + DATA_NAME] + f"neg_{negative_cnt}.csv", index=False)
        # pareto set and front
        # ps = problem.pareto_set(use_cache=False, flatten=False)
        # pf = problem.pareto_front(use_cache=False, flatten=False)
    end = time.time()
    elapsed_time = end - start
    print(f'{args.total_cfs} cfs generation time: {elapsed_time:.5f}')
