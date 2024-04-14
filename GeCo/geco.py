import numpy as np
import pandas as pd
import random
import timeit
import torch
from torch import nn
from collections import defaultdict

class Genetic():
    def __init__(self, encoder_normalize_data_catalog, predictive_model):
        self.encoder_normalize_data_catalog = encoder_normalize_data_catalog
        self.pred_model = predictive_model
        self.targetenc_dataset = self.encoder_normalize_data_catalog.data_frame
        self.cfs = []
        self.history = defaultdict(lambda: [])

    def do_loss_initializations(self):
        """iniaialize loss functions."""
        self.loss_y = nn.BCELoss(reduction='none')
        self.loss_prox = nn.MSELoss(reduction='none')

    def do_random_init(self, num_inits):
        remaining_cfs = np.zeros((num_inits, len(self.encoder_normalize_data_catalog.feature_names)))
        # kx is the number of valid inits found so far
        kx = 0
        while kx < num_inits:
            one_init = np.zeros(len(self.encoder_normalize_data_catalog.feature_names))
            for jx, feature in enumerate(self.encoder_normalize_data_catalog.feature_names):
                one_init[jx] = np.random.uniform(
                    self.targetenc_dataset[feature].min(), self.targetenc_dataset[feature].max())
            remaining_cfs[kx] = one_init
            kx += 1
        
        remaining_cfs = pd.DataFrame(remaining_cfs, columns=self.encoder_normalize_data_catalog.feature_names)
        return remaining_cfs

    def do_cf_initializations(self, total_CFs, population_size):
        """Intializes CFs and other related variables."""
        self.total_CFs = total_CFs
        self.population_size = population_size
        # CF initialization
        self.cfs = self.do_random_init(self.population_size)

    def do_param_initializations(self, total_CFs, initialization):
        """initializes parameters for the genetic algorithm."""
        if len(self.cfs) != total_CFs:
            # initialize CFs
            self.do_cf_initializations(total_CFs, initialization)
        else:
            self.total_CFs = total_CFs
        # initialize loss functions
        self.do_loss_initializations()


    def generate_counterfactuals(self, query_instance, total_CFs, population_size,
                                maxiterations):
        self.start_time = timeit.default_timer()

        # get the predicted value of query_instance
        self.test_pred = self.pred_model(query_instance).detach().numpy()[0]
        # parameter initialization
        self.do_param_initializations(total_CFs, population_size)
        # generate counterfactuals
        final_cfs = self.find_counterfactuals(query_instance, maxiterations)
        final_cfs = torch.tensor(final_cfs, dtype=torch.float32)

        return final_cfs


    def compute_yloss(self, cfs):
        """Computes the first part (y-loss) of the loss function."""
        y_cf = self.pred_model(cfs)
        y_expected = torch.ones(y_cf.shape)
        yloss = self.loss_y(y_cf, y_expected)
        
        return yloss

    def compute_proximity_loss(self, cfs, query_instance):
        """Compute weighted distance between two vectors."""
        cfs = torch.tensor(cfs, dtype=torch.float32, requires_grad=False)
        query_instance = torch.tensor(query_instance, dtype=torch.float32, requires_grad=False)
        proximity_loss = self.loss_prox(cfs, query_instance).sum(dim=1)

        return proximity_loss

    def compute_loss(self, population, query_instance):
        """Computes the overall loss"""
        population = torch.tensor(population, dtype=torch.float32)
        yloss = self.compute_yloss(population).squeeze(1).detach().numpy()
        proximity_loss = self.compute_proximity_loss(population, query_instance).detach().numpy()
        loss = np.reshape(np.array(yloss + proximity_loss), (-1, 1))
        index = np.reshape(np.arange(len(population)), (-1, 1))
        loss = np.concatenate([index, loss], axis=1)
        
        # population_fitness = loss[loss[:, 1].argsort()]
        # top_genes = population_fitness[:self.total_CFs]
        # select the top total_CFs genes
        top_genes = loss[:self.total_CFs]
        top_idx = [int(tup[0]) for tup in top_genes]
        self.history['yloss'].append(float(np.average(yloss[top_idx])))
        self.history['proximity_loss'].append(float(np.average(proximity_loss[top_idx])))
        self.history['total_loss'].append(float(np.average(yloss[top_idx] + proximity_loss[top_idx])))
        # print("yloss", np.average(yloss[top_idx]), "proximity_loss", np.average(proximity_loss[top_idx]), "total_loss", np.average(yloss[top_idx] + proximity_loss[top_idx]))

        return loss

    def mate(self, k1, k2):
        """Performs mating and produces new offsprings"""
        # genes for the new offspring
        one_init = np.zeros(len(self.encoder_normalize_data_catalog.feature_names))
        for j in range(len(self.encoder_normalize_data_catalog.feature_names)):
            # reference to the genes of the parents
            gp1 = k1[j]
            gp2 = k2[j]
            feat_name = self.encoder_normalize_data_catalog.feature_names[j]

            # random probability
            prob = random.random()

            # crossover operation
            if prob < 0.40:
                # if prob is less than 0.40, insert gene from parent 1
                one_init[j] = gp1
            elif prob < 0.80:
                # if prob is between 0.40 and 0.80, insert gene from parent 2
                one_init[j] = gp2
            # mutation operation
            else:
                # get a random value between the min and max value of the feature
                one_init[j] = np.random.uniform(self.targetenc_dataset[feat_name].min(), self.targetenc_dataset[feat_name].max())
        # new offspring
        return one_init

    def find_counterfactuals(self, query_instance, maxiterations):
        """Finds counterfactuals by generating cfs through the genetic algorithm"""
        population = self.cfs.values
        iterations = 0

        self.average_loss_list = []
        self.average_top_loss_list = []

        for iterations in range(maxiterations):
            # compute the loss of the population
            population_fitness = self.compute_loss(population, query_instance)
            population_fitness = population_fitness[population_fitness[:, 1].argsort()]
            #print("generation", iterations, "loss", np.average(population_fitness))
            self.average_loss_list.append(np.average([val[1] for val in population_fitness]))
            # selection
            top_members = self.total_CFs
            # save the average loss of the top members
            self.average_top_loss_list.append(np.average([val[1] for val in population_fitness[:top_members]]))
            # take over the top 50% of the fittest members of the current generation
            new_generation_1 = np.array([population[int(tup[0])] for tup in population_fitness[:top_members]])
            # rest of the next generation obtained from top of fittest members of current generation
            rest_members = self.population_size - top_members
            # change the rest of the members with the offsprings of the top 50% of the fittest members
            new_generation_2 = np.zeros((rest_members, len(self.encoder_normalize_data_catalog.feature_names)))
            for new_gen_idx in range(rest_members):
                parent1 = random.choice(population[:int(len(population) / 2)])
                parent2 = random.choice(population[:int(len(population) / 2)])
                # crossover
                child = self.mate(parent1, parent2)
                new_generation_2[new_gen_idx] = child

            if self.total_CFs > 0:
                # new generation
                population = np.concatenate([new_generation_1, new_generation_2])
            else:
                population = new_generation_2
            iterations += 1

        # list of counterfactuals
        self.cfs_preds = []
        population = population[:self.total_CFs]
        for i in range(self.total_CFs):
            prediction =  self.pred_model(torch.tensor(population[i], dtype=torch.float32)).detach().numpy()[0]
            self.cfs_preds.append(prediction)
        
        # excution time
        self.elapsed = timeit.default_timer() - self.start_time
        m, s = divmod(self.elapsed, 60)

        return population
