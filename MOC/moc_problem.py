import sys
sys.path.append('../')
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pymoo.core.problem import Problem
from collections import defaultdict


class MyProblem(Problem):
    def __init__(self, base):
        # define variables, objective functions, and constraints
        self.base = base
        super().__init__(
            n_var=len(self.base.encoder_normalize_data_catalog.feature_names),
            n_obj=2,
            xl=np.array(list(self.base.feature_range_min.values())),
            xu=np.array(list(self.base.feature_range_max.values())),
            )
        self.history = defaultdict(lambda: [])

    def _evaluate(self, cfs, out, *args, **kwargs):
        # define the objective functions
        loss_1 = self.base.compute_yloss(cfs)
        loss_2 = self.base.compute_proximity_loss(cfs, self.base.query_instance)
        out["F"] = np.column_stack([loss_1, loss_2])
        
        # from (size, 1) to (size, )
        loss = np.reshape(loss_1, (-1, )) + np.reshape(loss_2, (-1, ))
        loss = np.reshape(loss, (-1, 1))
        index = np.reshape(np.arange(len(loss)), (-1, 1))
        loss = np.concatenate([index, loss], axis=1)
        # population_fitness = loss[loss[:, 1].argsort()]
        # top_genes = population_fitness[:self.base.total_CFs]
        top_genes = loss[:self.base.total_CFs]
        top_idx = [int(tup[0]) for tup in top_genes]
        self.history['yloss'].append(float(np.average(loss_1[top_idx])))
        self.history['proximity_loss'].append(float(np.average(loss_2[top_idx])))
        self.history['total_loss'].append(float(np.average(loss_1[top_idx] + loss_2[top_idx])))
        # print("yloss", np.average(loss_1[top_idx]), "proximity_loss", np.average(loss_2[top_idx]), "total_loss", np.average(loss_1[top_idx] + loss_2[top_idx]))