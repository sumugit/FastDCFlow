import os
import torch
import random
import numpy as np

def setup(cfg):
    # GPU 設定
    cfg.device = torch.device("cpu")
    return cfg


def set_seed(seed):
    """config seed number
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True