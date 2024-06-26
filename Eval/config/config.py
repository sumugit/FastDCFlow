class Config:
    data_name = 'adult'           # dataset name
    pretrain = True               # pretrain flag
    total_cfs = 1000               # total cfs
    num_inputs = 500              # number of inputs (モデル間比較: 100, モデル内比較: all, 時間: 100, 温度: 500)
    pop_size = total_cfs*10       # population size
    maxiterations = 10            # max iterations
    lr_init = 1e-3                # learning rate
    epochs = 10                   # epochs
    batch_size = 64               # batch size
    print_freq = 5                # print frequency
    pred_thrsh = 0.5              # prediction threshold
    lambda_1 = 0.01               # lambda 1
    temperature = 1.0             # temperature
    device = 'cpu'                # device
    cvae_encoded_size = 12        # encoder dim (default: 12)
    cvae_hidden_size1 = 20        # hidden size (default: 20)
    cvae_hidden_size2 = 16        # hidden size (default: 16)
    cvae_hidden_size3 = 14        # hidden size (default: 14)
    wm1 = 1e-2                    # weght decay 1
    wm2 = 1e-2                    # weght decay 2
    wm3 = 1e-2                    # weght decay 3
    encoded_size = 4              # encoder dim (default: 12)
    hidden_size1 = 8              # hidden size (default: 20)
    hidden_size2 = 6              # hidden size (default: 16)
    p_drop = 0.5                  # dropout prob
    gamma = 0.5                   # initialize param of bn scale
    gamma_train = True            # initialize BN params (scale and mean)
    delta_rate = 1.0              # control hyper-parameter alpha of dropout
    seed = 42                     # seed
    