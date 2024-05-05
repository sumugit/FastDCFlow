# FastDCFlow
This is our implementation for the paper:

_Model-Based Counterfactual Explanations Incorporating Feature Space Attributes for Tabular Data [arkiv link(https://arxiv.org/abs/2404.13224)]_ 

Yuta Sumiya, Hayaru shouno

_at IEEE WCCI, 2024_

## Environments
- Python 3.9
- Pytorch 1.13.1
- Numpy 1.23.5
- Pandas 2.0.1

## Datasets
To evaluate the proposed approach, we used three open datasets: Adult, Bank, and Churn, which integrate both categorical and continuous variables:

- Adult: <https://archive.ics.uci.edu/dataset/2/adult> or <https://www.kaggle.com/datasets/wenruliu/adult-income-dataset>
- Bank: <https://archive.ics.uci.edu/dataset/222/bank+marketing> or <https://dataplatform.cloud.ibm.com/exchange/public/entry/view/50fa9246181026cd7ae2a5bc7ea568a4>
- Churn: <https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113>

There is a small dataset sample included in the folder `Eval/test_input`, which can be used to test the correctness of the code.

## Usage
You shold first edit `Eval/config/config.py` as below:
~~~~
class Config:
    data_name = 'adult'           # dataset name
    pretrain = True               # pretrain flag
    total_cfs = 1000              # total cfs
    num_inputs = 500              # number of inputs
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
~~~~

To use the FastDCFlow framework, follow these steps:
- Clone the repository to your local machine.
- Run `Eval/split_data.py` to split train and test data.
- Run `Eval/train_classifier_targetenc.py` to train the binary classifier.
- Edit `Eval/config/config.py` and set pretrain to `False`
- Run `FastDCFlow/fastdcflow_targetenc.py` to train the normalizing flow model on the training data.
- Edit `Eval/config/config.py` and set pretrain to `True`
- Run `FastDCFlow/fastdcflow_targetenc.py` to generate counterfactuals for each test input.

## Citation
Please cite our paper if you use our codes. Thanks!
```
coming soon.
```

In case that you have any difficulty about the implementation or you are interested in our work,  please feel free to communicate with us by:

Author: Yuta Sumiya (sumiya@uec.ac.jp / diddy2983@gmail.com)

Also, welcome to visit my academic homepage: https://sumugit.github.io
