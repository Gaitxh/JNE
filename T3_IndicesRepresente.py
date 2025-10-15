import os
import torch
import numpy as np
from sklearn.model_selection import (
    train_test_split,
)
import h5py
import random

from Encoders import *

# 设置随机种子
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 如果使用CUDA，还需要设置CUDA的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 初始化模型
# sub_set = [1]
sub_set = [1]
device = torch.device("cpu")
project_dir = os.path.dirname(os.path.abspath(__file__))
for sub_idx in sub_set:
    for run_i in ['clip_vit', 'vit']:
        for layer_i in range(1, 13):  # 注意，第一层是embedding的输出，后面是vit模块12层的输出
            for encoder_type in ['res_linear', 'res_nonlinear_relu', 'res_nonlinear_gelu']:
                for bias in [False]:
                    for hidden_size in [1024]:
                        "加载模型权重"
                        save_dir = project_dir + f'/Dataset_trainedModel_B{bias}_H{hidden_size}'

                        """
                        下面是计算指标R2，以及显著性体素
                        """
                        indices_save_model_dir = save_dir + (f'/{run_i}'
                                                             f'_S{sub_idx}_L{layer_i}'
                                                             f'_{encoder_type}_IndicesEvaluate.pth')
                        result = torch.load(indices_save_model_dir)
                        R2_, _, fdr_p = result
                        R2_ = R2_[fdr_p < 0.05]

                        print(f'{run_i}, '
                              f'S{sub_idx}, '
                              f'H{hidden_size}, '
                              f'L{layer_i}, '
                              f'B{bias}, '
                              f'{encoder_type}, '
                              f'R2max{R2_.max():.2f}-mean{np.mean(R2_):.4f}, '
                              f'Sig voxel{len(R2_)}'
                              )
