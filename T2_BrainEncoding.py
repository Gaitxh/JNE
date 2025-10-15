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
sub_set = [1,2,5,7]
device = torch.device("cpu")
project_dir = os.path.dirname(os.path.abspath(__file__))
for sub_idx in sub_set:
    for layer_i in range(12, 0, -1):  # 注意，第一层是embedding的输出，后面是vit模块12层的输出
        for run_i in [
                      'clip_vit',
                      # 'vit'
                      ]:
            for encoder_type in [
                # 'res_linear',
                'res_nonlinear_relu',
                # 'res_nonlinear_gelu',
            ]:
                for bias in [False]:
                    for hidden_size in [1024]:
                        try:
                            "加载fmri信号,并实现按HCP-MMP功能性脑区进行划分，将20W个体素划分到180个脑区中"
                            "注意，为了减少计算量，我们这里只关注视觉皮层的体素"
                            h5_file = h5py.File(project_dir + '/Dataset_fMRI/'
                                                              'Stan_fmri' + '_Sub' + str(sub_idx) + '.h5', "r")
                            fmri_data = h5_file["data"][:]

                            ## 加载embedding信号
                            h5_file = h5py.File(
                                project_dir + '/Dataset_ANN/Stan_' + run_i + '_Sub' + str(
                                    sub_idx) + '.h5', "r")
                            embed_data = h5_file["data"][:, layer_i, :]

                            ## 加载被试观看相同照片的序列组,即4个被试观看了同一种图片集合，然后判断这个集合在每个被试观看10000张图片中的位置
                            ## 划分训练集和测试集
                            sub_loc = np.where(np.stack([1, 2, 5, 7]) == sub_idx)[0]
                            unique_position = \
                                np.load(project_dir + '/Dataset_ANN/Sub_info_uniqueFig.npy')[
                                sub_loc, :][0]

                            data_squence = set(np.arange(embed_data.shape[0]))
                            train_squence = np.stack(list(data_squence - set(unique_position)))
                            test_squence = unique_position

                            X_train_val = embed_data[train_squence]
                            y_train_val = fmri_data[train_squence]
                            X_test = embed_data[test_squence]
                            y_test = fmri_data[test_squence]

                            X_train, X_val, y_train, y_val = train_test_split(
                                X_train_val, y_train_val, test_size=1000 / 9000, random_state=seed)

                            ###将数据信息送入GPU
                            X_train = torch.from_numpy(X_train).to(dtype=torch.float32).to(device)
                            y_train = torch.from_numpy(y_train).to(dtype=torch.float32).to(device)
                            X_val = torch.from_numpy(X_val).to(dtype=torch.float32).to(device)
                            y_val = torch.from_numpy(y_val).to(dtype=torch.float32).to(device)
                            X_test = torch.from_numpy(X_test).to(dtype=torch.float32).to(device)
                            y_test = torch.from_numpy(y_test).to(dtype=torch.float32).to(device)

                            save_dir = project_dir + f'/Dataset_trainedModel_B{bias}_H{hidden_size}'
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            save_model_dir = save_dir + (f'/{run_i}'
                                                         f'_S{sub_idx}_L{layer_i}'
                                                         f'_{encoder_type}.pth')

                            if not os.path.exists(save_model_dir):
                                begin_time = time.time()
                                results = (
                                    encoder_train(X_train, y_train, X_val, y_val, X_test, y_test,
                                                  encoder_type=encoder_type, save_model_dir=save_model_dir,
                                                  hidden_size=hidden_size, bias=bias))
                                end_time = time.time()

                                print(f'{run_i}'
                                      f'_S{sub_idx}'
                                      f'_H{hidden_size}'
                                      f'_L{layer_i}'
                                      f'_B{bias}'
                                      f'_{encoder_type}'
                                      f'_MaxR{np.stack(results["val_r2_max"]).max():.4f}'
                                      f'_MeanR{np.stack(results["val_r2_mean"]).max():.4f}'
                                      f'_{end_time - begin_time:.2f}s')

                                "画图"
                                fig_dir = save_dir + (f'/{run_i}'
                                                      f'_S{sub_idx}_L{layer_i}'
                                                      f'_{encoder_type}.jpg')
                                fig_polt(results, fig_dir)
                            else:
                                print(f'{run_i}'
                                      f'_S{sub_idx}'
                                      f'_H{hidden_size}'
                                      f'_L{layer_i}'
                                      f'_B{bias}'
                                      f'_{encoder_type} is finished')

                        except Exception:
                            print(f'{run_i}'
                                f'_S{sub_idx}'
                                f'_H{hidden_size}'
                                f'_L{layer_i}'
                                f'_B{bias}'
                                f'_{encoder_type} '
                                  f'has something wrong. '
                                  )

