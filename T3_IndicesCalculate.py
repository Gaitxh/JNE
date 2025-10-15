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
sub_set = [1]
# from T2_BrainEncoding import sub_set
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
                                ## 加载fmri信号,并实现按HCP-MMP功能性脑区进行划分，将20W个体素划分到180个脑区中
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

                                brain_map_model = encoder(input_size=X_train.shape[1], output_size=y_train.shape[1],
                                                          hidden_size=hidden_size, dropout_rate=0, bias=bias,
                                                          encoder_type=encoder_type)

                                "加载模型权重"
                                save_dir = project_dir + f'/Dataset_trainedModel_B{bias}_H{hidden_size}'
                                brain_map_model.load_state_dict(torch.load(save_dir + (f'/{run_i}'
                                                                                          f'_S{sub_idx}_L{layer_i}'
                                                                                          f'_{encoder_type}.pth')))

                                """
                                下面是计算指标R2，以及显著性体素
                                """
                                indices_save_model_dir = save_dir + (f'/{run_i}'
                                                                        f'_S{sub_idx}_L{layer_i}'
                                                                        f'_{encoder_type}_IndicesEvaluate.pth')
                                if os.path.exists(indices_save_model_dir):
                                    result = torch.load(indices_save_model_dir)
                                    R2_, _, fdr_p = result
                                    R2_ = R2_[fdr_p < 0.05]

                                    print(f'{run_i}'
                                          f'_S{sub_idx}'
                                          f'_H{hidden_size}'
                                          f'_L{layer_i}'
                                          f'_B{bias}'
                                          f'_{encoder_type}'
                                          f'R2max{R2_.max():.2f}-mean{np.mean(R2_):.4f}, '
                                          f'Sig voxel{len(R2_)}'
                                          )
                                else:

                                    "基于bootstrap实现显著体素筛选"
                                    with torch.no_grad():  ### 锁定映射模型的权重
                                        y_pred = brain_map_model(X_test)
                                        R2_ = r2_score(y_test.cpu().numpy(), y_pred.cpu().numpy())

                                    "bootstrp采样"
                                    rsq_dist = list()
                                    repeat_time = 200
                                    label_idx = np.arange(y_test.shape[0])
                                    for _ in tqdm(range(repeat_time)):
                                        sampled_idx = np.random.choice(label_idx, replace=True, size=len(label_idx))
                                        y_test_sampled = y_test[sampled_idx, :]
                                        y_hat_sampled = y_pred[sampled_idx, :]
                                        rsqs = r2_score(y_test_sampled.cpu().numpy(), y_hat_sampled.cpu().numpy())
                                        rsq_dist.append(rsqs)
                                    # fdr校正
                                    from statsmodels.stats.multitest import fdrcorrection

                                    fdr_p = fdrcorrection(np.sum(np.stack(rsq_dist) < 0, axis=0) / np.stack(rsq_dist).shape[0])[1]
                                    result = (R2_,
                                              np.stack(rsq_dist),
                                              fdr_p)
                                    torch.save(result, indices_save_model_dir)
                                    R2_ = R2_[fdr_p < 0.05]

                                print(f'{run_i}'
                                      f'_S{sub_idx}'
                                      f'_H{hidden_size}'
                                      f'_L{layer_i}'
                                      f'_B{bias}'
                                      f'_{encoder_type}'
                                      f'R2max{R2_.max():.2f}-mean{np.mean(R2_):.4f}, '
                                      f'Sig voxel{len(R2_)}'
                                      )

                            except Exception:
                                print(f'{run_i}, '
                                      f'S{sub_idx}, '
                                      f'B{bias}, '
                                      f'{encoder_type}, '
                                      f'has something wrong. '
                                      )