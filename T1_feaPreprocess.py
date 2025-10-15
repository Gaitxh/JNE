import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nb
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import h5py
import torch
import os
### 这个地方只关注representation，不需要进行fmri数据处理,因此，这里只提取了每一个block的前馈层的输出

for feature_name in ["vit_visual_block_outputs_stack"]:
    basic_dir = os.path.dirname(os.path.abspath(__file__)) + '/Dataset_ANN/'
    standard_ = True
    feature_output_dir = basic_dir + feature_name + '.npy'

    if feature_name == "clip_visual_block_outputs_stack":feature_name = 'clip_vit'
    elif feature_name == "vit_visual_block_outputs_stack":feature_name = 'vit'

    ANN_response = np.load(feature_output_dir)
    sub_num = [1, 2, 5, 7]
    data_dir = '/data1/shigaosheng/data_model/nsd/betas/'
    figure_unique_set = list()
    for sub_idx in sub_num:
        basic_path = '/data2/gaoxiaohui/pycortex/NSD_relevantdata/'
        # 加载tsv文件
        info = pd.read_csv(basic_path + 'Sub0' + str(sub_idx) + '/responses.tsv', delimiter='\t')
        complete_info = np.zeros([len(info['SESSION']), 3])
        complete_info[:, 0] = info['SESSION']
        complete_info[:, 1] = info['RUN']
        complete_info[:, 2] = info['73KID'] - 1  # 注意，这里的顺序是从1开始的，而不是0，因此需要减1
        ##加载掩码的体素
        mask_image = nb.load(basic_path + 'Sub0' + str(sub_idx) + '/T1_to_func1pt8mm_brain_mask.nii.gz').get_fdata()
        mask_image_flat = mask_image.reshape(1, -1)
        have_value_idx = np.where(mask_image_flat[0] == 1.0)[0]  # 基于brain mask掩码beta中的非脑组织信号，定位脑组织信号的序号位置

        ANN_response_list = []

        for session_i in np.unique(complete_info[:, 0]):

            session_loc = np.where(complete_info[:, 0] == session_i)[0]  # 因为一次加载的是750个试次的结果，需要找到每一个session对应的坐标，进而方便run的筛选
            session_run_sequence = complete_info[session_loc, :]
            for run_i in np.unique(session_run_sequence[:, 1]):
                run_loc = np.where(session_run_sequence[:, 1] == run_i)[0]  # 找寻每一个run的location，从1~12

                ### 提取对应run下的观看的照片的标号
                run_Fig_loc = session_run_sequence[run_loc, 2].astype(np.int32)  # 得到这个run下看图片的顺序标号
                if standard_:
                    run_ANN_response = np.zeros_like(ANN_response[run_Fig_loc])
                    for layer_i in range(ANN_response.shape[1]):
                        run_ANN_response[:, layer_i, :] = scaler.fit_transform(ANN_response[run_Fig_loc, layer_i, :])
                else:
                    run_ANN_response = ANN_response[run_Fig_loc]

                ANN_response_list.append(run_ANN_response)
            #     print('Sub', sub_idx, ' Session', session_i, ' Run', run_i, ' has been finished.')
            # print('Sub', sub_idx, ' Session', session_i, ' has been finished.')

        # 将所有session下所有run的数据串在一起，构成30000维度的矩阵
        Combine_ANN_response_list = np.concatenate(ANN_response_list, axis=0)

        # 将相同的图进行合并
        figure_unique = np.unique(complete_info[:, 2])
        figure_unique_set.append(figure_unique)
        print('Sub', sub_idx, ' has been finished, with len ', len(figure_unique), ', and shape ', complete_info.shape)
        # 重新归纳整理重复的，进行平均求解
        ANN_response_list = []

        for fig_i in tqdm(figure_unique):
            fig_loc = np.where(complete_info[:, 2] == fig_i)[0]
            ##Embedding
            ANN_response_list.append(np.mean(Combine_ANN_response_list[fig_loc], axis=0))

        Combine_ANN_response_list = np.stack(ANN_response_list, axis=0)
        if standard_:
            save_dir = basic_dir + '/Stan_' + feature_name + '_Sub' + str(
                sub_idx) + '.h5'
        else:
            save_dir = basic_dir + '/NoStan_' + feature_name + '_Sub' + str(
                sub_idx) + '.h5'

        with h5py.File(save_dir, 'w') as hf:
            hf.create_dataset('data', data=Combine_ANN_response_list)

        print('Sub', sub_idx, ' has been finished.')
    print("finished.")

    # 下面则是计算unique fig在每一个被试中所处的位置，用于区分训练集和测试集，不让他们之间产生混淆;下面，bingji是所有被试都看过的1000张图片，
    # 因为NSD数据集规定8个被试，每个被试观看独特地照片9000张，一致的照片1000张，因此一共9000*8+1000=73000张照片
    bingji_figure = list((set(figure_unique_set[0]) & set(figure_unique_set[1]) &
                     set(figure_unique_set[2]) & set(figure_unique_set[3])))
    position_set = list()
    for sub_i in range(len(figure_unique_set)):
        position_sub = list()
        for element_i in bingji_figure:
            position_sub.append(np.where(figure_unique_set[sub_i] == element_i)[0])
        position_set.append(np.concatenate(position_sub))

    result = (figure_unique_set, bingji_figure, position_set)

    np.save(basic_dir+'/Sub_info_uniqueFig.npy', position_set, allow_pickle=True)
