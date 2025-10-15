import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
def ROIevaluate(select_ROI, sub_idx, have_value_idx):
    save_dir = ('/data2/gaoxiaohui/Project1_LLMBrain_Integration/roi_data/subj0'
                + str(sub_idx) + '/roi_segmentation.pkl')
    with open(save_dir, 'rb') as f:
        roi_info = pickle.load(f)
    roi_names = roi_info['names']
    roi_voxels = roi_info['labels']
    voxel_loc = []
    for roi in select_ROI:
        matching_positions = [index for index, text in enumerate(roi_names) if text == roi]
        #如果存在两个相同的，则取对应voxel的并集
        union =roi_voxels[matching_positions[0]]
        for position_i in matching_positions[1:]:
            aa = roi_voxels[position_i]
            union = np.union1d(union, roi_voxels[position_i])
        voxel_loc.append(union)
    ## ROI comparison
    axis_labels = [v for v in select_ROI]

    ## 因为最后用于分析的是去掉颅骨等，因此需要找到其在原来空间中的位置，进而找到其在去除无关空间后的位置
    roi_voxel_list = []
    for roi_num, roi in enumerate(select_ROI):
        relevant_voxel_withoutmask = voxel_loc[roi_num]
        # 遍历B序列中的每个元素
        nonmaskvoxel_loc = []
        for i, elem in enumerate(relevant_voxel_withoutmask):
            # 在have_value_idx序列中查找元素的索引
            index = np.where(have_value_idx == elem)[0]
            if len(index) != 0:
                # 将索引记录下来
                nonmaskvoxel_loc.append(index)
        int_list_loc = [int(int_list_loc_i) for int_list_loc_i in nonmaskvoxel_loc]
        roi_voxel_list.append(int_list_loc)

    return roi_voxel_list

import numpy as np

