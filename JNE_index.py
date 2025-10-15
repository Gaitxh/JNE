import os

import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.model_selection import (
    train_test_split,
)
import h5py
import random
import nibabel as nb
from scipy.stats import pearsonr
import cortex
from Encoders import *
from untils import *
import warnings
import seaborn as sns
from scipy.stats import gaussian_kde
import scipy.stats as stats

def BrainNonlinearScore(JM_CLIP_VIT_):
    """
    Input_非线性编码输入和输出之间的雅可比矩阵
    Output_本研究定义的大脑非线性评估指标

    """
    "Step 1 计算均值，将JM矩阵维度从1000*768*显著激活体素→1*768*显著激活体素"
    mean_JM_CLIP_VIT_ = torch.mean(JM_CLIP_VIT_, dim=0) #维度 768*显著激活体素
    "Step 2 JM(1000*768*显著激活体素)-JM_mean(1*768*显著激活体素),torch.abs计算曼哈顿记录(L1-norm)"
    "另一种角度，减去均值是为了统一初始量纲，如果一个脑区对样本是敏感的，那么它的W的变化量是大的，它更可能是非线性的；反之，如果是不敏感的，则它的W变化量是小的，它可能倾向于线性，所以我们无需统一脑区(体素)之间的量纲。"
    abs_delta_JM_CLIP_VIT_ = torch.abs(JM_CLIP_VIT_-mean_JM_CLIP_VIT_) # 维度 样本个数(1000)*768*显著激活体素
    "Step 3 计算和，用于概括数据的整体水平, 这个步骤和Step2应该是一体，总体上就是计算L1-norm"
    sum_abs_delta_JM_CLIP_VIT_ = torch.sum(abs_delta_JM_CLIP_VIT_, dim=1) # 维度 样本个数*显著激活体素
    "Step 4 计算标准差，反应某一个体素下映射关系随样本变化的敏感性(W的漂移量),下面开始计算标准差"
    "这里需要科普一下均值和方差，均值反应的是整体的平均水平，而我们想要的是权重随样本的敏感性，因此我们方差更合适，方差反应的是数据点与均值的偏离程度。方差越大，数据越分散，即对不同样本的变化性越明显。"
    mean_sum_abs_delta_JM_CLIP_VIT_ = torch.mean(sum_abs_delta_JM_CLIP_VIT_, axis=0) # 维度 显著性激活体素
    "Step 5 每个数据点与均值的差的平方， 这个指标也反应了每个样本对应某个体素下的非线性指标大小"
    diff_value = (sum_abs_delta_JM_CLIP_VIT_-mean_sum_abs_delta_JM_CLIP_VIT_)**2 #维度 样本个数*显著性激活体素
    "Step 6 对每一个体素计算std,表征样本间漂移性.因为是方差，标准差的计算需要进行开方处理"
    std_sum_abs_delta_JM_CLIP_VIT_ = torch.sqrt(torch.sum(diff_value, axis=0)/diff_value.shape[0]) # 维度 显著激活体素
    JLNE = std_sum_abs_delta_JM_CLIP_VIT_
    print(f'mean_JM_CLIP_VIT_ {mean_JM_CLIP_VIT_.shape}, '
          f'abs_delta_JM_CLIP_VIT_ {abs_delta_JM_CLIP_VIT_.shape}, '
          f'sum_abs_delta_JM_CLIP_VIT_ {sum_abs_delta_JM_CLIP_VIT_.shape}, '
          f'mean_sum_abs_delta_JM_CLIP_VIT_ {mean_sum_abs_delta_JM_CLIP_VIT_.shape}, '
          f'diff_value {diff_value.shape}, '
          f'std_sum_abs_delta_JM_CLIP_VIT_ {std_sum_abs_delta_JM_CLIP_VIT_.shape}.')
    return JLNE, diff_value