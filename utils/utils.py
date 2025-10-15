import os
from random import weibullvariate
import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import (
    KFold,
    PredefinedSplit,
    train_test_split,
    ShuffleSplit,
)
import h5py
from scipy.stats import pearsonr
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import json
import pandas as pd
from tqdm import tqdm
import nibabel as nb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
import h5py
# import nilearn
# from nilearn.image import load_img, smooth_img
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, Lasso, MultiTaskLassoCV, MultiTaskLasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
# from fracridge import FracRidgeRegressorCV, FracRidgeRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import time
import h5py
import cortex
import seaborn as sns
import pickle
import matplotlib as mpl
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
from itertools import product
from scipy.stats import entropy
import math
import matplotlib.patches as patches
def scoring(y, yhat):
    return -torch.nn.functional.mse_loss(yhat, y)


def r2_score(Real, Pred):
    # print(Real.shape)
    # print(Pred.shape)
    SSres = np.mean((Real - Pred) ** 2, 0)
    # print(SSres.shape)
    SStot = np.var(Real, 0)
    # print(SStot.shape)
    return np.nan_to_num(1 - SSres / SStot)


def _validate_ls(ls):
    """Ensure that ls is a 1-dimensional torch float/double tensor."""
    try:
        assert isinstance(ls, torch.Tensor)
        assert ls.dtype is torch.float or ls.dtype is torch.double
        assert len(ls.shape) == 1
    except AssertionError:
        raise AttributeError(
            "invalid ls: should be 1-dimensional torch float/double tensor"
        )


def _validate_XY(X, Y):
    """Ensure that X and Y are 2-dimensional torch float/double tensors, with
    proper sizes."""
    try:
        for inp in [X, Y]:
            assert isinstance(inp, torch.Tensor)
            assert inp.dtype is torch.float or inp.dtype is torch.double
            assert len(inp.shape) == 2
        assert X.dtype is Y.dtype
        assert X.shape[0] == Y.shape[0]
    except AssertionError:
        raise AttributeError(
            "invalid inputs: X and Y should be float/double tensors of shape "
            "(n, d) and (n, m) respectively, where n is the number of samples, "
            "d is the number of features, and m is the number of outputs"
        )


class MultiRidge:
    """Ridge model for multiple outputs and regularization strengths. A separate
    model is fit for each (output, regularization) pair."""

    def __init__(self, ls, scale_X=True, scale_thresh=1e-8):
        """
        Arguments:
                       ls: 1-dimensional torch tensor of regularization values.
                  scale_X: whether or not to scale X.
             scale_thresh: when standardizing, standard deviations below this
                           value are not used (i.e. they are set to 1).
        """
        self.ls = ls
        self.scale_X = scale_X
        self.scale_thresh = scale_thresh
        self.X_t = None
        self.Xm = None
        self.Xs = None
        self.e = None
        self.Q = None
        self.Y = None
        self.Ym = None

    def fit(self, X, Y):
        """
        Arguments:
            X: 2-dimensional torch tensor of shape (n, d) where n is the number
               of samples, and d is the number of features.
            Y: 2-dimensional tensor of shape (n, m) where m is the number of
               targets.
        """
        self.Xm = X.mean(dim=0, keepdim=True)
        X = X - self.Xm
        if self.scale_X:
            self.Xs = X.std(dim=0, keepdim=True)
            self.Xs[self.Xs < self.scale_thresh] = 1
            X = X / self.Xs

        self.X_t = X.t()
        _, S, V = self.X_t.svd()
        self.e = S.pow_(2)
        self.Q = self.X_t @ V

        self.Y = Y
        self.Ym = Y.mean(dim=0)

        return self

    def _compute_pred_interms(self, y_idx, X_te_p):
        Y_j, Ym_j = self.Y[:, y_idx], self.Ym[y_idx]
        p_j = self.X_t @ (Y_j - Ym_j)
        r_j = self.Q.t() @ p_j
        N_te_j = X_te_p @ p_j
        return Ym_j, r_j, N_te_j

    def _predict_single(self, l, M_te, Ym_j, r_j, N_te_j):
        Yhat_te_j = (1 / l) * (N_te_j - M_te @ (r_j / (self.e + l))) + Ym_j
        return Yhat_te_j

    def _compute_single_beta(self, l, y_idx):
        Y_j, Ym_j = self.Y[:, y_idx], self.Ym[y_idx]
        beta = (1 / l) * (
                self.X_t @ (Y_j - Ym_j)
                - self.Q / (self.e + l) @ self.Q.t() @ self.X_t @ (Y_j - Ym_j)
        )
        return beta

    def get_model_weights_and_bias(self, l_idxs):
        betas = torch.zeros((self.X_t.shape[0], len(l_idxs)))
        for j, l_idx in enumerate(l_idxs):
            l = self.ls[l_idx]
            betas[:, j] = self._compute_single_beta(l, j)
        return betas, self.Ym

    def get_prediction_scores(self, X_te, Y_te, scoring):
        """Compute predictions for each (regulariztion, output) pair and return
        the scores as produced by the given scoring function.

        Arguments:
               X_te: 2-dimensional torch tensor of shape (n, d) where n is the
                     number of samples, and d is the number of features.
               Y_te: 2-dimensional tensor of shape (n, m) where m is the
                     number of targets.
            scoring: scoring function with signature scoring(y, yhat).

        Returns a (m, M) torch tensor of scores, where M is the number of
        regularization values.
        """
        X_te = X_te - self.Xm
        if self.scale_X:
            X_te = X_te / self.Xs
        M_te = X_te @ self.Q

        scores = torch.zeros(Y_te.shape[1], len(self.ls), dtype=X_te.dtype)
        for j, Y_te_j in enumerate(Y_te.t()):
            Ym_j, r_j, N_te_j = self._compute_pred_interms(j, X_te)
            for k, l in enumerate(self.ls):
                Yhat_te_j = self._predict_single(l, M_te, Ym_j, r_j, N_te_j)
                scores[j, k] = scoring(Y_te_j, Yhat_te_j).item()
        return scores

    def predict_single(self, X_te, l_idxs):
        """Compute a single prediction corresponding to each output.

        Arguments:
              X_te: 2-dimensional torch tensor of shape (n, d) where n is the
                    number of samples, and d is the number of features.
            l_idxs: iterable of length m (number of targets), with indexes
                    specifying the l value to use for each of the targets.

        Returns a (n, m) tensor of predictions.
        """
        X_te = X_te - self.Xm
        if self.scale_X:
            X_te = X_te / self.Xs

        M_te = X_te @ self.Q

        Yhat_te = []
        for j, l_idx in enumerate(l_idxs):
            Ym_j, r_j, N_te_j = self._compute_pred_interms(j, X_te)
            l = self.ls[l_idx]
            Yhat_te_j = self._predict_single(l, M_te, Ym_j, r_j, N_te_j)
            Yhat_te.append(Yhat_te_j)

        Yhat_te = torch.stack(Yhat_te, dim=1)
        return Yhat_te


class RidgeCVEstimator:
    def __init__(self, ls, cv, scoring, scale_X=True, scale_thresh=1e-8):
        """Cross-validated ridge estimator.

        Arguments:
                       ls: 1-dimensional torch tensor of regularization values.
                       cv: cross-validation object implementing split.
                  scoring: scoring function with signature scoring(y, yhat).
                  scale_X: whether or not to scale X.
             scale_thresh: when standardizing, standard deviations below this
                           value are not used (i.e. they are set to 1).
        """
        _validate_ls(ls)
        self.ls = ls
        self.cv = cv
        self.scoring = scoring
        self.scale_X = scale_X
        self.scale_thresh = scale_thresh
        self.base_ridge = None
        self.mean_cv_scores = None
        self.best_l_scores = None
        self.best_l_idxs = None

    def fit(self, X, Y, groups=None):
        """Fit ridge model to given data.

        Arguments:
                 X: 2-dimensional torch tensor of shape (n, d) where n is the
                    number of samples, and d is the number of features.
                 Y: 2-dimensional tensor of shape (n, m) where m is the number
                    of targets.
            groups: groups used for cross-validation; passed directly to
                    cv.split.

        A separate model is learned for each target i.e. Y[:, j].
        """
        _validate_XY(X, Y)
        cv_scores = []

        for idx_tr, idx_val in self.cv.split(X, Y, groups):
            X_tr, X_val = X[idx_tr], X[idx_val]
            Y_tr, Y_val = Y[idx_tr], Y[idx_val]

            self.base_ridge = MultiRidge(self.ls, self.scale_X, self.scale_thresh)
            self.base_ridge.fit(X_tr, Y_tr)
            split_scores = self.base_ridge.get_prediction_scores(
                X_val, Y_val, self.scoring
            )
            cv_scores.append(split_scores)
            del self.base_ridge

        cv_scores = torch.stack(cv_scores)
        self.mean_cv_scores = cv_scores.mean(dim=0)
        self.best_l_scores, self.best_l_idxs = self.mean_cv_scores.max(dim=1)
        self.base_ridge = MultiRidge(self.ls, self.scale_X, self.scale_thresh)
        self.base_ridge.fit(X, Y)
        return self

    def predict(self, X):
        """Predict using cross-validated model.

        Arguments:
            X_te: 2-dimensional torch tensor of shape (n, d) where n is the
                  number of samples, and d is the number of features.

        Returns a (n, m) matrix of predictions.
        """
        if self.best_l_idxs is None:
            raise RuntimeError("cannot predict without fitting")
        return self.base_ridge.predict_single(X, self.best_l_idxs)

    def get_model_weights_and_bias(self):
        if self.best_l_idxs is None:
            raise RuntimeError("cannot return weight without fitting")
        return self.base_ridge.get_model_weights_and_bias(self.best_l_idxs)


def column_standardization(test_data, train_data):
    # 计算每列的均值和标准差
    means = np.mean(train_data, axis=0)
    stds = np.std(train_data, axis=0)
    test_data = (test_data - means) / stds
    return test_data


def data_save(betas_sub_path, feature_name, result, sub_idx, layer_i, type_name):
    ## 存储rsqs_ma
    save_dir = betas_sub_path + feature_name + '_rsqs_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + type_name + '.h5'
    with h5py.File(save_dir, 'w') as hf:
        # 创建一个数据集
        hf.create_dataset('data', data=result[0])
    print('rsqs_ma, saved.')

    ## 存储mean_cv_scores_ma
    save_dir = betas_sub_path + feature_name + '_mean_cv_scores_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + type_name + '.h5'
    with h5py.File(save_dir, 'w') as hf:
        # 创建一个数据集
        hf.create_dataset('data', data=result[1])
    print('mean_cv_scores_ma, saved.')

    ## 存储best_l_scores_ma
    save_dir = betas_sub_path + feature_name + '_best_l_scores_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + type_name + '.h5'
    with h5py.File(save_dir, 'w') as hf:
        # 创建一个数据集
        hf.create_dataset('data', data=result[2])
    print('best_l_scores_ma, saved.')

    ## 存储best_l_idxs_ma
    save_dir = betas_sub_path + feature_name + '_best_l_idxs_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + type_name + '.h5'
    with h5py.File(save_dir, 'w') as hf:
        # 创建一个数据集
        hf.create_dataset('data', data=result[3])
    print('best_l_idxs_ma, saved.')

    ## 存储y_test_ma
    save_dir = betas_sub_path + feature_name + '_y_test_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + type_name + '.h5'
    with h5py.File(save_dir, 'w') as hf:
        # 创建一个数据集
        hf.create_dataset('data', data=result[4])
    print('y_test_ma, saved.')

    ## 存储y_hat_ma
    save_dir = betas_sub_path + feature_name + '_y_hat_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + type_name + '.h5'
    with h5py.File(save_dir, 'w') as hf:
        # 创建一个数据集
        hf.create_dataset('data', data=result[5])
    print('y_hat_ma, saved.')

    ## 存储weight_ma
    save_dir = betas_sub_path + feature_name + '_weight_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + type_name + '.h5'
    with h5py.File(save_dir, 'w') as hf:
        # 创建一个数据集
        hf.create_dataset('data', data=result[6])
    print('weight_ma, saved.')

    ## 存储bias_ma
    save_dir = betas_sub_path + feature_name + '_bias_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + type_name + '.h5'
    with h5py.File(save_dir, 'w') as hf:
        # 创建一个数据集
        hf.create_dataset('data', data=result[7])
    print('bias_ma, saved.')


def r2_score(Real, Pred):
    # print(Real.shape)
    # print(Pred.shape)
    SSres = np.mean((Real - Pred) ** 2, 0)
    # print(SSres.shape)
    SStot = np.var(Real, 0)
    # print(SStot.shape)
    return np.nan_to_num(1 - SSres / SStot)


def bootstrap_sampling(yhat, y_test, repeat, seed):
    np.random.seed(seed)
    rsq_dist = list()
    label_idx = np.arange(yhat.shape[0])
    for _ in tqdm(range(repeat)):
        sampled_idx = np.random.choice(label_idx, replace=True, size=len(label_idx))
        y_test_sampled = y_test[sampled_idx, :]
        y_hat_sampled = yhat[sampled_idx, :]
        rsqs = r2_score(y_test_sampled, y_hat_sampled)
        rsq_dist.append(rsqs)

    return rsq_dist


def fdr_correct_p(var, r2):
    ## 计算均值和方差
    mean_value = np.mean(var, axis=0)
    std_value = np.std(var, axis=0)

    from scipy.stats import norm
    prob1 = norm.cdf(r2, loc=mean_value, scale=std_value)
    prob = np.abs(1 - 2 * prob1)
    # 这个可以理解为实际计算的r2和随机计算的r2之间的偏移率，可以理解为错误率，
    # 当实际的r2与重采样得到的mean r2越近，越说明这些值不是受随机的影响。同样地，这个值越小越好。

    from statsmodels.stats.multitest import fdrcorrection
    fdr_p = fdrcorrection(prob)  # corrected p
    return fdr_p, prob

def data_load(betas_sub_path, feature_name, sub_idx, layer_i, type_name):
    ## 存储rsqs_ma
    save_dir = betas_sub_path + feature_name + '_rsqs_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + type_name + '.h5'
    R2 = h5py.File(save_dir, 'r')['data'][:]

    save_dir = betas_sub_path + feature_name + '_req_dist_corr_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + type_name + '.npy'
    R2_list = np.load(save_dir)

    save_dir = betas_sub_path + feature_name + '_fdr_p_corr_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + type_name + '.pth'
    fdr_p, prob = torch.load(save_dir, map_location='cpu')

    return R2, R2_list, fdr_p, prob

def data_load_signal(betas_sub_path, feature_name, zscore_type, sub_idx, layer_i):

    ####读取训练好的权重
    save_dir = betas_sub_path + feature_name + '_weight_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + 'image2text' + '.h5'
    weight_image2text = h5py.File(save_dir, 'r')['data'][:]

    save_dir = betas_sub_path + feature_name + '_weight_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + 'text2image' + '.h5'
    weight_text2image = h5py.File(save_dir, 'r')['data'][:]

    save_dir = betas_sub_path + feature_name + '_weight_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + 'image2image' + '.h5'
    weight_image2image = h5py.File(save_dir, 'r')['data'][:]

    save_dir = betas_sub_path + feature_name + '_weight_ma_sub_' + str(
        sub_idx) + '_layer' + str(layer_i) + '_' + 'text2text' + '.h5'
    weight_text2text = h5py.File(save_dir, 'r')['data'][:]


    ####读取integration特征
    save_dir = '/data2/gaoxiaohui/pycortex/Dataset/Fea_' + zscore_type + '/' + zscore_type + '_InteImage2text_Sub' + str(
        sub_idx) + '.h5'
    h5_file = h5py.File(save_dir, "r")
    cross_image_text_array = h5_file["data"][:]

    save_dir = '/data2/gaoxiaohui/pycortex/Dataset/Fea_' + zscore_type + '/' + zscore_type + '_InteText2image_Sub' + str(
        sub_idx) + '.h5'
    h5_file = h5py.File(save_dir, "r")
    cross_text_image_array = h5_file["data"][:]

    save_dir = '/data2/gaoxiaohui/pycortex/Dataset/Fea_' + zscore_type + '/' + zscore_type + '_InteImage2image_Sub' + str(
        sub_idx) + '.h5'
    h5_file = h5py.File(save_dir, "r")
    self_image_array = h5_file["data"][:]

    save_dir = '/data2/gaoxiaohui/pycortex/Dataset/Fea_' + zscore_type + '/' + zscore_type + '_InteText2text_Sub' + str(
        sub_idx) + '.h5'
    h5_file = h5py.File(save_dir, "r")
    self_text_array = h5_file["data"][:]

    ####读取fmri数据
    save_dir = '/data2/gaoxiaohui/pycortex/Dataset/Stan_fMRI_Sub' + str(sub_idx) + '.h5'
    h5_file = h5py.File(save_dir, "r")
    fmri_data = h5_file["data"][:]


    ####加载测试集数据
    _, image2text_test, _, y_test = train_test_split(
        cross_image_text_array[:, layer_i, :], fmri_data, test_size=0.2, random_state=42
    )
    _, text2image_test, _, y_test = train_test_split(
        cross_text_image_array[:, layer_i, :], fmri_data, test_size=0.2, random_state=42
    )
    _, image2image_test, _, y_test = train_test_split(
        self_image_array[:, layer_i, :], fmri_data, test_size=0.2, random_state=42
    )
    _, text2text_test, _, y_test = train_test_split(
        self_text_array[:, layer_i, :], fmri_data, test_size=0.2, random_state=42
    )

    ####基于测试集进行预测
    y_pred_image2text = image2text_test @ weight_image2text
    y_pred_text2image = text2image_test @ weight_text2image
    y_pred_image2image = image2image_test @ weight_image2image
    y_pred_text2text = text2text_test @ weight_text2text

    return y_test, y_pred_image2text, y_pred_text2image, y_pred_image2image, y_pred_text2text

def ROIevaluate(select_ROI, sub_idx, have_value_idx):
    save_dir = '/data2/gaoxiaohui/Project1_LLMBrain_Integration/roi_data/subj0' + str(sub_idx) + '/roi_segmentation.pkl'
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

def scatter_plot(labels, data, type):
    # 创建一个图形
    plt.figure(figsize=(25, 3))
    # 逐组绘制散点图
    for i, group in enumerate(data):
        x = np.full_like(group, i + 1)  # 为每组数据创建相应的横坐标
        x = x + np.random.uniform(-0.1, 0.1, size=x.shape)  # 为每个点添加随机的左右偏移量
        # 标记大于0的点为红色，小于等于0的点为灰色
        if type == "image":
            colors = np.where(group > 0, 'red', 'gray')
        elif type == "text":
            colors = np.where(group < 0, 'red', 'gray')
        plt.scatter(x, group, color=colors, s=10, alpha=0.6, label=f'Group {labels[i]}')  # s参数控制点的大小，alpha参数控制透明度
        # 计算均值
        mean_value = np.mean(group)
        # 绘制均值点
        plt.plot([i + 0.8, i + 1.2], [mean_value, mean_value],  color='orange', linewidth=3.5, zorder=5)  # 通过调整i + 0.8和i + 1.
    # 设置横坐标标签
    plt.ylim(-1, 1)
    plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels, fontsize=12, rotation=90)
    plt.yticks(fontsize=15)
    # 显示图形
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95)  # 通过调整left和right参数来减少图形与左右边缘之间的间距
    plt.show()
def compute_corr(x_np, y_np):
    # 将NumPy数组转换为PyTorch张量
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).float()
    # 计算均值
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    # 计算协方差
    covariance = torch.mean((x - mean_x) * (y - mean_y))
    # 计算标准差
    std_x = torch.sqrt(torch.mean((x - mean_x) ** 2))
    std_y = torch.sqrt(torch.mean((y - mean_y) ** 2))
    # 计算相关系数
    corr = covariance / (std_x * std_y)
    return corr.item()

def model_training_data_load(dir_, model_name, sub_idx, select_sequence_set, select_sequence):
    ""
    "加载fmri数据"
    save_dir = dir_ + 'Dataset_fMRI/Stan_fMRI_Sub' + str(sub_idx) + '.h5'
    h5_file = h5py.File(save_dir, "r")
    fmri_data = h5_file["data"][:]
    sequence = np.arange(fmri_data.shape[1])
    num_chunks = len(select_sequence_set)
    chunks = np.array_split(sequence, num_chunks)
    seq_array = chunks[select_sequence]
    fmri_data = fmri_data[:, seq_array]

    """
    加载integration
    """
    "image2text"
    save_dir = (dir_ + 'Dataset_ANN/' + model_name + '/Stan_InteImage2text_Sub' + str(sub_idx) + '.h5')
    h5_file = h5py.File(save_dir, "r")
    Image2text = h5_file["data"][:]
    "text2image"
    save_dir = (dir_ + 'Dataset_ANN/' + model_name + '/Stan_InteText2image_Sub' + str(sub_idx) + '.h5')
    h5_file = h5py.File(save_dir, "r")
    Text2image = h5_file["data"][:]
    "image2image"
    save_dir = (dir_ + 'Dataset_ANN/' + model_name + '/Stan_InteImage2image_Sub' + str(sub_idx) + '.h5')
    h5_file = h5py.File(save_dir, "r")
    Image2image = h5_file["data"][:]
    "text2text"
    save_dir = (dir_ + 'Dataset_ANN/' + model_name + '/Stan_InteText2text_Sub' + str(sub_idx) + '.h5')
    h5_file = h5py.File(save_dir, "r")
    Text2text = h5_file["data"][:]

    return fmri_data, Image2text, Text2image, Image2image, Text2text, seq_array


def generate_trends_list(repeat_num=5):
    """
    生成所有趋势的数值序列。

    参数:
        repeat_num (int): 趋势的长度（变化点的数量）。

    返回:
        trends_list (np.ndarray): 所有趋势的数值序列，形状为 (3^repeat_num, repeat_num + 1)。
    """
    # 定义可能的变化方向
    directions = ['+', '-', '=']

    # 生成所有趋势组合
    all_trends = list(product(directions, repeat=repeat_num))

    # 定义排序权重函数
    def calculate_weight(trend):
        w_plus = trend.count('+')
        w_minus = trend.count('-')
        return (w_minus - w_plus) / repeat_num + 1

    # 对趋势进行排序
    all_trends = sorted(all_trends, key=calculate_weight)

    # 生成趋势的数值序列
    trends_list = []
    for trend in all_trends:
        trend_values = []
        ini_values = 1
        trend_values.append(ini_values)
        for char in trend:
            if char == '+':
                ini_values = ini_values + 1
            elif char == '-':
                ini_values = ini_values - 1
            trend_values.append(ini_values)
        trends_list.append(np.stack(trend_values))

    # 将列表转换为 numpy 数组
    trends_list = np.stack(trends_list)
    return trends_list

# 计算移动平均
def moving_average(data, window_size):
    n = data.shape[0]  # 输入序列长度
    m = n // window_size  # 输出序列长度
    result = np.zeros((m, data.shape[1]))  # 初始化输出数组

    for i in range(m):
        start = i * window_size
        end = start + window_size
        result[i] = np.mean(data[start:end], axis=0)  # 计算窗口内的平均值

    return result


def plot_stacked_proportions(data, x_labels, title_name, category_labels=None):
    """
    绘制堆叠图，展示每个样本中各个类别的占比。

    参数:
    - data: N*M 的矩阵，N 是样本数，M 是特征维度（每个样本中不同类别的占比）。
    - x_labels: 列表，X 轴的标签（如 ['V1', 'V2', 'V3', ...]）。
    - category_labels: 可选，M 个类别的标签（如 ['1', '2', '3', '4', '5']），用于图例显示。
    """
    # 样本数 N 和特征维度 M
    N, M = data.shape

    # 如果没有提供 category_labels，则默认使用数字 1 到 M
    if category_labels is None:
        category_labels = [str(i+1) for i in range(M)]

    # 检查 X 轴标签的数量是否与样本数一致
    if len(x_labels) != N:
        raise ValueError("X 轴标签的数量必须与样本数一致。")

    # 初始化堆叠图的底部
    bottoms = np.zeros(N)

    # 绘制堆叠图
    plt.figure(figsize=(N*.2, 3))
    for i in range(M):
        plt.bar(x_labels, data[:, i], bottom=bottoms, label=category_labels[i])
        bottoms += data[:, i]  # 更新底部位置

    # 添加标签和标题
    plt.ylabel('Proportion')
    plt.title(title_name)
    plt.xticks(rotation=90)  # 旋转 X 轴标签，避免重叠
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # 显示图形
    plt.tight_layout()  # 自动调整布局，避免标签被截断
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_mean_std(mean_data, std_data, x_labels, line_labels=None):
    """
    绘制均值和标准差的折线图。

    参数:
    - mean_data: 均值数据，形状为 (N, M)，N 是样本数，M 是特征维度。
    - std_data: 标准差数据，形状为 (N, M)，与 mean_data 形状一致。
    - x_labels: X 轴的标签（如 ['V1', 'V2', ..., 'DVT']）。
    - line_labels: 可选，M 条线的标签（如 ['Method1', 'Method2', ..., 'Method6']），用于图例显示。
    """
    # 样本数 N 和特征维度 M
    N, M = mean_data.shape

    # 如果没有提供 line_labels，则默认使用数字 1 到 M
    if line_labels is None:
        line_labels = [f'Method {i+1}' for i in range(M)]

    # 检查 X 轴标签的数量是否与样本数一致
    if len(x_labels) != N:
        raise ValueError("X 轴标签的数量必须与样本数一致。")

    # 创建图形
    plt.figure(figsize=(N*.15, 2.5))

    # 绘制均值折线图
    for i in range(M):
        plt.plot(x_labels, mean_data[:, i], label=line_labels[i], marker='_', markersize=2)

    # 添加标准差作为误差线
    for i in range(M):
        plt.fill_between(x_labels,
                        mean_data[:, i] - .5*std_data[:, i],
                        mean_data[:, i] + .5*std_data[:, i],
                        alpha=0.1)

    # 添加标签和标题
    # 去掉左、右边线，加粗下边线
    ax = plt.gca()  # 获取当前坐标轴
    ax.spines['left'].set_visible(False)  # 去掉左边线
    ax.spines['right'].set_visible(False)  # 去掉右边线
    ax.spines['bottom'].set_linewidth(2)  # 加粗下边线
    ax.spines['top'].set_visible(False)  # 去掉上边线
    plt.xticks(rotation=90)  # 旋转 X 轴标签，避免重叠
    # plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.ylim(0, 0.55)

    # 显示图形
    plt.tight_layout()  # 自动调整布局，避免标签被截断
    plt.show()

def plot_color_labels(labels):
    """
    绘制颜色和标签的水平图例。

    参数:
    - labels: 标签列表，长度为 M（例如 ['Method1', 'Method2', ..., 'Method6']）。
    """
    # 获取默认的颜色循环
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']  # 提取颜色列表

    # 确保颜色数量足够
    if len(labels) > len(colors):
        raise ValueError("标签数量超过了默认颜色列表的长度。")

    # 创建图形
    fig, ax = plt.subplots(figsize=(len(labels) * 1.5, 1))  # 宽度根据标签数量调整
    ax.set_xlim(0, len(labels))
    ax.set_ylim(0, 1)
    ax.axis('off')  # 不显示坐标轴

    # 绘制颜色块和标签
    for i, label in enumerate(labels):
        # 绘制颜色块
        rect = patches.Rectangle((i, 0), 0.8, 0.8, facecolor=colors[i], edgecolor='black')
        ax.add_patch(rect)

        # 添加标签
        ax.text(i + 0.4, 0.9, f'L{label}', ha='center', va='bottom', fontsize=12)

    # 显示图形
    plt.tight_layout()
    plt.show()

def compute_interpolation_matrix(A):
    """
    计算插值矩阵：后一列减去前一列，如果不为0则返回1。

    参数:
    - A: 输入矩阵，形状为 (180, 6)。

    返回:
    - interpolation_matrix: 插值矩阵，形状为 (180, 5)。
    """
    # 初始化插值矩阵
    interpolation_matrix = np.zeros((A.shape[0], A.shape[1] - 1), dtype=int)

    # 计算差值并判断是否为零
    for i in range(A.shape[1] - 1):
        diff = A[:, i + 1] - A[:, i]  # 后一列减去前一列
        interpolation_matrix[:, i] = (diff != 0).astype(int)  # 如果不为0则返回1

    return interpolation_matrix


def mark_transitions(best_type_R2_roi_layer_list_type):
    # 获取数组的形状
    rows, cols = best_type_R2_roi_layer_list_type.shape

    # 创建一个新的数组来存储转换标记
    transition_matrix = np.zeros((rows, cols - 1), dtype=int)  # 使用 int 类型

    # 遍历每一行
    for i in range(rows):
        # 遍历每一列，直到倒数第二列
        for j in range(cols - 1):
            current = best_type_R2_roi_layer_list_type[i, j]
            next_ = best_type_R2_roi_layer_list_type[i, j + 1]

            # 判断转换类型并标记
            if current == 0 and next_ == 0:
                transition_matrix[i, j] = 1  # Image2image→Image2image
            elif current == 1 and next_ == 1:
                transition_matrix[i, j] = 2  # Image2text→Image2text
            elif current == 2 and next_ == 2:
                transition_matrix[i, j] = 3  # Text2text→Text2text
            elif current == 3 and next_ == 3:
                transition_matrix[i, j] = 4  # Text2image→Text2image
            elif current == 0 and next_ == 1:
                transition_matrix[i, j] = 5  # Image2image→Image2text
            elif current == 0 and next_ == 2:
                transition_matrix[i, j] = 6  # Image2image→Text2text
            elif current == 0 and next_ == 3:
                transition_matrix[i, j] = 7  # Image2image→Text2image
            elif current == 1 and next_ == 0:
                transition_matrix[i, j] = 8  # Image2text→Image2image
            elif current == 1 and next_ == 2:
                transition_matrix[i, j] = 9  # Image2text→Text2text
            elif current == 1 and next_ == 3:
                transition_matrix[i, j] = 10  # Image2text→Text2image
            elif current == 2 and next_ == 0:
                transition_matrix[i, j] = 11  # Text2text→Image2image
            elif current == 2 and next_ == 1:
                transition_matrix[i, j] = 12  # Text2text→Image2text
            elif current == 2 and next_ == 3:
                transition_matrix[i, j] = 13  # Text2text→Text2image
            elif current == 3 and next_ == 0:
                transition_matrix[i, j] = 14  # Text2image→Image2image
            elif current == 3 and next_ == 1:
                transition_matrix[i, j] = 15  # Text2image→Image2text
            elif current == 3 and next_ == 2:
                transition_matrix[i, j] = 16  # Text2image→Text2text
            else:
                transition_matrix[i, j] = 0  # 其他情况标记为 0

    return transition_matrix