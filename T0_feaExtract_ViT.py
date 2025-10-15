import copy
import argparse
import configparser
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import h5py
from transformers import ViTImageProcessor, ViTForImageClassification

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def extract_block_features(model_name="vit-base-patch16-224"):
    # 加载ViT模型和图像处理器
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name).to(device)
    model.eval()  # 设置模型为评估模式

    # 用于存储每一层输出的字典
    block_outputs = {}

    # 定义钩子函数，用于捕获每一层的输出
    def get_hook(layer_idx):
        def hook(module, input, output):
            # 提取 `[CLS]` token 的表示并移除多余的维度
            if layer_idx == 0:
                block_outputs[0] = input[0][0, 0, :].cpu().data.numpy()
            block_outputs[layer_idx+1] = output[0][0, 0, :].cpu().data.numpy()

        return hook

    # 为每个变换块注册钩子
    transformer_layers = model.vit.encoder.layer
    for i, layer in enumerate(transformer_layers):
        layer.register_forward_hook(get_hook(i))

    all_features = []

    # 读取图像数据
    NSD_COCO_img = h5py.File('/data2/shigaosheng/data_model/nsd/nsd_stimuli.hdf5', 'r')
    NSD_COCO_img = NSD_COCO_img['imgBrick']

    for p in tqdm(NSD_COCO_img):
        image = Image.fromarray(p)  # 将图像数组转换为PIL图像
        inputs = processor(images=image, return_tensors="pt").to(device)  # 处理图像并转移到设备

        # 每处理一张新图像时，清空 block_outputs
        block_outputs.clear()

        # 执行前向传播并捕获每层的输出
        with torch.no_grad():
            final_layer_output = model(**inputs).logits  # 获取输出
        all_features.append(np.vstack(list(block_outputs.values())))

    all_features = np.array(all_features)
    return all_features

if __name__ == "__main__":
    feature_output_dir = os.path.dirname(os.path.abspath(__file__)) + '/Dataset_ANN/'

    # 提取每一层的特征
    visual_vit_feat = extract_block_features()

    # 将特征保存为 numpy 文件
    np.save(f"{feature_output_dir}/vit_visual_block_outputs_stack.npy", visual_vit_feat)
