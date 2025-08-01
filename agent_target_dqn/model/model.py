#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import List
from agent_target_dqn.conf.conf import Config

import sys
import os

if os.path.basename(sys.argv[0]) == "learner.py":
    import torch

    torch.set_num_interop_threads(2)
    torch.set_num_threads(2)
else:
    import torch

    torch.set_num_interop_threads(4)
    torch.set_num_threads(4)


class OutputInputAttention(nn.Module):
    """注意力模块：Query来自输出，Key和Value来自输入，内部使用不影响外部接口"""
    def __init__(self, input_dim, output_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        assert output_dim % num_heads == 0, "输出维度必须能被注意力头数量整除"
        self.head_dim = output_dim // num_heads
        
        # 注意力投影层
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(input_dim, output_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_features, output_features):
        batch_size = input_features.size(0)
        
        # 增加序列维度
        input_features = input_features.unsqueeze(1)
        output_features = output_features.unsqueeze(1)
        
        # 生成Q, K, V
        q = self.q_proj(output_features)
        k = self.k_proj(input_features)
        v = self.v_proj(input_features)
        
        # 拆分到多个注意力头
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力权重
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, v)
        
        # 拼接多头结果
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, 1, self.output_dim
        )
        
        # 输出投影和残差连接
        output = self.out_proj(attn_output)
        output = output + output_features  # 残差连接保留原始信息
        return output.squeeze(1)  # 仅返回处理后的输出，不暴露权重


class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False, num_heads=8):
        super().__init__()
        self.feature_len = Config.DIM_OF_OBSERVATION
        self.action_shape = action_shape
        
        # 原有特征提取层
        self.feature_layer = nn.Sequential(
            make_fc_layer(self.feature_len, 256),
            nn.ReLU(),
            make_fc_layer(256, 256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU()
        )
        
        # 原有价值和优势流
        self.value_layer = nn.Sequential(
            make_fc_layer(128, 64),
            nn.ReLU(),
            make_fc_layer(64, 1)
        )
        self.advantage_layer = nn.Sequential(
            make_fc_layer(128, 64),
            nn.ReLU(),
            make_fc_layer(64, action_shape)
        )
        
        # 集成注意力模块，但不改变外部接口
        self.attention = OutputInputAttention(
            input_dim=self.feature_len,
            output_dim=action_shape,
            num_heads=num_heads
        )

    def forward(self, feature):
        # 保存原始输入用于注意力计算
        original_feature = feature
        
        # 原有DQN计算流程
        x = self.feature_layer(feature)
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # 应用注意力机制优化Q值，但不改变输出格式
        q_attended = self.attention(original_feature, q)
        
        # 保持原有输出格式，确保与外部框架兼容
        return q_attended


def make_fc_layer(in_features: int, out_features: int):
    # Wrapper function to create and initialize a linear layer
    # 创建并初始化一个线性层
    fc_layer = nn.Linear(in_features, out_features)

    # initialize weight and bias
    # 初始化权重及偏移量
    nn.init.orthogonal(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)

    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        # Create a MLP object
        # 创建一个 MLP 对象
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            # no relu for the last fc layer of the mlp unless required
            # 除非有需要，否则 mlp 的最后一个 fc 层不使用 relu
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
