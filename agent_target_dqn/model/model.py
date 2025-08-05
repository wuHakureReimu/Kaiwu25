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

#蒋的网络
class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        self.feature_len = Config.DIM_OF_OBSERVATION

        # 公共特征层
        self.feature_layer = MLP([self.feature_len, 256, 128], "feature")

        # Value 分支
        self.value_layer = MLP([128, 64, 1], "value")

        # Advantage 分支
        self.advantage_layer = MLP([128, 64, action_shape], "advantage")

    def forward(self, feature):
        x = self.feature_layer(feature)
        value = self.value_layer(x)  # [batch, 1]
        advantage = self.advantage_layer(x)  # [batch, action_shape]
        # dueling Q 组合
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

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

