#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration, including dimension settings, algorithm parameter settings.
# The last few configurations in the file are for the Kaiwu platform to use and should not be changed.
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:

    # features
    # 注:这里的feature维度指preprocessor的process里面的feature维度
    
    #蒋的特征
    FEATURES = [
        2,             # cur pos norm
        5,             # target pos feature
        50,             # history pos (-10) feature
        16,            # legal actions 8 + 8
        49,
    ]
    #蒋的特征
    FEATURE_SPLIT_SHAPE = FEATURES

    # Size of observation
    # observation的维度
    DIM_OF_OBSERVATION = sum(FEATURES)

    # Dimension of movement action direction
    # 移动动作方向的维度
    DIM_OF_ACTION_DIRECTION = 8

    # Dimension of flash action direction
    # 闪现动作方向的维度
    DIM_OF_TALENT = 8

    # Input dimension of reverb sample on learner. Note that different algorithms have different dimensions.
    # **Note**, this item must be configured correctly and should be aligned with the NumpyData2SampleData function data in definition.py
    # Otherwise the sample dimension error may be reported
    # learner上reverb样本的输入维度
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = 2 * (DIM_OF_OBSERVATION + DIM_OF_ACTION_DIRECTION) + 4

    # Update frequency of target network
    # target网络的更新频率
    TARGET_UPDATE_FREQ = 200

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.99    #0.9

    # epsilon
    EPSILON_MIN = 0.15   #0.1
    EPSILON_MAX = 0.80   #1.0
    EPSILON_DECAY = 8e-7    #1e-6

    # Initial learning rate
    # 初始的学习率
    START_LR = 0.0003  #1e-4
