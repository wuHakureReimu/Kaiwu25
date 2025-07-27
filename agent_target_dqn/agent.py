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
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    reset_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from agent_target_dqn.algorithm.algorithm import Algorithm
from agent_target_dqn.feature.definition import ObsData, ActData
from agent_target_dqn.feature.preprocessor import Preprocessor
from arena_proto.back_to_the_realm_v2.custom_pb2 import (
    RelativeDirection,
)


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.agent_type = agent_type
        self.logger = logger
        self.algorithm = Algorithm(device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.win_history = []

    def update_win_rate(self, is_win):
        self.win_history.append(is_win)
        if len(self.win_history) > 100:
            self.win_history.pop(0)
        return sum(self.win_history) / len(self.win_history) if len(self.win_history) > 10 else 0

    def reset(self):
        self.preprocessor.reset()
        self.last_action = -1

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.algorithm.predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, observation):
        obs_data, _ = self.observation_process(observation["obs"], observation["extra_info"])
        act_data = self.algorithm.predict_detail([obs_data], exploit_flag=True)
        act = self.action_process(act_data[0])
        return act

    @learn_wrapper
    def learn(self, list_sample_data):
        self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.algorithm.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.algorithm.model.load_state_dict(torch.load(model_file_path, map_location=self.algorithm.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def observation_process(self, obs, extra_info):
        (feature_vec, legal_action, reward_list) = self.preprocessor.process([obs, extra_info], self.last_action)
        return ObsData(feature=feature_vec, legal_act=legal_action), reward_list

    def action_process(self, act_data):
        result = act_data.move_dir
        result += act_data.use_talent * 8
        self.last_action = result
        return result
