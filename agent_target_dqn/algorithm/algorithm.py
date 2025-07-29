#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import os
import numpy as np
import torch
from copy import deepcopy
from agent_target_dqn.model.model import Model
from agent_target_dqn.conf.conf import Config
from agent_target_dqn.feature.definition import ActData


class Algorithm:
    def __init__(self, device, logger, monitor):
        # 维度配置
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT      # 移动8维 + 闪现8维
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.talent_direction = Config.DIM_OF_TALENT
        self.obs_shape = Config.DIM_OF_OBSERVATION

        self.epsilon_max = Config.EPSILON_MAX
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self._gamma = Config.GAMMA
        self.lr = Config.START_LR
        self.device = device
        self.model = Model(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
        )
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.target_model = deepcopy(self.model)
        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0
        self.logger = logger
        self.monitor = monitor

    def learn(self, list_sample_data):

        t_data = list_sample_data
        batch = len(t_data)

        batch_feature_vec = [frame.obs for frame in t_data]
        batch_action = torch.LongTensor(np.array([int(frame.act) for frame in t_data])).view(-1, 1).to(self.device)
        _batch_obs_legal = torch.stack([frame._obs_legal for frame in t_data]).bool().to(self.device)
        _batch_feature_vec = [frame._obs for frame in t_data]

        rew = torch.tensor(
            [frame.rew.cpu().item() if isinstance(frame.rew, torch.Tensor) else frame.rew for frame in t_data],
            device=self.device,
        )
        not_done = torch.tensor(
            [
                0 if (frame.done.cpu().item() if isinstance(frame.done, torch.Tensor) else frame.done) == 1 else 1
                for frame in t_data
            ],
            device=self.device,
        )
        batch_feature = self.__convert_to_tensor(batch_feature_vec)
        _batch_feature = self.__convert_to_tensor(_batch_feature_vec)

        model = getattr(self, "target_model")
        model.eval()
        with torch.no_grad():
            q = model(_batch_feature)
            q = q.masked_fill(~_batch_obs_legal, float(torch.min(q)))
            q_max = q.max(dim=1).values.detach()

        target_q = rew + self._gamma * q_max * not_done

        self.optim.zero_grad()

        model = getattr(self, "model")
        model.train()
        logits = model(batch_feature)

        loss = torch.square(target_q - logits.gather(1, batch_action).view(-1)).mean()
        loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optim.step()

        self.train_step += 1

        # Update the target network
        # 更新target网络
        if self.train_step % self.target_update_freq == 0:
            self.update_target_q()

        value_loss = loss.detach().item()
        q_value = target_q.mean().detach().item()
        reward = rew.mean().detach().item()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "q_value": q_value,
                "reward": reward,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

    def __convert_to_tensor(self, data):
        # Please check the data type carefully and make sure it is float32
        # 请仔细检查数据类型，确保是 float32
        if isinstance(data, list):
            if isinstance(data[0], torch.Tensor):
                processed = torch.stack(data, dim=0).to(self.device).float()
            elif isinstance(data[0], np.ndarray):
                processed = torch.from_numpy(np.stack(data, axis=0)).to(self.device).float()
            else:
                processed = torch.tensor(np.array(data), dtype=torch.float32).to(self.device)
        elif isinstance(data, np.ndarray):
            processed = torch.from_numpy(data.astype(np.float32)).to(self.device)
        elif torch.is_tensor(data):
            processed = data.to(self.device).float()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        return processed

    def predict_detail(self, list_obs_data, exploit_flag=False):
        batch = len(list_obs_data)

        feature_vec = [obs_data.feature for obs_data in list_obs_data]
        legal_act = [obs_data.legal_act for obs_data in list_obs_data]
        legal_act = torch.tensor(np.array(legal_act)).bool().to(self.device)

        model = self.model
        model.eval()

        # Exploration factor,
        # we want epsilon to decrease as the number of prediction steps increases, until it reaches 0.1
        # 探索因子, 我们希望epsilon随着预测步数越来越小，直到0.1为止
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(
            -self.epsilon_decay * self.predict_count
        )

        with torch.no_grad():
            # epsilon greedy
            # epsilon 贪婪
            if not exploit_flag and np.random.rand(1) < self.epsilon:
                random_action = np.random.rand(batch, self.act_shape)
                random_action = torch.tensor(random_action, dtype=torch.float32).to(self.device)
                random_action = random_action.masked_fill(~legal_act, 0)
                act = random_action.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                feature = self.__convert_to_tensor(feature_vec)
                logits = model(feature)
                logits = logits.masked_fill(~legal_act, float(torch.min(logits)))
                act = logits.argmax(dim=1).cpu().view(-1, 1).tolist()

        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        self.predict_count += 1
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())
