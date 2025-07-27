#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np
import math
from agent_target_dqn.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process


def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 16      # 8方向移动+8方向闪现
        self.reset()

    def reset(self):
        self.step_no = 0                       # 帧号
        self.cur_pos = (0, 0)                  # 当前坐标
        self.cur_pos_norm = np.array((0, 0))   # 标准化后的当前坐标
        self.end_pos = None                    # 终点坐标 tuple
        self.is_end_pos_found = False          # 中间变量，参与终点坐标估算以及相应特征向量生成
        self.history_pos = []                  # 存储前10步的坐标 [tuples]
        self.bad_move_ids = set()              # 中间变量，用来判断非法动作
        self.is_flashed = True                 # 闪现状态 True-available

    def _get_pos_feature(self, found, cur_pos, target_pos):      # 输入AB坐标，生成对应向量AB的特征向量
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))  # 对应向量
        dist = np.linalg.norm(relative_pos)                               # L2范数
        target_pos_norm = norm(target_pos, 128, -128)                     # B点坐标标准化
        feature = np.array(                                      # ***important---feature structure
            (
                found,           # if found
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),        # direction
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),        # direction
                target_pos_norm[0],                                    # B点坐标标准化
                target_pos_norm[1],
                norm(dist, 1.41 * 128),                                # L2范数标准化
            ),
        )
        return feature

    def pb2struct(self, frame_state, last_action):     # 每次观测后调用，这里维护所有属性
        obs, _ = frame_state
        self.step_no = obs["frame_state"]["step_no"]

        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        # History position
        # 历史位置
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

        # End position
        # 终点位置
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] == 4:
                end_pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
                end_pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
                if organ["status"] != -1:
                    self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                    self.is_end_pos_found = True
                # if end_pos is not found, use relative position to predict end_pos
                # 如果终点位置未找到，使用相对位置预测终点位置
                elif (not self.is_end_pos_found) and (
                    self.end_pos is None
                    or self.step_no % 100 == 0
                    or self.end_pos_dir != end_pos_dir
                    or self.end_pos_dis != end_pos_dis
                ):
                    distance = end_pos_dis * 20
                    theta = DirectionAngles[end_pos_dir]
                    delta_x = distance * math.cos(math.radians(theta))
                    delta_z = distance * math.sin(math.radians(theta))

                    self.end_pos = (
                        max(0, min(128, round(self.cur_pos[0] + delta_x))),
                        max(0, min(128, round(self.cur_pos[1] + delta_z))),
                    )

                    self.end_pos_dir = end_pos_dir
                    self.end_pos_dis = end_pos_dis

        # 自己看
        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)

        # History position feature
        # 历史位置特征
        # 记录前面10步得到一个位移特征向量，从探索的角度来看我们应该鼓励让这个向量dist大一点，可以在这调参优化
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])

        self.move_usable = True          # 这个属性放在这初始化应该是为了强制先观测处理，后求合法action
        self.last_action = last_action

        # 闪现状态
        if hero['talent']['status'] == 0:
            self.is_flashed = False

    def process(self, frame_state, last_action):          # 外层调用
        self.pb2struct(frame_state, last_action)      # 用原始观测更新属性

        # Legal action
        # 合法动作
        legal_action = self.get_legal_action()        # 用更新后的属性获取合法动作

        # Feature
        # ***更新后的属性在这直接打包成特征
        # 特征
        feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos, self.feature_history_pos, legal_action])

        # 特征、合法动作、奖励函数打包返回。***奖励函数在这被调用
        return (
            feature,
            legal_action,
            reward_process(self.feature_end_pos[-1], self.feature_history_pos[-1]),     # 传入两个距离
        )

    def get_legal_action(self):
        '''
        return format is like
        [True, True, ..., False, True, ...]
        '''
        # if last_action is move and current position is the same as last position, add this action to bad_move_ids
        # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中
        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
        else:
            self.bad_move_ids = set()        # 成功移动时重置

        legal_action = [self.move_usable] * self.move_action_num
        # 移动合法性更新
        for move_id in self.bad_move_ids:
            legal_action[move_id] = False

        # 闪现合法性判断&更新
        if self.is_flashed: legal_action[8:] = [True] * 8
        else: legal_action[8:] = [False] * 8

        if self.move_usable not in legal_action:
            self.bad_move_ids = set()
            return [self.move_usable] * self.move_action_num

        return legal_action
