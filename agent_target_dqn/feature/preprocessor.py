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

# 即时奖励涉及很多状态先后变化，写在大类里不好读，故封装到这
class ImRewardAttr:
    def __init__(self):
        # 1. 宝箱拾取的即时奖励
        self.last_treasure_collect_num = 0
        self.treasure_collect_num = 0

        # 2. buff拾取的即时奖励
        self.last_buff_collect_num = 0
        self.buff_collect_num = 0
        self.buff_cnt = 0

        # 3. 闪现的即时奖励
        self.if_flash = False

    def update(self, obs, last_action):
        self.last_treasure_collect_num = self.treasure_collect_num
        self.treasure_collect_num = obs['score_info']['treasure_collected_count']
        self.if_collect_ts = self.treasure_collect_num - self.last_treasure_collect_num

        self.last_buff_collect_num = self.buff_collect_num
        self.buff_collect_num = obs['score_info']['buff_count']
        self.if_collect_buff = self.buff_collect_num - self.last_buff_collect_num
        if self.if_collect_buff:
            self.buff_cnt += 1
        
        self.if_flash = last_action // 8
    
    def get_imReward(self):
        treasure_reward = 5 * self.if_collect_ts
        buff_reward = 5 * (0.5 ** self.buff_cnt) * self.if_collect_buff
        flash_reward = -0.5 * self.if_flash
        return treasure_reward + buff_reward + flash_reward


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
        self.target_pos = (0, 0)               # 目标坐标

        # 视野域信息 ---onehot coding
        self.treasure_flag = None
        self.end_flag = None
        self.obstacle_flag = None
        self.buff_flag = None

        self.organ_status = [0] * 15           # 组件状态

        # 即时奖励属性
        self.imRewardAttr = ImRewardAttr()

    def _get_pos_feature(self, found, cur_pos, target_pos):      # 输入AB坐标，生成对应向量AB的特征向量(tool fuction)
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

        # 选择目标位置
        organs = obs['frame_state']['organs']
        min_dist_sqr = (self.end_pos[0] - self.cur_pos[0])**2 + (self.end_pos[1] - self.cur_pos[1])**2       # type: ignore
        self.target_pos = self.end_pos
        targetIsEnd = True
        for organ in organs:
            subtype = organ['sub_type']
            if organ['status'] == 1 and (subtype == 1 or subtype == 2):    # 若组件可获取且是宝箱或buff，计算L2距离的平方
                dist_sqr = (organ['pos']['x'] - self.cur_pos[0])**2 + (organ['pos']['z'] - self.cur_pos[1])**2
                if dist_sqr < min_dist_sqr:
                    min_dist_sqr = dist_sqr
                    self.target_pos = (organ['pos']['x'], organ['pos']['z'])
                    targetIsEnd = False

        # 组件状态信息
        for organ in organs:
            if organ['sub_type'] == 1 or organ['sub_type'] == 2: self.organ_status[organ['config_id']] = organ['status']
            elif organ['sub_type'] == 4: self.organ_status[14] = organ['status']

        # 自己看
        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)     # 有关这部分可以进行修改优化，训练验证能跑后再进行优化
        # 目标点的特征向量(这里用了个三元操作符，后续验证能跑会修改简洁)
        self.feature_target_pos = self._get_pos_feature(self.is_end_pos_found if targetIsEnd else 1, self.cur_pos, self.target_pos)

        # History position feature
        # 历史位置特征
        # 记录前面10步得到一个位移特征向量，从探索的角度来看我们应该鼓励让这个向量dist大一点，可以在这调参优化
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])

        self.move_usable = True          # 这个属性放在这初始化应该是为了强制先观测处理，后求合法action
        self.last_action = last_action

        # 闪现状态
        if hero['talent']['status'] == 0:
            self.is_flashed = False
        else: self.is_flashed = True
        
        # 视野域信息维护 + 找到与障碍物最短距离
        min_obstacle_dist = 6 * 1.41
        map_info = obs['map_info']
        treasure_map = np.zeros((11, 11), dtype=np.float32)
        end_map = np.zeros((11, 11), dtype=np.float32)
        obstacle_map = np.zeros((11, 11), dtype=np.float32)
        buff_map = np.zeros((11, 11), dtype=np.float32)
        for r, row_data in enumerate(map_info):
            for c, value in enumerate(row_data['values']):
                if value == 0:
                    obstacle_map[r, c] = 1
                    tempdist = ((r-5)**2+(c-5)**2)**(1/2)
                    if tempdist < min_obstacle_dist:
                        min_obstacle_dist = tempdist
                elif value == 4: treasure_map[r, c] = 1
                elif value == 6: buff_map[r, c] = 1
                elif value == 3: end_map[r, c] = 1
        self.treasure_flag = treasure_map.flatten()
        self.end_flag = end_map.flatten()
        self.obstacle_flag = obstacle_map.flatten()
        self.buff_flag = buff_map.flatten()
        # 与障碍物的最短L2距离标准化
        self.obstacle_dist = norm(min_obstacle_dist, 1.41 * 128)

        # 即时奖励相关属性维护
        self.imRewardAttr.update(obs, last_action)

    def process(self, frame_state, last_action):          # 外层调用
        self.pb2struct(frame_state, last_action)      # 用原始观测更新属性

        # Legal action
        # 合法动作
        legal_action = self.get_legal_action()        # 用更新后的属性获取合法动作

        # 当前位置的One-hot编码
        curposx_onehot, curposz_onehot = self.get_cur_pos_onehot()

        # Feature
        # ***更新后的属性在这直接打包成特征
        # 特征
        feature = np.concatenate([
            self.cur_pos_norm,
            curposx_onehot,
            curposz_onehot,
            self.organ_status,
            self.feature_target_pos,              # baseline以end pos为目标，现在改成target pos
            self.feature_history_pos,
            legal_action,
            self.treasure_flag,
            self.end_flag,
            self.obstacle_flag,
            self.buff_flag
            ])

        # 特征、合法动作、奖励函数打包返回。***奖励函数在这被调用
        return (
            feature,
            legal_action,
            reward_process(
                self.feature_target_pos[-1],
                self.feature_history_pos[-1],
                self.obstacle_dist,
                self.imRewardAttr.get_imReward()
                ),
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

    def get_cur_pos_onehot(self):
        x = [0] * 64
        z = [0] * 64
        x[self.cur_pos[0]] = 1
        z[self.cur_pos[1]] = 1
        return(x, z)