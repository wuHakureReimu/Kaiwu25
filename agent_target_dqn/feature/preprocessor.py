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
import time
from agent_target_dqn.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process


def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


#蒋的代码
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
        self.obstacle_flag = None
        self.last_pos=(0,0)
        #下为v18
        self.train_map=np.full((128,128),-1,dtype=int)  #train_map方向适应obs方向
        self.critical_point={'end':[],'trea':[],'buff':[]}

        # v19 by Jiang
        # 为了防止寻路过程中不小心走到终点，我需要设计一个全局特征表征宝箱收集状态，以及终点
        # 在宝箱收集数量比较低的时候，在离终点比较近的范围内（初步考虑1格）会得到巨额惩罚
        self.treasure_num = 0      # 这个特征我决定归一化0-1处理，便于训练

    def _get_pos_feature(self, found, cur_pos, target_pos):      # 输入AB坐标，生成对应向量AB的特征向量(tool fuction)
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))  # 对应向量
        dist = np.linalg.norm(relative_pos)                               # L2范数
        target_pos_norm = norm(target_pos, 128, -128)                     # B点坐标标准化
        feature = np.array(                                      # ***important---feature structure
            (
                #found,           # if found
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
        if(len(self.history_pos)>=1):self.last_pos=self.history_pos[-1]
        else:self.last_pos=self.cur_pos
        self.history_pos.append(self.cur_pos)
        #v17增加
        while(len(self.history_pos) < 10):
            self.history_pos.append(self.cur_pos)
        #v17增加
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
        self.treasure_num = 0
        organs = obs['frame_state']['organs']
        end_min_dist_sqr = (self.end_pos[0] - self.cur_pos[0])**2 + (self.end_pos[1] - self.cur_pos[1])**2     
        self.target_pos = self.end_pos
        targetIsEnd = True
        if(self.step_no<=1500): #大于1500步后,直接寻找终点(当max_step改变时需要调整)
            min_dist_sqr=10000   #可能大于end_min_dist_sqr,但宝箱的优先级较高
            for organ in organs:
                subtype = organ['sub_type']
                if organ['status'] == 1 and (subtype == 1 or subtype == 2):    # 若组件可获取且是宝箱或buff，计算L2距离的平方
                    dist_sqr = (organ['pos']['x'] - self.cur_pos[0])**2 + (organ['pos']['z'] - self.cur_pos[1])**2
                    if((subtype == 1 and dist_sqr < min_dist_sqr) or (subtype == 2 and dist_sqr < min_dist_sqr and dist_sqr < end_min_dist_sqr)):
                        min_dist_sqr = dist_sqr
                        self.target_pos = (organ['pos']['x'], organ['pos']['z'])
                        targetIsEnd = False
                
                # 这里维护已收集宝箱数的特征
                if organ['status'] == 0 and subtype == 1:
                    self.treasure_num += 1/8
        else:
            self.treasure_num = 1   # 1500步以后视为宝箱全部收集完毕

        # 自己看
        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        # 目标点的特征向量（消融实验发现found特征不重要）
        self.feature_target_pos = self._get_pos_feature(1, self.cur_pos, self.target_pos)

        # v19 by Jiang
        # 终点特征
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)

        # 历史路径特征
        # v17
        self.feature_history_pos=[]
        self.whole_history_dist=0
        for hpos in self.history_pos:
            one_hpos_feature=self._get_pos_feature(1, self.cur_pos, hpos)
            self.whole_history_dist+=one_hpos_feature[-1]
            self.feature_history_pos.append(one_hpos_feature)
        self.feature_history_pos=np.concatenate(self.feature_history_pos)
        #v17

        self.move_usable = True          # 这个属性放在这初始化应该是为了强制先观测处理，后求合法action
        self.last_action = last_action

        # 闪现状态
        if hero['talent']['status'] == 0:self.is_flashed = False
        else:self.is_flashed = True
        
        # 视野域信息维护
        map_info = obs['map_info']
        #v18
        for x in range(11):
            for z in range(11):
                if(self.train_map[self.cur_pos[0]-5+x][self.cur_pos[1]-5+z]!=0):  #已经判为0的地方不改
                    self.train_map[self.cur_pos[0]-5+x][self.cur_pos[1]-5+z]=map_info[x]['values'][z]
                    if(self.train_map[self.cur_pos[0]-5+x][self.cur_pos[1]-5+z]==2):
                        self.train_map[self.cur_pos[0]-5+x][self.cur_pos[1]-5+z]=1
        #v18
        #v17
        obstacle_size=7   #调成5,7,9,11  调的同时要改feature
        obstacle_map = np.zeros((obstacle_size,obstacle_size), dtype=np.float32)
        start=int((11-obstacle_size)/2)        
        for r in range(0,obstacle_size):
            for c in range(0,obstacle_size):
                if map_info[r+start]['values'][c+start] == 0: obstacle_map[r,c] = 1
        self.obstacle_flag = obstacle_map.flatten()
        #v17
        return targetIsEnd

    def process(self, frame_state, last_action):          # 外层调用
        targetIsEnd=self.pb2struct(frame_state, last_action)      # 用原始观测更新属性
        
        # Legal action
        # 合法动作
        v18=True
        if(v18==True):legal_action,forbidden_reward = self.my_train_map_get_legal_action()
        else:legal_action,forbidden_reward = self.get_legal_action()        # 用更新后的属性获取合法动作

        # Feature
        # ***更新后的属性在这直接打包成特征
        # 特征
        feature = np.concatenate([
            self.cur_pos_norm,
            self.feature_target_pos,            # baseline以end pos为目标，现在改成target pos
            self.feature_history_pos,
            legal_action,
            self.obstacle_flag,
            self.treasure_num,                  # v19 by Jiang
            self.feature_end_pos                # v19 by Jiang
            ])
        
        # ***奖励函数
        normalize=1.41*128
        cal_dist=np.linalg.norm(np.array(self.last_pos)-np.array(self.target_pos))-np.linalg.norm(np.array(self.cur_pos)-np.array(self.target_pos))
        target_reward=2*cal_dist/normalize
        if(targetIsEnd == True):target_reward*=1.5
        #v17
        if(self.step_no<10):history_reward=0.2*(self.whole_history_dist)
        else:history_reward=1.1*(self.whole_history_dist-18/normalize)
        #v17
        if(last_action>7 and cal_dist>=3.0):flash_reward=0.06
        else:flash_reward=0
        # v19 by Jiang
        end_reward = 0
        if self.treasure_num <= 6/8:
            if self.feature_end_pos[-1] < norm(1.41, 1.41*128):
                end_reward = -1
        reward = -0.008 + target_reward + history_reward + forbidden_reward + flash_reward + end_reward

        return (
            feature,
            legal_action,
            reward,
        )

    def get_legal_action(self):
        # if last_action is move and current position is the same as last position, add this action to bad_move_ids
        # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中
        forbidden_reward=0
        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
            forbidden_reward=-0.5  #给个撞墙惩罚
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

        return legal_action,forbidden_reward
        #legal_action=[True, True, ..., False, True, ...]

    #v18
    def my_train_map_get_legal_action(self):
        px,pz=self.cur_pos[0],self.cur_pos[1]
        k=5
        for i in range(6,10):   #可以调，注意算力
            if(px-i<0 or px+i>127 or pz-i<0 or pz+i>127):break
            else:J=True
            for j in range(pz-i,pz+i+1):
                if(self.train_map[px-i][j]==-1 or self.train_map[px+i][j]==-1):
                    J=False
                    break
            for j in range(px-i,px+i+1):
                if(self.train_map[j][pz-i]==-1 or self.train_map[j][pz+i]==-1):
                    J=False
                    break
            if(J==True):k+=1
            else:break
        '''
        print(px,pz,k)
        print("处理前:")
        for i in range(px-k,px+k+1):
            for j in range(pz-k,pz+k+1):
                print(self.train_map[i][j],end=' ')
            print()
        '''
        judge=True
        for i in range(px-k,px):  #x小
            if(i<=px-k+1):
                for j in range(pz-k,pz+k+1):
                    if(self.train_map[i][j]!=0):
                        judge=False
                        break
                if(judge==False):break
            else:
                if(self.train_map[i][pz-k]==0 and self.train_map[i][pz+k]==0):
                    for j in range(pz-k,pz+k+1):
                        if(self.train_map[i][j]>1):
                            judge=False
                            break
                else:judge=False
                if(judge==True):
                    for j in range(pz-k,pz+k+1):
                        self.train_map[i][j]=0
                else:break

        judge=True
        for i in range(px+k,px,-1):  #x大
            if(i>=px+k-1):
                for j in range(pz-k,pz+k+1):
                    if(self.train_map[i][j]!=0):
                        judge=False
                        break
                if(judge==False):break
            else:
                if(self.train_map[i][pz-k]==0 and self.train_map[i][pz+k]==0):
                    for j in range(pz-k,pz+k+1):
                        if(self.train_map[i][j]>1):
                            judge=False
                            break
                else:judge=False
                if(judge==True):
                    for j in range(pz-k,pz+k+1):
                        self.train_map[i][j]=0
                else:break
        
        judge=True
        for j in range(pz-k,pz):  #z小
            if(j<=pz-k+1):
                for i in range(px-k,px+k+1):
                    if(self.train_map[i][j]!=0):
                        judge=False
                        break
                if(judge==False):break
            else:
                if(self.train_map[px-k][j]==0 and self.train_map[px+k][j]==0):
                    for i in range(px-k,px+k+1):
                        if(self.train_map[i][j]>1):
                            judge=False
                            break
                else:judge=False
                if(judge==True):
                    for i in range(px-k,px+k+1):
                        self.train_map[i][j]=0
                else:break
        
        judge=True
        for j in range(pz+k,pz,-1):  #z大
            if(j>=pz+k-1):
                for i in range(px-k,px+k+1):
                    if(self.train_map[i][j]!=0):
                        judge=False
                        break
                if(judge==False):break
            else:
                if(self.train_map[px-k][j]==0 and self.train_map[px+k][j]==0):
                    for i in range(px-k,px+k+1):
                        if(self.train_map[i][j]>1):
                            judge=False
                            break
                else:judge=False
                if(judge==True):
                    for i in range(px-k,px+k+1):
                        self.train_map[i][j]=0
                else:break
        '''
        print("处理后:")
        for i in range(px-k,px+k+1):
            for j in range(pz-k,pz+k+1):
                print(self.train_map[i][j],end=' ')
            print()
        time.sleep(5)
        '''
        
        legal_action=[True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True]
        kx=[1,1,0,-1,-1,-1,0,1]
        kz=[0,1,1,1,0,-1,-1,-1]
        for k in range(8):
            if(self.train_map[px+kx[k]][pz+kz[k]]==0):
                legal_action[k]=False
            if(legal_action[k]==False and self.train_map[px+kx[k]*3][pz+kz[k]*3]==0 and self.train_map[px+kx[k]*5][pz+kz[k]*5]==0):
                legal_action[k+8]=False

        return legal_action,0
    
    

