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
import random
from kaiwu_agent.utils.common_func import Frame, attached

from tools.train_env_conf_validate import read_usr_conf
from agent_target_dqn.feature.definition import (
    sample_process,
)
from tools.metrics_utils import get_training_metrics


@attached
def workflow(envs, agents, logger=None, monitor=None):
    try:
        env, agent = envs[0], agents[0]
        episode_num_every_epoch = 1
        last_save_model_time = 0
        last_put_data_time = 0
        monitor_data = {}

        # Read and validate configuration file
        # 配置文件读取和校验
        usr_conf = read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error(f"usr_conf is None, please check agent_target_dqn/conf/train_env_conf.toml")
            return
        
        ordinary_buffer=[]
        ordinary_buffer_size=2000*180
        success_buffer=[]
        success_buffer_size=2000*90
        while True:
            for g_data, monitor_data,is_win in run_episodes(episode_num_every_epoch, env, agent, usr_conf, logger, monitor):
                #判断胜利机制
                if(is_win==True):
                    last_portion=0.5
                    start_index=int(len(g_data)*(1-last_portion))
                    success_buffer+=g_data[start_index:]
                    while(len(success_buffer)>success_buffer_size):success_buffer.pop(0)

                ordinary_buffer+=g_data
                while(len(ordinary_buffer)>ordinary_buffer_size):ordinary_buffer.pop(0)

                for i in range(25):  #25
                    batch=[]
                    if(len(ordinary_buffer)>=2000*5):batch+=random.sample(ordinary_buffer, 60)
                    if(len(success_buffer)>=200):batch+=random.sample(success_buffer, 20)
                    if(len(batch)>0):
                        agent.learn(batch)
                    else:break
                
                g_data.clear()

            # Save model file
            # 保存model文件
            now = time.time()
            if now - last_save_model_time >= 600:
                agent.save_model()
                last_save_model_time = now

            # Report monitoring metrics
            # 上报监控指标
            if now - last_put_data_time >= 60:
                monitor.put_data({os.getpid(): monitor_data})
                last_put_data_time = now

    except Exception as e:
        raise RuntimeError(f"workflow error")


def run_episodes(n_episode, env, agent, usr_conf, logger, monitor):
    try:
        for episode in range(n_episode):
            collector = list()
            win_rate = 0

            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                logger.info(f"training_metrics is {training_metrics}")

            # Reset the task and get the initial state
            # 重置任务, 并获取初始状态
            obs, extra_info = env.reset(usr_conf=usr_conf)
            if extra_info["result_code"] < 0:
                logger.error(
                    f"env.reset result_code is {extra_info['result_code']}, result_message is {extra_info['result_message']}"
                )
                raise RuntimeError(extra_info["result_message"])
            elif extra_info["result_code"] > 0:
                continue

            # At the start of each game, support loading the latest model file
            # The call will load the latest model from a remote training node
            # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
            agent.reset()
            agent.load_model(id="latest")

            # Feature processing
            # 特征处理
            obs_data, _ = agent.observation_process(obs, extra_info)

            done = False
            step = 0

            max_step_no = int(os.environ.get("max_step_no", "0"))

            while not done:
                # Agent performs inference, gets the predicted action for the next frame
                # Agent 进行推理, 获取下一帧的预测动作
                act_data, model_version = agent.predict(list_obs_data=[obs_data],win_rate=win_rate)

                # Unpack ActData into action
                # ActData 解包成动作
                act = agent.action_process(act_data[0])

                # Interact with the environment, execute actions, get the next state
                # 与环境交互, 执行动作, 获取下一步的状态
                step_no, _obs, terminated, truncated, _extra_info = env.step(act)
                if _extra_info["result_code"] != 0:
                    logger.warning(
                        f"_extra_info.result_code is {_extra_info['result_code']}, \
                        _extra_info.result_message is {_extra_info['result_message']}"
                    )
                    break

                step += 1

                # Feature processing
                # 特征处理
                _obs_data, reward = agent.observation_process(_obs, _extra_info)
                # Determine task over, and update the number of victories
                # 判断任务结束, 并更新胜利次数
                game_info = _extra_info["game_info"]
                if truncated:
                    win_rate = agent.update_win_rate(False)
                    reward = -2  #原为-3
                    logger.info(
                        f"Game truncated! step_no:{step_no} score:{game_info['total_score']} win_rate:{win_rate}"
                    )
                elif terminated:
                    win_rate = agent.update_win_rate(True)
                    reward = 30  #原为10
                    logger.info(
                        f"Game terminated! step_no:{step_no} score:{game_info['total_score']} win_rate:{win_rate}"
                    )
                done = terminated or truncated or (max_step_no > 0 and step >= max_step_no)

                # Construct task frames to prepare for sample construction
                # 构造任务帧，为构造样本做准备
                frame = Frame(
                    obs=obs_data.feature,
                    _obs=_obs_data.feature,
                    obs_legal=obs_data.legal_act,
                    _obs_legal=_obs_data.legal_act,
                    act=act,
                    rew=reward,
                    done=done,
                    ret=reward,
                )

                collector.append(frame)
                # If the task is over, the sample is processed and sent to training
                # 如果任务结束，则进行样本处理，将样本送去训练
                if done:
                    if(terminated):diy2=1
                    else:diy2=0
                    if monitor:
                        monitor_data = {
                            "diy_1": step,
                            "diy_2": diy2,
                            "diy_3": _obs['frame_state']['heroes'][0]['pos']['x'],
                            "diy_4": _obs['frame_state']['heroes'][0]['pos']['z'],
                            "diy_5": win_rate,
                        }

                    if len(collector) > 0:
                        collector = sample_process(collector)
                        yield collector, monitor_data,terminated  #加了判断是否胜利
                    break

                # Status update
                # 状态更新
                obs_data = _obs_data
                extra_info = _extra_info
    except Exception as e:
        logger.error(f"run_episodes error")
        raise RuntimeError(f"run_episodes error")
