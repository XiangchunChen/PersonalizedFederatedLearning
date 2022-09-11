import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Environment import MultiHopNetwork
from Task import Task
from DDPG import DDPG


def run_network(env, agent,task_num):
    step = 0
    completion_time_dic = {}
    reward_dic = {}
    # TODO: 在一轮迭代中，client上传参数，server下载模型
    for episode in range(1):
        # initial observation
        print("episode: ", episode)
        observation = env.reset()
        episode_done = False
        time_step = 1
        max_time = 10
        task_count = 0
        task_time_dic = {}
        subtask_time_dic = {}
        avg_reward_dic = {}
        # key: task_id, value: time
        while not episode_done or time_step < max_time:
            # RL choose action based on observation
            print("-------------------timestep: ", time_step, "-------------------")
            tasks = getTasksByTime(env.taskList, time_step)
            task_count += len(tasks)

            if len(tasks) != 0:
                for i in range(len(tasks)):
                    task = tasks[i]
                    print("task:", task.subId)
                    action = agent.choose_action(observation)

                    observation_, reward, done, finishTime = env.step(action, task, time_step)
                    if task.taskId in avg_reward_dic.keys():
                        avg_reward_dic[task.taskId] = max(avg_reward_dic[task.taskId], reward)
                    else:
                        avg_reward_dic[task.taskId] = reward

                    subtask_time_dic[task.subId] = finishTime-time_step+1
                    if task.taskId in task_time_dic.keys():
                        task_time_dic[task.taskId] = max(task_time_dic[task.taskId], finishTime-time_step+1)
                    else:
                        task_time_dic[task.taskId] = finishTime-time_step+1
                    # TODO s, lstm_s,  a, r, s_, lstm_s_
                    agent.store_transition(observation, action, reward, observation_)

                    if (step > 10) and (step % 5 == 0):
                        agent.learn()

                    observation = observation_

            else:
                env.add_new_state(time_step)
            print(task_count)
            if task_count == task_num:
                completion_time = 0
                for task, time_value in task_time_dic.items():
                    print(task, time_value)
                    completion_time += time_value
                print("subtask_time_dic")
                print(subtask_time_dic)
                completion_time_dic[episode] = completion_time
                avg_reward = 0
                for task, reward in avg_reward_dic.items():
                    avg_reward += reward
                print("completion_time:", completion_time)
                reward_dic[episode] = avg_reward
                print("reward:", avg_reward)
                break

            time_step += 1
        step += 1
    agent.save_net()
    return completion_time_dic, reward_dic

def checkAllocated(taskList):
    res = False
    for task in taskList:
        if not task.isAllocated:
            res = True
    return res

def getTasksByTime(taskList, time_step):
    tasks = []
    for task in taskList:
        if task.release_time == time_step:
            tasks.append(task)
    sorted(tasks, key=lambda task: task.subId)
    return tasks

def destory(destory_path):
    df = pd.read_csv(destory_path)
    df.to_csv("DDPGTO/file/now_schedule.csv", index=0)

def plotCompletionTime(completion_time_dic,name):
    f1 = open("result/"+name+".csv", "w")
    x = []
    y = []
    for key, value in completion_time_dic.items():
        f1.write(str(key)+","+str(value)+"\n")
        x.append(key)
        y.append(value)
    f1.close()
    plt.plot(x, y)
    plt.ylabel(name)
    plt.xlabel('Episodes')
    plt.savefig("result/"+name+'.pdf')
    plt.show()


if __name__ == "__main__":
    ali_data = "Rfile/Test/Random_test_data"
    task_file_path = ali_data+"/train_info.csv"
    task_pre_path = ali_data+"/train_pre.csv"
    network_node_path = "Rfile/network_node_info.csv"
    network_edge_path = "Rfile/network_edge_info.csv"
    device_path = "Rfile/device_info.csv"
    schedule_path = "Rfile/now_schedule.csv"
    destory_path = "now_schedule.csv"
    edges_devices_num = 16
    devices =[1,2,3,4,5,6,7,8]
    f1 = open(task_file_path, "r")
    lines = f1.readlines()
    task_num = len(lines)

    dic_task_time = {}
    env = MultiHopNetwork(devices, edges_devices_num, schedule_path, network_edge_path, network_node_path, device_path, task_file_path, task_pre_path)

    s_dim = env.n_features
    a_dim = 3  # < device, bandwidth, waitTime>
    a_bound = env.n_actions

    agent = DDPG(a_dim, s_dim, a_bound)
    completion_time_dic, reward_dic = run_network(env, agent, task_num)
    plotCompletionTime(completion_time_dic, "completion_time")
    plotCompletionTime(reward_dic, "reward")
    agent.plot_cost()