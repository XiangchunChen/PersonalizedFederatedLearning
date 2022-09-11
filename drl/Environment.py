import math
import sys

import networkx as nx
import pandas as pd

from EdgeDevice import EdgeDevice
from Task import Task

import tensorflow.compat.v1 as tf

class MultiHopNetwork:
    def __init__(self, action_spaces, n_features, schedule_path, network_edge_path, network_node_path, device_path, task_file_path, task_pre_path):
        self.map = []
        # TODO: revise
        self.action_space = action_spaces
        self.n_actions = len(self.action_space)
        # TODO: revise
        self.n_features = n_features
        self.task_release_time = {}
        self.schedule_path = schedule_path
        self.network_edge_path = network_edge_path
        self.network_node_path = network_node_path
        self.device_path = device_path
        self.task_file_path = task_file_path
        self.task_pre_path = task_pre_path
        self._build_network()

    def _build_network(self):
        cecGraph = nx.Graph()
        # network_edge_path, network_node_path, device_path
        f1 = open(self.network_node_path, 'r')
        lines = f1.readlines()
        for line in lines:
            line = line.replace('\n', '').replace('\r', '')
            info = line.split(',')
            if len(info) == 3:
                cecGraph.add_node(int(info[0]), name=info[1], weight=info[2])
        f1.close()

        f2 = open(self.network_edge_path, 'r')
        lines = f2.readlines()
        for line in lines:
            line = line.replace('\n', '').replace('\r', '')
            info = line.split(',')
            if len(info) == 3:
                cecGraph.add_edge(int(info[0]), int(info[1]), weight=float(info[2]))
                cecGraph[int(info[0])][int(info[1])]['flow'] = []
        f2.close()

        deviceList = []
        f1 = open(self.device_path, 'r')
        lines = f1.readlines()
        for line in lines:
            line = line.replace('\n', '').replace('\r', '')
            info = line.split(',')
            if len(info) == 3:
                device = EdgeDevice(int(info[0]), int(info[1]), float(info[2]))
                # device.printInfo()
                deviceList.append(device)
        f1.close()

        # task_file_path = "task/task_info_30.csv"
        taskList = []
        self.task_dic = {}
        f1 = open(self.task_file_path, 'r')
        lines = f1.readlines()

        for line in lines:
            # print(line)
            line = line.replace('\n', '').replace('\r', '')
            info = line.split(',')
            # if len(info) == 8:
            # print(info)
            task = Task(int(info[0]), int(info[1]), int(info[2]),
                        int(info[3]), int(info[4]), int(info[5]))
            # task.printInfo()
            taskList.append(task)
            if int(info[1]) in self.task_dic.keys():
                self.task_dic[int(info[1])] = self.task_dic[int(info[1])] + 1
            else:
                self.task_dic[int(info[1])] = 1
        f1.close()

        f1 = open(self.task_pre_path, "r")
        lines = f1.readlines()
        edges_dic = {}
        for line in lines:
            # print(line)
            info = line.strip("\n").split(",")
            # print(info)
            start = int(info[1])
            end = int(info[0])
            if start != end:
                if start in edges_dic.keys():
                    tempList = edges_dic[start]
                    tempList.append(end)
                    edges_dic[start] = tempList
                else:
                    edges_dic[start] = [end]

        for key, val in edges_dic.items():
            for task in taskList:
                if task.subId == key:
                    task.setSucceList(val)
                    break
        f1.close()

        now_schedule = pd.read_csv(self.schedule_path)
        self.cecGraph = cecGraph
        self.deviceList = deviceList
        self.state_df = now_schedule
        self.state = self.state_df.iloc[0]
        self.taskList = taskList

    def getAverageWaittime(self):
        return 0

    def getAverageCtime(self):
        return 0

    def step(self, action_index, task, t):
        # 在环境中对action_index进行转换
        list_arr = list(action_index)

        action = (math.ceil(list_arr[0] * 1000) % 6) + 1
        print(action)
        # action = self.action_space[action_index]
        bandwidth = (math.ceil(list_arr[1] * 1000) % 9) + 1
        waitTime = (math.ceil(list_arr[2] * 1000) % 9) + 1

        reward, finishTime = self.update_state(action, bandwidth, waitTime, task, t)
        self.task_release_time[task.subId + task.taskId] = finishTime
        next_state = self.state_df.iloc[0]
        done = True
        return next_state, reward, done, finishTime

    def add_new_state(self,t):
        a = self.state_df.iloc[0].values.tolist()
        d = pd.DataFrame(columns=self.state_df.columns)
        d.loc[-1] = a
        d['time'] = t
        self.state_df = pd.concat([d, self.state_df], axis=0, ignore_index=True)
        self.state_df.to_csv("file/now_schedule.csv", index=0)

    def updateDeviceWaitTime(self, deviceId, waitTime):
        for device in self.deviceList:
            if deviceId == device.deviceId:
                device.setWaitTime(waitTime)

    def getNextEdgeWeight(self):
        edge_waitTime = 0
        return edge_waitTime

    # waitTime是指flow的等待时间
    def update_state(self, action, bandwidth, waitTime, task, t):
        # TODO 为了提高效率，将waitTime预设为0
        waitTime = 0
        ctime = 0
        paths = self.searchGraph(self.cecGraph, task.source, action, task.dataSize)
        list1 = self.getEdgeList(paths)
        print("list1", list1)
        print("task.source, action", task.source, action)
        path_bandwidth = sys.maxsize
        max_edge_waitTime = 0
        for edge in self.cecGraph.edges:
            if edge in list1:
                edge_name = "edge_weight_"+str(edge[0])+str(edge[1])
                # path_bandwidth一定非0，所以才会有waitTime说法，如果是同时通过多条，waitTime是否有效呢
                path_bandwidth = min(self.cecGraph[edge[0]][edge[1]]['weight'], path_bandwidth)
                edge_waitTime = 0
                if t in self.state_df['time'].values.tolist():
                    edge_waitTime = self.state_df[edge_name][self.state_df['time'] == t].values.tolist()[0]
                max_edge_waitTime = max(max_edge_waitTime, edge_waitTime)

        if max_edge_waitTime <= waitTime:
            # 如果决定等待一段时间再进行的话，这个defer execution会使得path_bandwidth比预想中更大, 误差范围是[pathbandwidth, future_pathbandwidth]
            max_edge_waitTime = waitTime
            if path_bandwidth > bandwidth:
                path_bandwidth = bandwidth
        else:
            if path_bandwidth > bandwidth:
                path_bandwidth = bandwidth
        begin_t = t+max_edge_waitTime

        ctime = task.dataSize/path_bandwidth
        if task.source != action:
            for i in range(begin_t, begin_t + int(ctime) + 1):
                if i in self.state_df['time'].values.tolist():
                    for j in range(1, len(list1)+1):
                        edge_0 = list1[j-1][0]
                        edge_1 = list1[j-1][1]
                        edge_name = "edge_weight_"+str(edge_0)+str(edge_1)
                        self.state_df.loc[self.state_df['time'] == i, edge_name] = self.state_df.loc[self.state_df['time'] == i, edge_name] + begin_t+int(ctime) - i
                else:
                    a = self.state_df.iloc[0].values.tolist()
                    d = pd.DataFrame(columns=self.state_df.columns)
                    d.loc[-1] = a
                    d['time'] = i
                    for j in range(1, len(list1)+1):
                        edge_0 = list1[j-1][0]
                        edge_1 = list1[j-1][1]
                        edge_name = "edge_weight_"+str(edge_0)+str(edge_1)
                        d[edge_name] = begin_t+int(ctime) - i
                    self.state_df = pd.concat([d, self.state_df], axis=0, ignore_index=True)

            end_ctime = begin_t + int(ctime) + 1
            if end_ctime not in self.state_df['time'].values.tolist():
                a = self.state_df.iloc[0].values.tolist()
                d = pd.DataFrame(columns=self.state_df.columns)
                d.loc[-1] = a
                d['time'] = end_ctime
                for j in range(1, len(list1)+1):
                    edge_0 = list1[j-1][0]
                    edge_1 = list1[j-1][1]
                    edge_name = "edge_weight_"+str(edge_0)+str(edge_1)
                    d[edge_name] = 0
                self.state_df = pd.concat([d, self.state_df], axis=0, ignore_index=True)
        else:
            end_ctime = t

        target_action_device = None
        for device in self.deviceList:
            if device.deviceId == action:
                target_action_device = device
        device_name = "device_"+str(target_action_device.deviceId)+"_time"
        p_waitTime = 0
        waitTimeList = self.state_df[device_name][self.state_df['time'] == end_ctime].values.tolist()
        if len(waitTimeList) != 0:
            p_waitTime = waitTimeList[0]

        begin_ptime = end_ctime + p_waitTime
        preList = task.getSucceList()
        max_new_ctime = 0
        for temp_task in self.taskList:
            if temp_task.subId in preList:
                key = temp_task.subId + temp_task.taskId
                # print("subId",task.subId,"preList",preList)
                max_new_ctime = max(self.task_release_time[key], max_new_ctime)
        if max_new_ctime > begin_ptime:
            p_waitTime = max_new_ctime - end_ctime

        begin_ptime = end_ctime + p_waitTime
        ptime = task.cload/target_action_device.cpuNum
        for i in range(begin_ptime, begin_ptime + int(ptime)+1):
            if i in self.state_df['time'].values.tolist():
                device_name = "device_"+str(target_action_device.deviceId)+"_time"
                self.state_df.loc[self.state_df['time'] == i, device_name] = self.state_df.loc[self.state_df['time'] == i, device_name] + begin_ptime + int(ptime)+1-i
            else:
                a = self.state_df.iloc[0].values.tolist()
                d = pd.DataFrame(columns=self.state_df.columns)
                d.loc[-1] = a
                d['time'] = i
                d[device_name] = begin_ptime + int(ptime)+1-i
                self.state_df = pd.concat([d, self.state_df], axis=0, ignore_index=True)
        final_time = begin_ptime + int(ptime) + 1
        if final_time not in self.state_df['time'].values.tolist():
            a = self.state_df.iloc[0].values.tolist()
            d = pd.DataFrame(columns=self.state_df.columns)
            d.loc[-1] = a
            d['time'] = final_time
            d[device_name] = 0
            self.state_df = pd.concat([d, self.state_df], axis=0, ignore_index=True)

        self.state_df.to_csv("file/now_schedule.csv", index=0)
        waitTime = max_edge_waitTime + p_waitTime
        reward = math.exp(-(ctime+ptime+waitTime))
        b_i = (task.subId - task.taskId *10)
        if b_i <= 0:
            b_i = 1
        print(b_i)
        # # TODO b表示task的总层级(简单化，是subtask的数目）
        # # dic ={2: 2, 3: 1, 7: 2, 8: 17, 15: 5, 16: 1, 25: 3, 30: 1, 31: 3, 34: 10, 35: 1, 37: 1, 49: 7, 52: 2, 53: 1, 60: 1, 65: 1, 66: 3, 67: 1, 69: 1, 73: 5, 76: 4, 78: 2, 79: 1, 81: 2, 82: 4, 87: 4, 89: 3, 90: 9, 92: 1, 95: 1, 97: 4, 98: 2, 102: 1, 110: 1, 113: 1, 119: 2, 121: 1, 126: 1, 127: 16}
        b = self.task_dic[task.taskId]
        if b > 1:
            weight = (reward - (b_i-1)/(b-1))/b_i
        else:
            weight = 1
        if weight < 0:
            weight = 0
        if weight > 1:
            weight = 1
        # weight = tf.clip_by_value((reward - (b_i-1)/(b-1))/b_i, 0, 1)
        reward = (1+weight)*reward
        return reward, final_time

    def getEdgeList(self, path):
        pathList = []
        if len(path) >= 2:
            for i in range(len(path)-1):

                if path[i] > path[i+1]:
                    list = (path[i+1], path[i])
                else:
                    list = (path[i], path[i+1])
                pathList.append(list)
        elif len(path) == 1:
            pathList = [(path[0])]
        return pathList

    def searchGraph(self, graph, start, end, dataSize):
        results = []
        self.generatePath(graph, [start], end, results)
        results.sort(key=lambda x: len(x))
        if len(results) == 0:
            return []
        minPath = results[0]
        minCtime = self.calculateCtime(graph, minPath, dataSize)
        for path in results:
            tempCtime = self.calculateCtime(graph, path, dataSize)
            if tempCtime < minCtime:
                minPath = path
                minCtime = tempCtime
        return minPath

    def calculateCtime(self, graph, path, dataSize):
        sum = 0
        for i in range(len(path)-1):
            sum += dataSize/graph.edges[path[i], path[i+1]]['weight']
        return sum

    def generatePath(self, graph, path, end, results):
        state = path[-1]
        if state == end:
            results.append(path)
        else:
            for arc in graph[state]:
                if arc not in path:
                    self.generatePath(graph, path + [arc], end, results)

    def get_observation(self):
        return self.state

    def reset(self):
        df = pd.read_csv("now_schedule.csv")
        df.to_csv("file/now_schedule.csv", index=0)
        self._build_network()
        return self.state
