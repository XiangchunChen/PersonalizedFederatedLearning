import networkx as nx
import pandas as pd

from drl.EdgeDevice import EdgeDevice


class FLNetwork:
    def __init__(self, action_spaces, n_features, schedule_path, network_edge_path, network_node_path, device_path):
        self.map = []
        self.action_space = action_spaces
        self.n_actions = len(self.action_space)
        self.n_features = n_features
        self.schedule_path = schedule_path
        self.network_edge_path = network_edge_path
        self.network_node_path = network_node_path
        self.device_path = device_path
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

        now_schedule = pd.read_csv(self.schedule_path)
        self.cecGraph = cecGraph
        self.deviceList = deviceList
        self.state_df = now_schedule
        self.state = self.state_df.iloc[0]

