'''
Author: your name
Date: 2021-02-21 21:43:53
LastEditTime: 2021-02-24 15:21:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /csy/ours/dataset.py
'''
from dgl.data.graph_serialize import GraphData
import torch
import numpy as np
import dgl
import utils


def batcher():
    def batcher_dev(batch):
        idx, graph_q, graph_k = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return list(idx), graph_q, graph_k
    return batcher_dev


class RandomWalkDataset(torch.utils.data.Dataset):
    def __init__(self,
                 graph,
                 rw_hops=10,
                 restart_prob=0.8,
                 n_pos=2,
                 step_dist=[1.0, 0.0, 0.0],):
        super(RandomWalkDataset).__init__()
        self.graph = graph
        self.rw_hops = rw_hops
        self.restart_prob = restart_prob
        self.step_dist = step_dist
        self.length = graph.number_of_nodes()
        self.total = self.length
        assert sum(step_dist) == 1.0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        node_idx = idx
        # print(idx)
        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        max_nodes_per_seed = self.rw_hops

        trace_1 = dgl.sampling.random_walk(self.graph,
                                           [node_idx],
                                           length=self.rw_hops,
                                           prob=self.graph.in_degrees(),
                                           restart_prob=self.restart_prob)
        trace_2 = dgl.sampling.random_walk(self.graph,
                                           [node_idx],
                                           length=self.rw_hops,
                                           prob=self.graph.in_degrees(),
                                           restart_prob=self.restart_prob)

        graph_q = dgl.node_subgraph(
            graph=self.graph, nodes=torch.unique(trace_1[0]))
        graph_q.ndata["seed"] = torch.zeros(
            graph_q.number_of_nodes(), dtype=torch.bool)
        graph_q.ndata["seed"][graph_q.ndata['_ID'] == idx] = True

        graph_k = dgl.node_subgraph(
            graph=self.graph, nodes=torch.unique(trace_2[0]))
        graph_k.ndata["seed"] = torch.zeros(
            graph_k.number_of_nodes(), dtype=torch.bool)
        graph_k.ndata["seed"][graph_k.ndata['_ID'] == idx] = True

        return idx, graph_q, graph_k
