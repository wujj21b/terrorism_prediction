from __future__ import division
from __future__ import print_function

import numpy as np

"""
Classes that are used to sample node neighborhoods
"""
class UniformNeighborSampler(object):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, visible_time, deg):
        self.adj_info = adj_info
        self.visible_time = visible_time
        self.deg = deg

    def __call__(self, inputs):
        with open('/data/Experiment/random_seed.txt','r') as f:
            seed=int(f.read())
        np.random.seed(seed)
        nodeids, num_samples, timeids,support_size,num_layers,k = inputs
        adj_lists = []
        for idx in range(len(nodeids)):
            node = nodeids[idx]
            timeid = timeids[idx // support_size]
            adj = self.adj_info[node, :]
            neighbors = []
            for neighbor in adj:
                if num_layers-k==1:
                    if self.visible_time[neighbor] <= timeid:
                        neighbors.append(neighbor)
                elif num_layers-k==2:
                    if self.visible_time[neighbor] <= timeid and self.deg[neighbor] > 0:
                        for second_neighbor in self.adj_info[neighbor]:
                            if self.visible_time[second_neighbor] <= timeid:
                                neighbors.append(neighbor)
                                break
                elif num_layers-k==3:
                    if self.visible_time[neighbor] <= timeid and self.deg[neighbor] > 0:
                        flag=0
                        for second_neighbor in self.adj_info[neighbor]:
                            if self.visible_time[second_neighbor] <= timeid and self.deg[second_neighbor] > 0:
                                for third_neighbor in self.adj_info[second_neighbor]:
                                    if self.visible_time[third_neighbor] <= timeid:
                                        neighbors.append(neighbor)
                                        flag=1
                                        break
                            if flag==1:
                                break
            assert len(neighbors) > 0
            if len(neighbors) < num_samples:
                neighbors = np.random.choice(neighbors, num_samples, replace=True)
            elif len(neighbors) > num_samples:
                neighbors = np.random.choice(neighbors, num_samples, replace=False)
            adj_lists.append(neighbors)
        return np.array(adj_lists, dtype=np.int32)
