#!python3
#-*- coding: utf-8 -*-

from sumtree import SumTree
import random
from state import State
from transition import Transition
import numpy as np
from replaymemory import ReplayMemory

print("permemory.pyを実行する")

class PERMemory(ReplayMemory):
    epsilon = 0.0001
    alpha = 0.6

    def __init__(self, CAPACITY):
        super(PERMemory, self).__init__(CAPACITY)
        self.tree = SumTree(CAPACITY)
        self.size = 0

    # Proportional prioritizationによるpriorityの計算
    def _getPriority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha

    def push(self, state, action, state_next, reward):
        """state, action, state_next, rewardをメモリに保存します"""
        self.size += 1

        priority = self.tree.max()
        if priority <= 0:
            priority = 1

        self.tree.add(priority, Transition(state, action, state_next, reward))

    def sample(self, batch_size):
        data_list = []
        indexes = []
        for rand in np.random.uniform(0, self.tree.total(), batch_size):
            (idx, _, data) = self.tree.get(rand)
            data_list.append(data)
            indexes.append(idx)

        return data_list, indexes

    def update(self, idx, td_error):
        priority = self._getPriority(td_error)
        self.tree.update(idx, priority)

    def __len__(self):
        return self.size
