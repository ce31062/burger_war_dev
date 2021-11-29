#!python3
#-*- coding: utf-8 -*-

import random
from state import State
from transition import Transition
from replaymemory import ReplayMemory

print("random_replaymemory.pyを実行する")

class RandomReplayMemory(ReplayMemory):

    def push(self, state, action, state_next, reward):
        """state, action, state_next, rewardをメモリに保存します"""

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        """batch_size分だけ、ランダムに保存内容を取り出します"""
        return random.sample(self.memory, batch_size), None

    def __len__(self):
        return len(self.memory)
