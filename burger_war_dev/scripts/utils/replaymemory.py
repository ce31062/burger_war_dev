#!python3
#-*- coding: utf-8 -*-

from state import State
from transition import Transition

print("replaymemory.pyを実行する")
class ReplayMemory(object):

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, state, action, state_next, reward):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
