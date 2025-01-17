#!python3
#-*- coding: utf-8 -*-

# https://raw.githubusercontent.com/y-kamiya/machine-learning-samples/7b6792ce37cc69051e9053afeddc6d485ad34e79/python3/reinforcement/dqn/sum_tree.py

# copy from https://github.com/jaara/AI-blog/blob/5aa9f0b/SumTree.py
# and add some functions
import numpy

print("sumtree.pyを実行する")

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )
        self.index_leaf_start = capacity - 1

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def max(self):
        return self.tree[self.index_leaf_start:].max()

    def add(self, p, data):
        idx = self.write + self.index_leaf_start

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
