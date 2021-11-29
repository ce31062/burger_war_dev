#!python3
#-*- coding: utf-8 -*-

from collections import namedtuple

print("transition.pyを実行する")

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward')
)
