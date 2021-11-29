#!python3
#-*- coding: utf-8 -*-

from collections import namedtuple

print("state.pyを実行する")

State = namedtuple (
    'State', ('pose', 'lidar', 'image', 'mask')
)
