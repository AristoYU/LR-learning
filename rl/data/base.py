#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
# @Author: AristoYU
# @Date: 2025-03-26 09:37:53
# @LastEditTime: 2025-03-26 09:40:30
# @LastEditors: AristoYU
# @Description: 
# @FilePath: /LR-learning/rl/data/base.py
"""

from collections import namedtuple, deque
import numpy as np


PPOExperience = namedtuple('PPOExperience', ['observations', 'actions', 'values', 'returns', 'advantages', 'log_probs'])


class ReplayBuffer(object):

    def __init__(self, capacity: int = 1000):
        self.buffer = deque(maxlen=capacity)
        self.expr_type = None

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        if self.expr_type is None:
            self.expr_type = type(experience)
        self.buffer.append(experience)

    def sample(self, sample_size: int = 1):
        replace = False if sample_size < len(self) else True
        indices = np.random.choice(len(self), sample_size, replace=replace)
        # collate experiences
        for idx in indices:
            yield {key: getattr(self.buffer[idx], key) for key in self.expr_type._fields}

    def clear(self):
        self.buffer.clear()
