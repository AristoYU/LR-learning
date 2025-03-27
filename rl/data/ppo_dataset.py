#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
# @Author: AristoYU
# @Date: 2025-03-26 09:38:30
# @LastEditTime: 2025-03-26 09:41:47
# @LastEditors: AristoYU
# @Description: 
# @FilePath: /LR-learning/rl/data/ppo_dataset.py
"""

from torch.utils.data import IterableDataset

from .base import ReplayBuffer


class PPODataset(IterableDataset):

    def __init__(self, buffer: ReplayBuffer, sample_step_num: int = 1):
        self.buffer = buffer
        self.sample_step_num = sample_step_num

    def __len__(self):
        return self.sample_step_num

    def __iter__(self):
        for data in self.buffer.sample(self.sample_step_num):
            yield data