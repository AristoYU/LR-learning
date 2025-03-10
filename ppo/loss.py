#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
# @Author: AristoYU
# @Date: 2025-03-10 10:57:53
# @LastEditTime: 2025-03-10 12:51:29
# @LastEditors: AristoYU
# @Description: PPO loss functions
# @FilePath: /LR-learning/PPO/loss.py
"""

import torch
from torch import Tensor
import torch.nn.functional as F


def policy_loss(advantages: Tensor, ratio: Tensor, clip_coef: float) -> Tensor:
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    return torch.max(pg_loss1, pg_loss2).mean()


def value_loss(
    new_values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_coef: float,
    clip_vloss: bool,
    vf_coef: float,
) -> Tensor:
    new_values = new_values.view(-1)
    if not clip_vloss:
        values_pred = new_values
    else:
        values_pred = old_values + torch.clamp(
            new_values - old_values, -clip_coef, clip_coef
        )
    return vf_coef * F.mse_loss(values_pred, returns)


def entropy_loss(entropy: Tensor, ent_coef: float) -> Tensor:
    return -entropy.mean() * ent_coef
