#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
# @Author: AristoYU
# @Date: 2025-03-26 09:42:21
# @LastEditTime: 2025-04-01 14:55:26
# @LastEditors: AristoYU
# @Description: 
# @FilePath: /LR-learning/rl/model/loss/ppo_loss.py
"""

import torch
import torch.nn.functional as F


def policy_loss_factory(clip_coef: float):
    def policy_loss(advantages: torch.Tensor, ratio: torch.Tensor):
        advantages = advantages.unsqueeze(-1)
        p_loss1 = -advantages * ratio
        p_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        return torch.max(p_loss1, p_loss2).mean()
    return policy_loss


def value_loss_factory(clip_coef: float, clip_vloss: bool, vf_coef: float):
    def value_loss(new_values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor):
        new_values = new_values.view(-1)
        if not clip_vloss:
            values_pred = new_values
        else:
            values_pred = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
        return vf_coef * F.mse_loss(values_pred, returns)
    return value_loss


def entropy_loss_factory(ent_coef: float):
    def entropy_loss(entropy: torch.Tensor):
        return -entropy.mean() * ent_coef
    return entropy_loss
