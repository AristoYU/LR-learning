#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
# @Author: AristoYU
# @Date: 2025-03-26 10:03:36
# @LastEditTime: 2025-03-26 10:55:14
# @LastEditors: AristoYU
# @Description: 
# @FilePath: /LR-learning/rl/model/module/ppo_module.py
"""

from copy import deepcopy
import math
from typing import Optional
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from lightning.pytorch import LightningModule
from torchmetrics import MeanMetric

from rl.data.base import PPOExperience, ReplayBuffer
from rl.data.ppo_dataset import PPODataset
from ..loss.ppo_loss import policy_loss_factory, value_loss_factory, entropy_loss_factory


class PPOLinearModuleBase(nn.Module):

    def __init__(self, net_cfg: dict, inp_channels: int, out_channels: int, *args, **kwargs):
        super().__init__()

        self.net = self._create_mlp(net_cfg, inp_channels, out_channels)

    def _create_mlp(self, net_cfg, inp_channels, out_channels):
        channels = [inp_channels] + net_cfg['channels'] + [out_channels]
        use_layer_norm = net_cfg.get('use_layer_norm', False)
        act_func = net_cfg.get('act_func', 'ReLU')

        _mlp = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_chn, out_chn, bias=True),
                nn.LayerNorm(out_chn) if use_layer_norm else nn.Identity(),
                getattr(nn, act_func)())
            for in_chn, out_chn in zip(channels[:-2], channels[1:-1])
        ])
        _mlp.append(nn.Linear(channels[-2], channels[-1], bias=True))
        return _mlp

    def forward(self, x: torch.Tensor):
        return self.net(x)


class PPOConvMoudleBase(nn.Module):

    def __init__(self, net_cfg, inp_channels, out_channels, *args, **kwargs):
        super().__init__()

        self.net = self.create_conv(net_cfg, inp_channels, out_channels)

    def _create_conv(self, net_cfg, inp_channels, out_channels):
        channels = [inp_channels] + net_cfg['channels'] + [out_channels]
        conv_cfg = net_cfg.get('conv_cfg', {})
        norm_cfg = net_cfg.get('norm_cfg', {})
        act_cfg = net_cfg.get('act_cfg', {})
        act_func = act_cfg.pop('type', 'ReLU')

        _conv = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_chn, out_chn, **conv_cfg),
                nn.BatchNorm2d(out_chn, **norm_cfg),
                getattr(nn, act_func)(**act_cfg))
            for in_chn, out_chn in zip(channels[:-2], channels[1: -1])
        ])
        _conv.append(nn.AdaptiveAvgPool2d((1, 1)))
        _conv.append(nn.Conv2d(channels[-2], channels[-1], **conv_cfg))
        return _conv

    def forward(self, x: torch.Tensor):
        bs = x.shape[0]
        return self.net(x).reshape(bs, -1)


class PPOActorModule(nn.Module):

    def __init__(self, net_cfg: dict, envs: gym.vector.SyncVectorEnv, use_mlp: bool = False, use_conv: bool = False):
        super().__init__()

        assert not (use_mlp and use_conv), '`use_mlp` and `use_conv` cannot be True at the same time'
        assert use_mlp or use_conv, '`use_mlp` and `use_conv` cannot be False at the same time'

        out_channels = envs.single_action_space.n
        if use_mlp:
            inp_channels = math.prod(envs.single_observation_space.shape)
            self._net = PPOLinearModuleBase(net_cfg, inp_channels, out_channels)
        elif use_conv:
            inp_channels = envs.single_observation_space.shape[-1]
            self._net = PPOConvMoudleBase(net_cfg, envs.single_observation_space.shape[0], envs.action_space.n)

    def forward(self, x: torch.Tensor):
        return self._net(x)


class PPOCriticModule(nn.Module):

    def __init__(self, net_cfg: dict, envs: gym.vector.SyncVectorEnv, use_mlp: bool = False, use_conv: bool = False):
        super().__init__()

        assert not (use_mlp and use_conv), '`use_mlp` and `use_conv` cannot be True at the same time'
        assert use_mlp or use_conv, '`use_mlp` and `use_conv` cannot be False at the same time'

        out_channels = 1
        if use_mlp:
            inp_channels = math.prod(envs.single_observation_space.shape)
            self._net = PPOLinearModuleBase(net_cfg, inp_channels, out_channels)
        elif use_conv:
            inp_channels = envs.single_observation_space.shape[-1]
            self._net = PPOConvMoudleBase(net_cfg, envs.single_observation_space.shape[0], envs.action_space.n)

    def forward(self, x: torch.Tensor):
        return self._net(x)


class PPOModule(nn.Module):

    def __init__(self, envs: gym.vector.SyncVectorEnv, actor_cfg: dict, critic_cfg: dict, *args, **kwargs):

        super().__init__()

        self.actor: PPOActorModule = PPOActorModule(**actor_cfg, envs=envs)
        self.critic: PPOCriticModule = PPOCriticModule(**critic_cfg, envs=envs)

    def get_action(self, obs: torch.Tensor, act: Optional[torch.Tensor] = None, greedy: bool = False):
        act_logits = self.actor(obs)
        # TODO: add continuous action space support
        if greedy:
            probs = F.softmax(act_logits, dim=-1)
            return torch.argmax(probs, dim=-1)
        else:
            dist = Categorical(logits=act_logits)
            if act is None:
                act = dist.sample()
            return act, dist.log_prob(act), dist.entropy()

    def get_value(self, obs: torch.Tensor):
        return self.critic(obs)

    def forward(self, obs: torch.Tensor, act: torch.Tensor = None, greedy: bool = False):
        if greedy:
            return self.get_action(obs, act, greedy)
        else:
            act, log_prob, entropy = self.get_action(obs, act)
            val = self.get_value(obs)
            return act, log_prob, entropy, val

    @torch.no_grad()
    def estimate_returns_and_advantages(
            self,
            rewards: torch.Tensor,
            values: torch.Tensor,
            dones: torch.Tensor,
            next_obs: torch.Tensor,
            next_done: torch.Tensor,
            num_steps: int,
            gamma: float,
            gae_lambda: float):
        next_values = self.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = torch.logical_not(next_done)
            else:
                next_non_terminal = torch.logical_not(dones[t + 1])
                next_values = values[t + 1]
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        returns = advantages + values
        return returns, advantages


class PPOLightningModule(LightningModule):

    def __init__(
            self,
            env_cfg: dict,
            data_cfg: dict,
            ppo_cfg: dict,
            loss_cfg: dict = {
                'policy_loss': {'clip_coef': 0.2},
                'value_loss': {'clip_coef': 0.2, 'clip_vloss': False, 'vf_coef': 1.0},
                'entropy_loss': {'ent_coef': 0.0}},
            optim_cfg: dict = {'lr': 1e-4},
            running_cfg: dict = {
                'seed': 42,
                'log_root': '',
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'normalize_advantages': False,
                'update_interval': 1,
                'update_steps': 10},
            torchmetrics_cfg: dict = {},
            **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self._load_loss(loss_cfg)
        self.train_envs = self._load_env(env_cfg, data_cfg, 
                                         os.path.join(self.hparams.running_cfg.get('log_root', ''), 'video', 'train'),
                                         self.hparams.running_cfg.get('seed', 42))
        self.val_env = self._load_env(self.hparams.env_cfg, {'batch_size': 1},
                                      os.path.join(self.hparams.running_cfg.get('log_root', ''), 'video', 'val'),
                                      self.hparams.running_cfg.get('seed', 42))

        self.replay_buffer = self._load_replay_buffer(data_cfg)
        self.ppo_model: PPOModule = PPOModule(envs=self.train_envs, **ppo_cfg)

        self.avg_pg_loss = MeanMetric(**torchmetrics_cfg)
        self.avg_value_loss = MeanMetric(**torchmetrics_cfg)
        self.avg_ent_loss = MeanMetric(**torchmetrics_cfg)

        self._reset_data = False

    def forward(self, obs: torch.Tensor, act: Optional[torch.Tensor] = None, *args, **kwargs):
        return self.ppo_model(obs, act, *args, **kwargs)

    def on_fit_start(self):
        num_steps = self.hparams.data_cfg.get('total_timestep_n', 1000)

        self.observation_shape = self.train_envs.single_observation_space.shape
        self.action_shape = self.train_envs.single_action_space.shape

        self.observations_buffer = torch.zeros((num_steps, self.train_envs.num_envs) + self.observation_shape,
                                               device=self.device)
        self.actions_buffer = torch.zeros((num_steps, self.train_envs.num_envs) + self.action_shape, device=self.device)
        self.log_probs_buffer = torch.zeros((num_steps, self.train_envs.num_envs), device=self.device)
        self.rewards_buffer = torch.zeros((num_steps, self.train_envs.num_envs), device=self.device)
        self.dones_buffer = torch.zeros((num_steps, self.train_envs.num_envs), device=self.device)
        self.values_buffer = torch.zeros((num_steps, self.train_envs.num_envs), device=self.device)

        self._load_data()

    def on_train_epoch_start(self):
        if self._reset_data:
            self._load_data()
            self._reset_data = False

    def training_step(self, batch: dict[str, torch.Tensor]):
        _, new_log_prob, entropy, new_value = self(batch['observations'], batch['actions'])
        log_ratio = new_log_prob - batch['log_probs']
        ratio = torch.exp(log_ratio)

        # policy loss
        advantages = batch['advantages']
        if self.hparams.running_cfg.get('normalize_advantages', False):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        p_loss = self.policy_loss(batch['advantages'], ratio)

        # value loss
        v_loss = self.value_loss(new_value, batch['values'], batch['returns'])

        # entropy loss
        e_loss = self.entropy_loss(entropy)

        # update metrics
        self.avg_pg_loss.update(p_loss)
        self.avg_value_loss.update(v_loss)
        self.avg_ent_loss.update(e_loss)

        return p_loss + e_loss + v_loss

    def validation_step(self, *args, **kwargs):
        step = 0
        done = False
        cumulative_rew = 0
        next_obs = torch.tensor(self.val_env.reset(seed=self.hparams.running_cfg.get('seed', 42))[0], device=self.device)
        while not done:
            action = self(next_obs, greedy=True)
            next_obs, reward, done, truncate, _ = self.val_env.step(action.cpu().numpy())
            done = done or truncate
            cumulative_rew += reward
            next_obs = torch.tensor(next_obs, device=self.device)
            step += 1

        self.log('val/episode_reward', cumulative_rew.item(), prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log('val/episode_steps', step, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        return {'loss': 0}

    def on_train_epoch_end(self):
        update_interval = self.hparams.running_cfg.get('update_interval', 1)
        if (self.current_epoch + 1) % update_interval == 0:
            self.replay_buffer.clear()
            self._reset_data = True

        log_data = {
            "Loss/policy_loss": self.avg_pg_loss.compute(),
            "Loss/value_loss": self.avg_value_loss.compute(),
            "Loss/entropy_loss": self.avg_ent_loss.compute()}
        self.log_dict(log_data, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.reset_metrics()

    def on_fit_end(self):
        self.train_envs.close()
        self.val_env.close()

    def reset_metrics(self):
        self.avg_pg_loss.reset()
        self.avg_value_loss.reset()
        self.avg_ent_loss.reset()

    def configure_optimizers(self):
        lr = self.hparams.optim_cfg.get('lr', 1e-4)
        return torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4)

    def _load_loss(self, loss_cfg: dict):
        self.policy_loss = policy_loss_factory(**loss_cfg['policy_loss'])
        self.value_loss = value_loss_factory(**loss_cfg['value_loss'])
        self.entropy_loss = entropy_loss_factory(**loss_cfg['entropy_loss'])

    @torch.no_grad()
    def _load_data(self):
        self.eval()

        env_eps = 0
        env_rew = 0
        env_eps_len = 0

        num_steps = self.hparams.data_cfg.get('total_timestep_n', 1000)

        gamma = self.hparams.running_cfg.get('gamma', 0.99)
        gae_lambda = self.hparams.running_cfg.get('gae_lambda', 0.95)

        next_observations = torch.tensor(self.train_envs.reset()[0], device=self.device)
        next_dones = torch.zeros(self.train_envs.num_envs, device=self.device)
        for step in range(0, num_steps):
            self.observations_buffer[step] = next_observations
            self.dones_buffer[step] = next_dones

            actions, log_probs, _, values = self(next_observations)
            self.values_buffer[step] = values.flatten()
            self.actions_buffer[step] = actions
            self.log_probs_buffer[step] = log_probs

            next_observations, rewards, dones, truncateds, info = self.train_envs.step(actions.cpu().numpy())
            dones = torch.logical_or(torch.tensor(dones), torch.tensor(truncateds))
            self.rewards_buffer[step] = torch.tensor(rewards.astype(np.float32), device=self.device).view(-1)

            next_observations = torch.tensor(next_observations, device=self.device)
            next_dones = dones.to(self.device)

            episode = info.get('episode', None)
            if episode:
                for r, l in zip(episode['r'], episode['l']):
                    env_eps += 1
                    env_rew += r
                    env_eps_len += l

        self.logger.log_metrics({"env/mean_episodes_reward": env_rew / (env_eps + 1e-8),
                                 "env/mean_episodes_length": env_eps_len / (env_eps + 1e-8)},
                                self.current_epoch)

        returns, advantages = self.ppo_model.estimate_returns_and_advantages(self.rewards_buffer, self.values_buffer,
                                                                             self.dones_buffer, next_observations,
                                                                             next_dones, num_steps, gamma, gae_lambda)

        obs_data = self.observations_buffer.reshape((-1,) + self.observation_shape)
        log_prob_data = self.log_probs_buffer.reshape(-1)
        act_data = self.actions_buffer.reshape((-1,) + self.action_shape)
        adv_data = advantages.reshape(-1)
        ret_data = returns.reshape(-1)
        val_data = self.values_buffer.reshape(-1)
        for obs, log_prob, act, adv, ret, val in zip(obs_data, log_prob_data, act_data, adv_data, ret_data, val_data):
            self.replay_buffer.append(PPOExperience(
                observations=obs,
                actions=act,
                values=val,
                returns=ret,
                advantages=adv,
                log_probs=log_prob))

        self.train()

    @classmethod
    def _load_env(cls, env_cfg: dict, data_cfg: dict, log_root: str = None, seed: int = 42) -> gym.vector.SyncVectorEnv:
        def make_env(env_type, env_cfg: dict, idx: int):
            def thunk():
                env = gym.make(env_type, **env_cfg)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                if idx == 0 and log_root:
                    env = gym.wrappers.RecordVideo(env, log_root)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env
            return thunk

        curr_env_cfg = deepcopy(env_cfg)
        batch_size = data_cfg.get('batch_size', 1)
        env_type = curr_env_cfg.pop('type')
        envs = gym.vector.SyncVectorEnv([make_env(env_type, curr_env_cfg, _) for _ in range(batch_size)])
        return envs

    @classmethod
    def _load_replay_buffer(cls, data_cfg: dict) -> ReplayBuffer:
        total_timestep_n = data_cfg.get('total_timestep_n', 1000)
        buffer = ReplayBuffer(total_timestep_n)
        return buffer

    def train_dataloader(self):
        batch_size = self.hparams.data_cfg.get('batch_size', 1)
        sample_timestep_n = max(self.hparams.data_cfg['sample_timestep_n'],
                                self.hparams.running_cfg['update_steps'] * batch_size)
        dataset = PPODataset(self.replay_buffer, sample_timestep_n)
        return DataLoader(dataset, batch_size=batch_size)

    def val_dataloader(self):

        class FakeDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                return self.data[index]

        dataset = FakeDataset(list([i for i in range(1)]))
        return DataLoader(dataset, batch_size=1)
