import torch

from tqc import DEVICE
import numpy as np

def eval_policy(policy, eval_env, max_episode_steps, eval_episodes=10):
    policy.eval()
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        t = 0
        while not done and t < max_episode_steps:
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            t += 1
    avg_reward /= eval_episodes
    policy.train()
    return avg_reward


def eval_policy_save_transition(policy, eval_env, max_episode_steps, eval_episodes=10):
    policy.eval()
    avg_reward = 0.
    states_epi = []
    actions_epi = []
    rewards_epi = []
    dones_epi = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        t = 0
        states = []
        actions = []
        rewards = []
        dones = []
        while not done and t < max_episode_steps:
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            avg_reward += reward
            t += 1
        states_epi.append(states)
        actions_epi.append(actions)
        rewards_epi.append(rewards)
        dones_epi.append(dones)

    avg_reward /= eval_episodes
    policy.train()

    return avg_reward, (states_epi, actions_epi, rewards_epi, dones_epi,)


def quantile_huber_loss_f(quantiles, samples):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=DEVICE).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss
