import itertools
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    From OpenAI Spinning up
    """

    def __init__(self, obs_dim, act_dim, size):
        self.states = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.idx, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.states[self.idx] = obs
        self.next_states[self.idx] = next_obs
        self.actions[self.idx] = act
        self.rewards[self.idx] = rew
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.max_size  # move index up, loop back to 0 if over size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.states[idxs],
                     next_state=self.next_states[idxs],
                     action=self.actions[idxs],
                     reward=self.rewards[idxs],
                     done=self.dones[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class ActorCritic(nn.Module):
    def __init__(self, actor, critic1, critic2):
        super().__init__()
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2

    def act(self, state):
        with torch.no_grad():
            a, _ = self.actor(state)
            return a.numpy()


class SAC:
    def __init__(self, actor, critic1, critic2, obs_size, action_size, buffer_size=1000,
                 alpha=0.2, gamma=0.99):
        self.p_optim = None
        self.q_optim = None

        self.current = ActorCritic(actor, critic1, critic2)
        self.target = deepcopy(self.current)

        self.buffer = ReplayBuffer(obs_size, action_size, buffer_size)

        self.alpha = alpha
        self.gamma = gamma

        self.q_params = itertools.chain(self.current.critic1.parameters(), self.current.critic2.parameters())

    def q_loss(self, batch):
        state, action, reward, next_state, done = batch['state'], batch['action'], batch['reward'], batch['next_state'], \
        batch['done']

        with torch.no_grad():
            next_action, log_prob = self.current.actor(next_state)
            q1 = self.target.critic1(next_state, next_action)
            q2 = self.target.critic2(next_state, next_action)
            q = torch.min(q1, q2)

        y = reward + self.gamma * (1 - done) * (q - self.alpha * log_prob)

        # loss for each critic
        l1 = torch.pow(self.current.critic1(state, action) - y, 2).mean()
        l2 = torch.pow(self.current.critic2(state, action) - y, 2).mean()

        return l1 + l2  # expected value = (l1 + l2)/2, but we don't need to divide by 2

    def p_loss(self, batch):
        state = batch['state']

        action, log_prob = self.current.actor(state)
        q1 = self.current.critic1(state, action)
        q2 = self.current.critic2(state, action)
        q = torch.min(q1, q2)

        loss = (-q + self.alpha * log_prob).mean()  # negative of loss function, to turn max into min
        # (for optimizer.step which assumes minimization)

        return loss

    def update(self, data, polyak=0.005):
        # First run one gradient descent step for Q1 and Q2
        self.q_optim.zero_grad()
        loss_q = self.q_loss(data)
        loss_q.backward()
        self.q_optim.step()

        # Freeze Q-networks so we don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.p_optim.zero_grad()
        loss_pi = self.p_loss(data)
        loss_pi.backward()
        self.p_optim.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.current.parameters(), self.target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def eval(self, env: gym.Env, duration):
        state, _ = env.reset()
        self.current.actor.eval()
        total = 0
        for t in range(duration):
            a, _ = self.current.actor(torch.as_tensor(state, dtype=torch.float32))
            a = a.detach()

            next_state, reward, done, _, _ = env.step(a)
            total += reward
            state = next_state

            if done:
                break

        return total

    def train(self, env: gym.Env, epochs, duration=96, update_every=50, lr=1e-3, batch_size=96):
        print('training')

        self.p_optim = torch.optim.Adam(self.current.actor.parameters(), lr=lr)
        self.q_optim = torch.optim.Adam(self.q_params, lr=lr)

        self.current.actor.train()

        for epoch in range(epochs):
            state, _ = env.reset()  # start from beginning

            for t in range(duration):
                a, _ = self.current.actor(torch.as_tensor(state, dtype=torch.float32))
                a = a.detach()

                next_state, reward, done, _, _ = env.step(a)

                reward = int(t == duration-1)

                self.buffer.store(state, a, reward, next_state, done)

                state = next_state

                if done:
                    break

            if epoch % update_every == 0:
                for _ in range(update_every):
                    b = self.buffer.sample_batch(batch_size=batch_size)
                    self.update(b)

                print(f"Epoch {epoch}: {self.eval(env, duration)}")
                self.current.actor.train()

            pass
