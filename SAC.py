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

    def save(self, fp):
        torch.save(self.actor.state_dict(), f"{fp}_actor.pt")
        torch.save(self.critic1.state_dict(), f"{fp}_critic1.pt")
        torch.save(self.critic2.state_dict(), f"{fp}_critic2.pt")


class SAC:
    def __init__(self, actor, critic1, critic2, obs_size, action_size, buffer_size=1000,
                 alpha=0.2, gamma=0.99):
        self.p_optim = None
        self.q_optim = None

        self.current = ActorCritic(actor, critic1, critic2)
        self.target = deepcopy(self.current)

        self.buffer = ReplayBuffer(obs_size, action_size, buffer_size)

        self.learn_alpha = isinstance(alpha, str)
        if self.learn_alpha:
            self.alpha = torch.tensor(0.3, dtype=torch.float32, requires_grad=True)  # some initial value, idk how to initialize
            self.alpha_optim = None
            self.target_entropy = -1.0 * action_size  # Anas set at this value, idk why
        else:
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

        alpha_loss = 0
        if self.learn_alpha:
            alpha_loss = -(self.alpha * (log_prob + self.target_entropy).detach()).mean()

        return loss, alpha_loss

    def update(self, data, polyak=0.995):
        # First run one gradient descent step for Q1 and Q2
        self.q_optim.zero_grad()
        loss_q = self.q_loss(data)
        loss_q.backward()
        self.q_optim.step()

        # Freeze Q-networks so we don't waste computational effort
        # computing gradients for them during the policy learning step.
        for param in self.q_params:
            param.requires_grad = False

        # Next run one gradient descent step for pi.
        self.p_optim.zero_grad()
        loss_pi, loss_alpha = self.p_loss(data)
        loss_pi.backward()
        self.p_optim.step()

        if self.learn_alpha:
            self.alpha_optim.zero_grad()
            loss_alpha.backward()
            self.alpha_optim.step()

        # Unfreeze Q-networks so you can optimize it at next step.
        for param in self.q_params:
            param.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for curr_param, targ_param in zip(self.current.parameters(), self.target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                targ_param.data.mul_(polyak)
                targ_param.data.add_((1 - polyak) * curr_param.data)

    def eval(self, env: gym.Env, duration, mode='machine'):
        state, _ = env.reset()
        self.current.actor.eval()
        env.render_mode = mode
        total = 0
        while True:
            a, _ = self.current.actor(torch.as_tensor(state, dtype=torch.float32))
            a = a.detach()

            next_state, reward, done, _, _ = env.step(a)
            total += reward
            state = next_state

            if done:
                break

        env.render_mode = "machine"
        return total

    def train(self, env: gym.Env, epochs, duration=96, update_every=50, lr=1e-3, batch_size=96):
        print('training')

        self.p_optim = torch.optim.Adam(self.current.actor.parameters(), lr=lr)
        self.q_optim = torch.optim.Adam(self.q_params, lr=lr)

        if self.learn_alpha:
            self.alpha_optim = torch.optim.Adam([self.alpha], lr=lr)

        self.current.actor.train()

        for epoch in range(epochs):
            state, _ = env.reset()  # start from beginning

            while True:
                a, _ = self.current.actor(torch.as_tensor(state, dtype=torch.float32))
                a = a.detach()

                next_state, reward, done, _, _ = env.step(a)

                self.buffer.store(state, a, reward, next_state, done)

                state = next_state

                if done:
                    break

            if epoch % update_every == 0:
                for _ in range(update_every):
                    b = self.buffer.sample_batch(batch_size=batch_size)
                    # b = self.debug_make_the_same()
                    self.update(b)

                print(f"Epoch {epoch}: {self.eval(env, duration)}")
                self.current.actor.train()

            pass
        x = input("Ready to see the final product?")
        for _ in range(10):
            self.eval(env, duration=100, mode="human")

    def debug_make_the_same(self):
        self.current.actor.load_state_dict(torch.load("./init_model_actor.pt"))
        self.current.critic1.load_state_dict(torch.load("./init_model_critic1.pt"))
        self.current.critic2.load_state_dict(torch.load("./init_model_critic2.pt"))

        self.target = deepcopy(self.current)
        import pickle
        data = pickle.load(open("./spinningup/data.pkl", "rb"))
        data['state'] = data.pop("obs")
        data['next_state'] = data.pop("obs2")
        data['action'] = data.pop("act")
        data['reward'] = data.pop("rew")
        data['done'] = data.pop("done")  # re-order keys so they print the same

        return data
