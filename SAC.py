import itertools
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn
import wandb


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
    def __init__(self, actor, critic1, critic2, obs_size, action_size, buffer_size=1e6,
                 alpha=0.2, gamma=0.99):
        self.p_optim = None
        self.q_optim = None

        self.current = ActorCritic(actor, critic1, critic2)
        self.target = deepcopy(self.current)

        self.buffer = ReplayBuffer(obs_size, action_size, int(buffer_size))

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

        return l1 + l2, {'critic1-loss':l1, 'critic2-loss':l2, 'target':y.mean(),
                         'q1': self.current.critic1(state, action).mean(),
                         'q2': self.current.critic2(state, action).mean()}  # expected value = (l1 + l2)/2, but we don't need to divide by 2

    def p_loss(self, batch):
        state = batch['state']

        action, log_prob, action_logs = self.current.actor(state, metrics=True)
        q1 = self.current.critic1(state, action)
        q2 = self.current.critic2(state, action)
        q = torch.min(q1, q2)

        loss = (-q + self.alpha * log_prob).mean()  # negative of loss function, to turn max into min
        # (for optimizer.step which assumes minimization)

        alpha_loss = 0
        if self.learn_alpha:
            alpha_loss = -(self.alpha * (log_prob + self.target_entropy).detach()).mean()

        return loss, alpha_loss, {'q-vals': q.detach().mean(), 'log_probs': log_prob.detach().mean(), 'actor': action_logs}

    def update(self, data, polyak=0.995):
        logs = {}
        # First run one gradient descent step for Q1 and Q2
        self.q_optim.zero_grad()
        loss_q, q_logs = self.q_loss(data)
        loss_q.backward()
        self.q_optim.step()

        logs['q_loss'] = loss_q
        logs.update(q_logs)  # add critic1 and critic2 losses

        # Freeze Q-networks so we don't waste computational effort
        # computing gradients for them during the policy learning step.
        for param in self.q_params:
            param.requires_grad = False

        # Next run one gradient descent step for pi.
        self.p_optim.zero_grad()
        loss_pi, loss_alpha, p_logs = self.p_loss(data)
        loss_pi.backward()
        self.p_optim.step()

        logs['p_loss'] = loss_pi
        logs['alpha_loss'] = loss_alpha
        logs['alpha'] = self.alpha
        logs['p_logs'] = p_logs

        if self.learn_alpha:
            self.alpha_optim.zero_grad()
            loss_alpha.backward()
            self.alpha_optim.step()

        # Unfreeze Q-networks so you can optimize it at next step.
        for param in self.q_params:
            param.requires_grad = True

        # Finally, update target networks by polyak averaging.
        diff = 0
        with torch.no_grad():
            for curr_param, targ_param in zip(self.current.parameters(), self.target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                targ_param.data.mul_(polyak)
                targ_param.data.add_((1 - polyak) * curr_param.data)

                diff += torch.linalg.norm(targ_param.data - curr_param.data).sum().item()
        logs['target distance'] = diff

        return logs

    def eval(self, env: gym.Env, mode='rgb_array'):
        state, _ = env.reset()
        self.current.actor.eval()  # put it in eval mode, so actions are deterministic
        env.render_mode = mode
        total = 0

        frames = []

        while True:
            a, _ = self.current.actor(torch.as_tensor(state, dtype=torch.float32))
            a = a.detach()

            next_state, reward, done, _, d = env.step(a)
            total += reward
            state = next_state

            if 'frame' in d:
                frames.append(d['frame'])

            if done:
                break

        env.render_mode = "machine"
        self.current.actor.train()  # put it back into training mode
        return total, np.array(frames)

    def train(self, env: gym.Env, epochs, update_every=50, lr=1e-3, batch_size=96, polyak=0.995,
              log=False):
        print('training')

        self.p_optim = torch.optim.Adam(self.current.actor.parameters(), lr=lr)
        self.q_optim = torch.optim.Adam(self.q_params, lr=lr)

        if self.learn_alpha:
            self.alpha_optim = torch.optim.Adam([self.alpha], lr=lr)

        self.current.actor.train()

        total_reward = 0
        for epoch in range(epochs):
            state, _ = env.reset()  # start from beginning

            to_store = []
            while True:
                a, _ = self.current.actor(torch.as_tensor(state, dtype=torch.float32))
                a = a.detach()

                next_state, reward, done, _, _ = env.step(a)

                # to_store.append((state, a, next_state, done))

                self.buffer.store(state, a, reward, next_state, done)

                state = next_state

                total_reward += reward  # add reward for this game
                if done:
                    break


            # for tup in to_store:  # give the entire state,action sequence the reward we get at the end
            #     assert reward != 0
            #     state, a, next_state, done = tup
            #     self.buffer.store(state, a, reward, next_state, done)

            if epoch % update_every == 0:
                for _ in range(update_every * 100):
                    b = self.buffer.sample_batch(batch_size=batch_size)
                    # b = self.debug_make_the_same()
                    update_logs = self.update(b, polyak=polyak)

                test_reward, video = self.eval(env, mode='rgb_array' if log else 'machine')

                # log the average training reward, single episode test reward, loss metrics from _last_ update step
                logs = {"train-reward": total_reward/update_every, "test-reward": test_reward, "update": update_logs}
                if video.any():
                    logs['video'] = wandb.Video(video, fps=10)

                print(f"Epoch {epoch}: {test_reward}")
                self.current.actor.train()

                if log is True:
                    wandb.log(logs)

                total_reward = 0

        x = input("Ready to see the final product?")
        for _ in range(3):
            self.eval(env, mode="human")

    def debug_make_the_same(self):
        torch.manual_seed(0)
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
