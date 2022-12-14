import gym
import torch
from torch import nn

import MountainCarEnv
import Networks
from SAC import SAC


def eval(env: gym.Env, duration):
    state, _ = env.reset()
    env.render_mode = "human"
    total = 0
    while True:
        a, _ = [1], 0

        next_state, reward, done, _, _ = env.step(a)
        total += reward
        state = next_state

        if done:
            break

    print(total)
    # env.render_mode = "machine"
    return total


if __name__ == "__main__":
    env = MountainCarEnv.Continuous_MountainCarEnv(duration=100, render_mode="machine", const_reward=False)
    print("Hello World!")

    env.action_space.seed(42)

    # eval(env, 100)

    # exit()

    observation, info = env.reset(seed=42)

    actor = Networks.SquashedGaussianMLPActor(2, 1, (256,256), nn.ReLU, 3)
    critic1 = Networks.MLPQFunction(2, 1, (256,256), nn.ReLU)
    critic2 = Networks.MLPQFunction(2, 1, (256,256), nn.ReLU)

    sac = SAC(actor, critic1, critic2, 2, 1)
    sac.train(env, epochs=100000, duration=15, batch_size=500, update_every=50)
