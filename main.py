import gym
from torch import nn

import Networks
from SAC import SAC

if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v4", render_mode="human")
    print("Hello World!")

    env.action_space.seed(42)

    observation, info = env.reset(seed=42)

    actor = Networks.SquashedGaussianMLPActor(4, 1, (256,256), nn.ReLU, 3)
    critic1 = Networks.MLPQFunction(4, 1, (256,256), nn.ReLU)
    critic2 = Networks.MLPQFunction(4, 1, (256,256), nn.ReLU)

    sac = SAC(actor, critic1, critic2, 4, 1)
    sac.train(env, epochs=10000, duration=15, batch_size=50, update_every=5)
