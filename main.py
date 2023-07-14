import numpy as np
import random
import gym
from gym.spaces import Box
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer
from model import QNetworkDueling
from utils import EnvWrapper, ReplayMemoryFast, CustomLambda, DQN

env_wrapper = EnvWrapper("CarRacing-v2", debug=True)

state_size = (84, 84, 4)
action_size = env_wrapper.action_space.shape[0]

session = tf.compat.v1.InteractiveSession()

agent = DQN(state_size=state_size,
            action_size=action_size,
            session=session,
            summary_writer=None,
            exploration_period=1000000,
            minibatch_size=32,
            discount_factor=0.99,
            experience_replay_buffer=1000000,
            target_qnet_update_frequency=20000,
            initial_exploration_epsilon=1.0,
            final_exploration_epsilon=0.1,
            reward_clipping=1.0)

session.run(tf.compat.v1.global_variables_initializer())

for episode in range(1, 11):
    state = env_wrapper.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.action(state, training=True)
        next_state, reward, done, _ = env_wrapper.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

    print("Episode:", episode, "Total Reward:", total_reward)

env_wrapper.env.close()
