import cv2
import gym
import numpy as np
from gym.spaces import Box

class EnvWrapper:
    def __init__(self, env_name, debug=False):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 4))
        self.frame_num = 0
        self.frames = np.zeros((84, 84, 4), dtype=np.uint8)
        self.debug = debug

        if self.debug:
            cv2.startWindowThread()
            cv2.namedWindow("Game")

    def step(self, a):
        ob, reward, done, info = self.env.step(a)
        return self.process_frame(ob), reward, done, info

    def reset(self):
        self.frame_num = 0
        return self.process_frame(self.env.reset())

    def render(self):
        return self.env.render()

    def process_frame(self, frame):
        state_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        state_resized = cv2.resize(state_gray, (84, 110))
        gray_final = state_resized[16:100, :]

        if self.frame_num == 0:
            self.frames[:, :, 0] = gray_final
            self.frames[:, :, 1] = gray_final
            self.frames[:, :, 2] = gray_final
            self.frames[:, :, 3] = gray_final
        else:
            self.frames[:, :, 3] = self.frames[:, :, 2]
            self.frames[:, :, 2] = self.frames[:, :, 1]
            self.frames[:, :, 1] = self.frames[:, :, 0]
            self.frames[:, :, 0] = gray_final

        self.frame_num += 1

        if self.debug:
            cv2.imshow('Game', gray_final)

        return self.frames.copy()

class CustomLambda(Layer):
    def __init__(self, **kwargs):
        super(CustomLambda, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.identity(inputs)

class ReplayMemoryFast:
    def __init__(self, memory_size, minibatch_size):
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size

        self.experience = [None] * self.memory_size
        self.current_index = 0
        self.size = 0

    def store(self, observation, action, reward, newobservation, is_terminal):
        self.experience[self.current_index] = (observation, action, reward, newobservation, is_terminal)
        self.current_index += 1
        self.size = min(self.size + 1, self.memory_size)

        if self.current_index >= self.memory_size:
            self.current_index -= self.memory_size

    def sample(self):
        if self.size < self.minibatch_size:
            return []

        samples_index = np.floor(np.random.random((self.minibatch_size,)) * self.size)
        samples = [self.experience[int(i)] for i in samples_index]

        return samples
