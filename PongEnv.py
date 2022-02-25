import gym
import numpy as np
import cv2


class skip4Frames(gym.Wrapper):
    def __init__(self, env):
        super(skip4Frames, self).__init__(env)

    def step(self, action):
        total_rewards = 0.0
        done = False
        next_state = None
        info = None
        for _ in range(4):
            next_state, reward, done, info = self.env.step(action)
            total_rewards += reward
            if done:
                break
        return next_state, total_rewards, done, info

    def reset(self):
        return self.env.reset()


class ProcessFrames(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    @staticmethod
    def rgb2grayscale(img):
        return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

    @staticmethod
    def process_frame(frame):
        new_frame = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        new_frame = ProcessFrames.rgb2grayscale(new_frame)
        resized_screen = cv2.resize(new_frame, (84, 110), interpolation=cv2.INTER_AREA)
        cropped_frame = resized_screen[18:102, :]
        cropped_frame = np.reshape(cropped_frame, [84, 84, 1])
        return cropped_frame.astype(np.uint8)

    def observation(self, observation):
        processed_frame = ProcessFrames.process_frame(observation)
        return np.array(processed_frame).astype(np.float32) / 255.0


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


def make():
    env = gym.make("PongNoFrameskip-v4")
    env = skip4Frames(env)
    env = ProcessFrames(env)
    return ImageToPyTorch(env)
