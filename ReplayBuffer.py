import collections
import numpy as np


class ReplayBuffer:
    def __init__(self, size):
        self.state_buffer = collections.deque(maxlen=size)
        self.action_buffer = collections.deque(maxlen=size)
        self.next_state_buffer = collections.deque(maxlen=size)
        self.reward_buffer = collections.deque(maxlen=size)
        self.done_buffer = collections.deque(maxlen=size)

    def __len__(self):
        return len(self.state_buffer)

    def append(self, state, action, reward, done, next_state):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.next_state_buffer.append(next_state)

    def sample(self, batch_size):
        # set "replace" to false so that we do not sample the same experience twice
        samples = np.random.choice(len(self.state_buffer), batch_size, replace=False)
        states = [self.state_buffer[i] for i in samples]
        actions = [self.action_buffer[i] for i in samples]
        rewards = [self.reward_buffer[i] for i in samples]
        dones = [self.done_buffer[i] for i in samples]
        next_states = [self.next_state_buffer[i] for i in samples]
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)
