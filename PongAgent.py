import ReplayBuffer
import PongEnv
import NeuralNetwork
import torch
import numpy as np


class PongAgent:
    def __init__(self, GAMMA, ALPHA, EPSILON_init, min_EPSILON, EPSILON_decay, batch_size, replay_mem_size,
                 sync_target_breakpoint, num_of_episodes, load_saved_weights):
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.EPSILON = EPSILON_init
        self.min_EPSILON = min_EPSILON
        self.EPSILON_decay = EPSILON_decay
        self.batch_size = batch_size
        self.replay_mem_size = replay_mem_size
        self.sync_target_breakpoint = sync_target_breakpoint
        self.num_of_episodes = num_of_episodes
        self.load_saved_weights = load_saved_weights
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU for calculations if available
        self.replay_buffer = ReplayBuffer.ReplayBuffer(replay_mem_size)
        self.env = PongEnv.make()
        # initializing Q networks
        self.network_q = NeuralNetwork.DQN(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.target_network_q = NeuralNetwork.DQN(self.env.observation_space.shape, self.env.action_space.n).to(
            self.device)
        self.learn()

    def policy(self, state):
        if np.random.random() < self.EPSILON:
            action = self.env.action_space.sample()
        else:
            state_v = torch.tensor(np.array([state], copy=False)).to(self.device)
            q_vals = self.network_q(state_v)
            action = int(torch.max(q_vals, dim=1)[1].item())
        return action

    def learn(self):
        adam = torch.optim.Adam(self.network_q.parameters(), self.ALPHA)  # initialize optimizer
        loss_function = torch.nn.MSELoss()
        step = 0
        scores = []

        for episode in range(self.num_of_episodes):
            score = 0.0
            state = self.env.reset()
            done = False

            while not done:
                step += 1

                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                if not self.load_saved_weights:
                    self.replay_buffer.append(state, action, reward, done, next_state)

                    if len(self.replay_buffer) >= self.replay_mem_size:
                        # decay EPSILON
                        if self.EPSILON > self.min_EPSILON:
                            self.EPSILON = self.EPSILON - self.EPSILON_decay
                        else:
                            self.EPSILON = self.min_EPSILON

                        states_sample, actions_sample, rewards_sample, dones_sample, next_states_sample = \
                            self.replay_buffer.sample(self.batch_size)

                        # calculate the loss
                        states_sample_tensor = torch.tensor(states_sample).to(self.device)
                        actions_sample_tensor = torch.tensor(actions_sample).to(self.device)
                        rewards_sample_tensor = torch.tensor(rewards_sample).to(self.device)
                        next_states_sample_tensor = torch.tensor(next_states_sample).to(self.device)

                        actions_sample_tensor = actions_sample_tensor.type(torch.int64)
                        state_action_values = self.network_q(states_sample_tensor).gather(1, actions_sample_tensor.unsqueeze(-1)).squeeze(-1)
                        next_state_values = self.target_network_q(next_states_sample_tensor).max(1)[0]
                        for i in range(len(dones_sample)):
                            if dones_sample[i]:
                                next_state_values[i] = 0.0
                        next_state_values = next_state_values.detach()
                        y_vector = rewards_sample_tensor + self.GAMMA * next_state_values
                        loss = loss_function(state_action_values, y_vector)  # calculate MEAN SQUARED ERROR loss

                        # update network parameters
                        adam.zero_grad()
                        loss.backward()
                        adam.step()

                        # copy weights from network_q to target_network_q
                        if step % self.sync_target_breakpoint == 0:
                            self.target_network_q.load_state_dict(self.network_q.state_dict())
                else:
                    self.env.render()

                state = next_state

            scores.append(score)

            average_score = np.mean(scores[-100:])

            print('episode: ', episode, 'score: ', score, 'average score: %.2f' % average_score,
                  'epsilon: %.2f' % self.EPSILON, 'steps: ', step)


PongAgent(GAMMA=0.99, ALPHA=1e-4, EPSILON_init=1.0, min_EPSILON=0.02, EPSILON_decay=1e-5, batch_size=32,
          replay_mem_size=10000, sync_target_breakpoint=1000, num_of_episodes=10000, load_saved_weights=False)
