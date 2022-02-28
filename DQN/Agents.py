import nni
import numpy as np
from statistics import mean
import torch
import torch.optim as optim
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.abspath(os.path.join('../..', 'Reinforcement_Project')))

from Models import DQN
from memory_data import SamplesMemory


class Agent:
    def __init__(self, action_space, batch_size, gamma, memory_size,
                 max_eps, min_eps, eps_decay, device):

        self.gamma = gamma
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.eps_decay = eps_decay
        self.cur_eps = max_eps
        self.policy_net = None
        self.output_size = None

        self.memory = SamplesMemory(memory_size, device)
        self.batch_size = batch_size

        self.idx2action = {i: action for i, action in enumerate(action_space)}
        self.action2idx = {action: i for i, action in enumerate(action_space)}

        self.device = device

    def get_action(self, state):
        # epsilon greedy
        if np.random.random() < self.cur_eps:  # exploration
            return self.idx2action[np.random.choice(self.output_size)]
        else:  # exploitation
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=self.device)
                return self.idx2action[self.policy_net(state).argmax().item()]

    def decrement_epsilon(self):
        return max(self.cur_eps * self.eps_decay, self.min_eps)

    def learn(self):
        pass

    def train(self, env, episodes_num, steps_num, learn_freq, is_nni=False):
        losses = []
        scores_list = []

        self.policy_net.train()
        for episode in range(1, episodes_num + 1):
            print(f'Episode: {episode} \nSteps:')
            state = env.reset()

            score = 0
            for step in range(1, steps_num + 1):
                if step % 50 == 0:
                    print(step, end=' ')
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                action = self.action2idx[action]

                self.memory.add_sample(state, action, reward, next_state, done)
                score += reward
                state = next_state

                if len(self.memory.memory_buffer) >= self.batch_size and step % learn_freq == 0:
                    losses.append(self.learn())

                if done:
                    self.cur_eps = self.decrement_epsilon()
                    break

            scores_list.append(score)
            avg_reward = mean(scores_list[-100:]) if len(scores_list) >= 100 else mean(scores_list)
            print(f'\nScore: {score}, Mean average until now (up to last 100): '
                  f'{avg_reward}')

            if is_nni:
                if episode % 20 == 0:
                    nni.report_intermediate_result(episode)

            if avg_reward > 200 and len(scores_list) >= 100:
                print(f'Mission accomplished! average reward: {avg_reward} in episode {episode}.')
                break

                # todo: save model
                # todo: plots

        if is_nni:
            nni.report_final_result(episode)

        env.close()


class DQNAgent(Agent):
    def __init__(self, input_size, output_size, action_space, batch_size, lr, gamma, memory_size,
                 max_eps, min_eps, eps_decay, target_update, hidd1_size, hidd2_size, device):
        super(DQNAgent, self).__init__(action_space, batch_size, gamma, memory_size, max_eps,
                                       min_eps, eps_decay, device)

        self.hidd1 = hidd1_size
        self.hidd2 = hidd2_size
        self.output_size = output_size
        self.policy_net = DQN(input_size, output_size, self.hidd1, self.hidd2).to(device)
        self.target_net = DQN(input_size, output_size, self.hidd1, self.hidd2).to(device)  # copy.deepcopy?
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.target_update = target_update
        self.update = 0

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.get_batch(self.batch_size)

        q_val = self.policy_net(states).gather(1, actions)
        next_q_val = self.target_net(next_states).max(1, keepdim=True)[0].detach()

        # formula
        target = (rewards + self.gamma * next_q_val * (1 - dones)).to(self.device)

        # loss and optimizer
        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_val, target)  # use this loss or MSE?
        loss.backward()
        self.optimizer.step()
        self.update += 1

        # every target_update num, load weights from policy_net to target_net
        if self.update % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

