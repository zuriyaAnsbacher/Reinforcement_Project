import nni
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from statistics import mean

from Models import Actor, Critic
from memory_data import SamplesMemory
from utils import create_plot
from Noise import OU_Noise1, OU_Noise2

class DDPGAgent:
    def __init__(self, state_size, action_size, batch_size, gamma, lr_actor, lr_critic, tau,
                 max_eps, min_eps, eps_decay, hidden_actor, hidden_critic, memory_size, action_noise, device):

        self.model_name = 'DDPG'

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.tau = tau
        self.cur_eps = max_eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay

        self.device = device

        self.memory = SamplesMemory(memory_size, device)
        self.batch_size = batch_size
        if action_noise == 'ou1':
            self.noise = OU_Noise1(self.action_size)
        elif action_noise == 'ou2':
            self.noise = OU_Noise2(self.action_size)

        self.hidd1_actor = hidden_actor[0]
        self.hidd2_actor = hidden_actor[1]
        self.actor = Actor(self.state_size, self.action_size, self.hidd1_actor, self.hidd2_actor).to(device)  # todo: weight_init
        self.actor_target = Actor(self.state_size, self.action_size, self.hidd1_actor, self.hidd2_actor).to(device)  # todo: weight_init
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)

        self.hidd1_critic = hidden_critic[0]
        self.hidd2_critic = hidden_critic[1]
        self.critic = Critic(self.state_size, self.action_size, self.hidd1_critic, self.hidd2_critic).to(device)  # todo: weight_init
        self.critic_target = Critic(self.state_size, self.action_size, self.hidd1_critic, self.hidd2_critic).to(device)  # todo: weight_init
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)  #todo: add weight decay?
        self.copy_params_to_target()

        self.critic_loss_func = nn.MSELoss()
    
    def choose_action(self):
        pass

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        action += self.cur_eps * self.noise.add_noise()
        action = np.clip(action, -1, 1)

        return action

    def decrement_epsilon(self):
        return max(self.cur_eps - self.eps_decay, self.min_eps)

    def save(self, file_path):
        torch.save(self.actor.state_dict(), file_path)  # todo: +'_actor.pth'

    def load(self, file_path):
        self.actor.load_state_dict(torch.load(file_path, map_location=self.device))  # todo: + '.pth'

    def set_eval(self):
        self.actor.eval()

    def copy_params_to_target(self, curr_tau=1.0):
        for net, target_net in zip([self.actor, self.critic], [self.actor_target, self.critic_target]):
            for params, target_params in zip(net.parameters(), target_net.parameters()):
                target_params.data.copy_(params.data * curr_tau + target_params.data * (1.0 - curr_tau))

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.get_batch(self.batch_size, continuous_action=True)

        # q values
        next_action = self.actor_target(next_states)
        q_val = self.critic(states, actions)
        next_q = self.critic_target(next_states, next_action)

        # formula (bellman equation)
        q_prime = rewards + (self.gamma * next_q * (1 - dones))

        # critic: loss and optimizer
        critic_loss = self.critic_loss_func(q_val, q_prime)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # actor: loss and optimizer
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.copy_params_to_target(curr_tau=self.tau)

        self.cur_eps = self.decrement_epsilon()
        self.noise.reset()

    def train(self, env, paths, episodes_num, steps_num, learn_freq, update_factor, record=False, record_freq=50, is_nni=False):
        scores_list = []

        for episode in range(1, episodes_num + 1):
            print(f'Episode: {episode} \nSteps:')
            state = env.reset()
            self.noise.reset()

            score = 0
            for step in range(1, steps_num + 1):
                if step % 50 == 0:
                    print(step, end=' ')
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                self.memory.add_sample(state, action, reward, next_state, done)
                if len(self.memory.memory_buffer) > self.batch_size and step % learn_freq == 0:
                    for _ in range(update_factor):
                        self.learn()

                score += reward
                state = next_state

                # if len(self.memory.memory_buffer) > self.batch_size and step % learn_freq == 0:
                #     self.learn()

                if done:
                    # epsilon decrement? or it happened every step?
                    break

            scores_list.append(score)
            avg_reward = mean(scores_list[-100:]) if len(scores_list) >= 100 else mean(scores_list)
            print(f'\nScore: {score}, Mean average until now (up to last 100): '
                  f'{avg_reward}')

            goal = 200 if not is_nni else 210
            if avg_reward > goal and len(scores_list) >= 100:
                print(f'Mission accomplished! average reward: {avg_reward} in episode {episode}.')

                if not is_nni:
                    filename = ''.join([self.model_name, '_params', paths['args_set_num']])
                    self.save(''.join([paths['saved_model_path'], filename, '.pth']))
                    create_plot(scores_list, self.model_name,
                                ''.join([paths['saved_plots_path'], 'train_', filename, '.png']))

                break

        if is_nni:
            nni.report_final_result(episode)

        env.close()



