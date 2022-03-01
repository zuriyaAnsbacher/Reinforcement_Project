import nni
import gym
import json
import argparse
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join('../..', 'Reinforcement_Project')))

from Agents import DQNAgent, DoubleDQNAgent, DuelingDDQNAgent
from utils import set_seed, quantize_space, test


# todo: adjust the main to all kinds of DQN
def main():

    parser = argparse.ArgumentParser()
    # args that change each run
    parser.add_argument('--model', choices=['dqn', 'ddqn', 'd3qn'], default='ddqn')
    parser.add_argument('--train', default=True, help='train agent')
    parser.add_argument('--use_nni_params', default=False, help='if true, get params from json file')
    parser.add_argument('--set_num', type=str, default='1')

    # args that usually stay fixed
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--episodes', type=int, default=800, help='number of episodes in train')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--max_eps', type=float, default=1.0)
    parser.add_argument('--min_eps', type=float, default=0.01)

    args = parser.parse_args()

    if args.use_nni_params:
        with open('best_params.json') as json_file:
            best_params = json.load(json_file)[f'{args.model}_params{args.set_num}']
    else:
        best_params = {"batch_size": 128,
                       "gamma": 0.98,
                       "lr": 0.003,
                       "max_steps": 1000,
                       "target_update": 400,
                       "learn_freq": 4,
                       "eps_decay": 0.96,
                       "hidden_layers_size":  [64, 64]
                       }

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(device)

    paths = {'saved_model_path': './results/trained_models/',
             'saved_plots_path': './results/plots/',
             'args_set_num': args.set_num}

    env = gym.make('LunarLanderContinuous-v2')
    set_seed(env, seed=0)

    state_size = 8  # state vector dim
    action_space = quantize_space(actions_range=[(-1, 1), (-1, 1)], bins=[5, 5])

    agent_params = [state_size, len(action_space), action_space, args.memory_size, args.max_eps, args.min_eps,
                    best_params['batch_size'], best_params['lr'], best_params['gamma'], best_params['eps_decay'],
                    best_params['target_update'], best_params['hidden_layers_size'], device]

    if args.model == 'dqn':
        agent = DQNAgent(*agent_params)
    elif args.model == 'ddqn':
        agent = DoubleDQNAgent(*agent_params)  # todo: check
    elif args.model == 'd3qn':
        agent = DuelingDDQNAgent(*agent_params)  # todo: check

    if args.train:
        train_params = [env, paths, args.episodes, best_params['max_steps'], best_params['learn_freq']]
        agent.train(*train_params)
    else:
        test(agent, env, paths, agent.model_name)


def main_nni():
    # fixed params
    params = {'cuda_device': 0,
              'model': 'dqn',
              'train': True,
              'memory_size': 1000000,
              'episodes': 1000,
              'max_eps': 1.0,
              'min_eps': 0.01}

    # nni params
    nni_params = nni.get_next_parameter()
    params['batch_size'] = int(nni_params['batch_size'])
    params['gamma'] = float(nni_params['gamma'])
    params['lr'] = float(nni_params['lr'])
    params['target_update'] = int(nni_params['target_update'])
    params['learn_freq'] = int(nni_params['learn_freq'])
    params['eps_decay'] = float(nni_params['eps_decay'])
    params['max_steps'] = int(nni_params['max_steps'])
    params['hidd1_size'] = int(nni_params['hidd1_size'])
    params['hidd2_size'] = int(nni_params['hidd2_size'])

    device = torch.device(f"cuda:{params['cuda_device']}" if torch.cuda.is_available() else "cpu")
    print(device)

    paths = None

    env = gym.make('LunarLanderContinuous-v2')
    set_seed(env, seed=0)

    state_size = 8  # state vector dim
    action_space = quantize_space(actions_range=[(-1, 1), (-1, 1)], bins=[5, 5])

    agent_params = [state_size, len(action_space), action_space, params['batch_size'],
                    params['lr'], params['gamma'], params['memory_size'], params['max_eps'],
                    params['min_eps'], params['eps_decay'], params['target_update'],
                    params['hidd1_size'], params['hidd2_size'], device]

    if params['model'] == 'dqn':
        agent = DQNAgent(*agent_params)

    if params['train']:
        train_params = [env, paths, params['episodes'], params['max_steps'], params['learn_freq']]
        agent.train(*train_params, is_nni=True)


if __name__ == "__main__":
    main()
    # main_nni()
