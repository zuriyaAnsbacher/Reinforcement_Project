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


# for video, need to install the packages 'pyglet', 'imageio-ffmpeg'


def main():
    parser = argparse.ArgumentParser()

    # args that change each run
    parser.add_argument('--model', choices=['dqn', 'ddqn', 'd3qn'], default='dqn')
    parser.add_argument('--train', default=True, help='train agent')
    parser.add_argument('--use_nni_params', default=False, help='if true, get params from json file')
    parser.add_argument('--set_num', type=str, default='temp')

    # args that usually stay fixed
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--episodes', type=int, default=800, help='number of episodes in train')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--max_eps', type=float, default=1.0)
    parser.add_argument('--min_eps', type=float, default=0.01)
    parser.add_argument('--record', type=bool, default=True)
    parser.add_argument('--record_freq', type=int, default=15)

    args = parser.parse_args()

    if args.use_nni_params:
        with open('best_params.json') as json_file:
            best_params = json.load(json_file)[f'{args.model}_params{args.set_num}']
    else:
        best_params = {"batch_size": 256,
                       "gamma": 0.9779,
                       "lr": 0.001583,
                       "max_steps": 900,
                       "target_update": 400,
                       "learn_freq": 3,
                       "eps_decay": 0.954,
                       "hidden_layers_size": [64, 32, 32] if args.model == 'd3qn' else [256, 64]
                       }


    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(device)

    paths = {'saved_model_path': './results/trained_models/',
             'saved_plots_path': './results/plots/',
             'saved_video_path': './results/video/',
             'args_set_num': args.set_num}

    if not os.path.exists('./results'):
        os.mkdir('./results')

    for directory in ['trained_models', 'plots', 'video']:
        if not os.path.exists(f'./results/{directory}'):
            os.mkdir(f'./results/{directory}')

    env = gym.make('LunarLanderContinuous-v2')
    set_seed(env, seed=0)

    state_size = 8  # state vector dim
    action_space = quantize_space(actions_range=[(-1, 1), (-1, 1)], bins=[5, 5])

    agent_params = [state_size, len(action_space), action_space,
                    best_params['batch_size'], best_params['lr'], best_params['gamma'],
                    best_params['eps_decay'], best_params['target_update'], best_params['hidden_layers_size'],
                    args.memory_size, args.max_eps, args.min_eps, device]

    if args.model == 'dqn':
        agent = DQNAgent(*agent_params)
    elif args.model == 'ddqn':
        agent = DoubleDQNAgent(*agent_params)
    elif args.model == 'd3qn':
        agent = DuelingDDQNAgent(*agent_params)

    if args.train:
        train_params = [env, paths, args.episodes, best_params['max_steps'], best_params['learn_freq'],
                        args.record, args.record_freq]
        agent.train(*train_params)
    else:
        test(agent, env, paths, agent.model_name, record=args.record, record_freq=args.record_freq)


def main_nni():
    # fixed params
    params = {'cuda_device': 0,
              'model': 'dqn',
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
    params['hidden_layers_size'] = [int(nni_params['hidd1_size']), int(nni_params['hidd2_size'])]
    if params['model'] == 'd3qn':
        params['hidden_layers_size'].append(int(nni_params['hidd3_size']))

    device = torch.device(f"cuda:{params['cuda_device']}" if torch.cuda.is_available() else "cpu")
    print(device)

    paths = None

    env = gym.make('LunarLanderContinuous-v2')
    set_seed(env, seed=0)

    state_size = 8  # state vector dim
    action_space = quantize_space(actions_range=[(-1, 1), (-1, 1)], bins=[5, 5])

    agent_params = [state_size, len(action_space), action_space,
                    params['batch_size'], params['lr'], params['gamma'],
                    params['eps_decay'], params['target_update'], params['hidden_layers_size'],
                    params['memory_size'], params['max_eps'], params['min_eps'], device]

    if params['model'] == 'dqn':
        agent = DQNAgent(*agent_params)
    elif params['model'] == 'ddqn':
        agent = DoubleDQNAgent(*agent_params)
    elif params['model'] == 'd3qn':
        agent = DuelingDDQNAgent(*agent_params)

    train_params = [env, paths, params['episodes'], params['max_steps'], params['learn_freq']]
    agent.train(*train_params, is_nni=True)


if __name__ == "__main__":
    main()
    # main_nni()
