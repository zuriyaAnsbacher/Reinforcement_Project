import nni
import json
import gym
import argparse
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join('../..', 'Reinforcement_Project')))

from utils import set_seed, test
from Agent import DDPGAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ddpg')
    parser.add_argument('--train', default=True, help='train agent')
    parser.add_argument('--use_nni_params', default=True, help='if true, get params from json file')
    parser.add_argument('--set_num', type=str, default='1', help='set idx from nni')

    parser.add_argument('--memory_size', type=int, default=100000, help='memory buffer size')  # default=10000000
    parser.add_argument('--episodes', type=int, default=800, help='Number of episodes used in training')
    parser.add_argument('--cuda_device', type=int, default=1)
    parser.add_argument('--max_eps', default=1.0)
    parser.add_argument('--min_eps', default=0.01)
    parser.add_argument('--action_noise', choices=['ou1', 'ou2'], default='ou1')


    args = parser.parse_args()

    if args.use_nni_params:
        with open('best_params.json') as json_file:
            best_params = json.load(json_file)[f'{args.model}_params{args.set_num}']

    else:
        best_params = {"batch_size": 64,
                       "gamma": 0.9806754085998094,
                       "tau": 0.006188502175143961,
                       "lr_actor": 0.0021846212925658737,
                       "lr_critic": 0.002548247661148013,
                       "learn_freq": 4,
                       "update_factor": 8,
                       "eps_decay": 0.9508974596590599,
                       "max_steps": 700,
                       "hidden_actor": [64, 64],
                       "hidden_critic": [64, 64]
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

    state_size = env.observation_space.shape[0]  # = 8 (state vector dim)
    action_size = env.action_space.shape[0]   # = 2 (continuous vector dim)

    agent_params = [state_size, action_size, best_params['batch_size'], best_params['gamma'], best_params['lr_actor'],
                    best_params['lr_critic'], best_params['tau'], args.max_eps, args.min_eps, best_params['eps_decay'],
                    best_params['hidden_actor'], best_params['hidden_critic'], args.memory_size, args.action_noise, device]


    agent = DDPGAgent(*agent_params)

    if args.train:
        train_params = [env, paths, args.episodes, best_params['max_steps'], best_params['learn_freq'], best_params['update_factor']]
        agent.train(*train_params)
    else:
        test(agent, env, paths, agent.model_name)


def main_nni():
    # fixed params
    params = {'cuda_device': 0,
              'memory_size': 1000000,
              'episodes': 600,
              'max_eps': 1.0,
              'min_eps': 0.01,
              'action_noise': 'ou1'}

    # nni params
    nni_params = nni.get_next_parameter()
    params['batch_size'] = int(nni_params['batch_size'])
    params['gamma'] = float(nni_params['gamma'])
    params['tau'] = float(nni_params['tau'])
    params['lr_actor'] = float(nni_params['lr_actor'])
    params['lr_critic'] = float(nni_params['lr_critic'])
    params['learn_freq'] = int(nni_params['learn_freq'])
    params['update_factor'] = int(nni_params['update_factor'])
    params['eps_decay'] = float(nni_params['eps_decay'])
    params['max_steps'] = int(nni_params['max_steps'])
    params['hidden_actor'] = [int(nni_params['hidden1_actor']), int(nni_params['hidden2_actor'])]
    params['hidden_critic'] = [int(nni_params['hidden1_critic']), int(nni_params['hidden2_critic'])]

    device = torch.device(f"cuda:{params['cuda_device']}" if torch.cuda.is_available() else "cpu")
    print(device)

    paths = None

    env = gym.make('LunarLanderContinuous-v2')
    set_seed(env, seed=0)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    agent_params = [state_size, action_size, params['batch_size'], params['gamma'], params['lr_actor'],
                    params['lr_critic'], params['tau'], params['max_eps'], params['min_eps'],
                    params['eps_decay'], params['hidden_actor'], params['hidden_critic'],
                    params['memory_size'], params['action_noise'], device]

    agent = DDPGAgent(*agent_params)

    train_params = [env, paths, params['episodes'], params['max_steps'], params['learn_freq'], params['update_factor']]
    agent.train(*train_params, is_nni=True)


if __name__ == '__main__':
    main()
    # main_nni()
