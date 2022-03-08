import nni
import gym
import argparse
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join('../..', 'Reinforcement_Project')))

from utils import set_seed
from Agent import DDPGAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, help='train agent')
    parser.add_argument('--episodes', type=int, default=800, help='Number of episodes used in training')
    parser.add_argument('--max_steps', type=int, default=700, help='maximum steps in episode')
    parser.add_argument('--set_num', type=str, default='0', help='set idx from nni')

    parser.add_argument('--lr_actor', type=float, default=0.0005848867245515764, help='learning rate actor network')
    parser.add_argument('--lr_critic', type=float, default=0.0014302222580882661, help='learning rate critic network')
    parser.add_argument('--weight_decay', default=0, help='models weight decay factor')
    parser.add_argument('--max_eps', default=0.7) # 1.0232960877577921
    parser.add_argument('--min_eps', default=0.01)
    parser.add_argument('--eps_decay', type=float, default=0.000002414)  # default=0.000002414332405642902, becuase they used '-' for decay
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.9829207330295889, help='discount factor')
    parser.add_argument('--learn_freq', type=int, default=5)
    parser.add_argument('--update_factor', type=int, default=5, help='steps for weights update')
    parser.add_argument('--memory_size', type=int, default=100000, help='memory buffer size')  # default=10000000
    parser.add_argument('--tau', default=0.0034742948471750195)
    parser.add_argument('--hidden_actor', default=[64, 64], help='the actor hidden sizes')
    parser.add_argument('--hidden_critic', default=[256, 256], help='the critic hidden sizes')
    parser.add_argument('--action_noise', choices=['ou1', 'ou2'], default='ou1')
    parser.add_argument('--cuda_device', type=int, default=0)

    args = parser.parse_args()

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

    agent_params = [state_size, action_size, args.batch_size, args.gamma, args.lr_actor,
                    args.lr_critic, args.tau, args.max_eps, args.min_eps, args.eps_decay,
                    args.hidden_actor, args.hidden_critic, args.memory_size, args.action_noise, device]

    agent = DDPGAgent(*agent_params)

    if args.train:
        train_params = [env, paths, args.episodes, args.max_steps, args.learn_freq, args.update_factor]
        agent.train(*train_params)


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
    # main()
    main_nni()
