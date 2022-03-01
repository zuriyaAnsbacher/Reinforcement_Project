import nni
import gym
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
    parser.add_argument('--model', choices=['dqn', 'ddqn', 'd3qn'], default='dqn')
    parser.add_argument('--train', default=False, help='train agent')
    parser.add_argument('--set_num', type=str, default='1')
    # parser.add_argument('--verbose', choices=[0, 1, 2, 3], default=0, help=' Verbose used in train '
    #                                                                        ' 0 (no plots, no logs, no video), '
    #                                                                        ' 1 (yes plots, no logs, no video),'
    #                                                                        ' 2 (yes plots, yes logs, no video), '
    #                                                                        ' 3 (yes plots, yes logs, yes video)')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--memory_size', type=int, default=100000)  # todo: up to 1e6
    parser.add_argument('--gamma', type=float, default=0.9814456989493644, help='discount factor')
    parser.add_argument('--lr', type=float, default=0.0005596196627828316)
    parser.add_argument('--episodes', type=int, default=800, help='number of episodes in train')
    parser.add_argument('--max_steps', type=int, default=1000, help='number of time steps in an episode (train)')
    parser.add_argument('--target_update', type=int, default=400, help='number of updates')
    parser.add_argument('--learn_freq', type=int, default=3, help='number of steps for agent weights update')
    parser.add_argument('--eps_decay', type=float, default=0.9535960248897746)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--max_eps', type=float, default=1.0)
    parser.add_argument('--min_eps', type=float, default=0.01)
    parser.add_argument('--hidden_layers_size', type=list, default=[64, 64])

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(device)

    paths = {'saved_model_path': './results/trained_models/',
             'saved_plots_path': './results/plots/',
             'args_set_num': args.set_num}

    # todo: decide how to do the train and the test (each X episode, check on test? (Ran) or if get 200 on train, check on test?)

    env = gym.make('LunarLanderContinuous-v2')
    set_seed(env, seed=0)

    state_size = 8  # state vector dim
    action_space = quantize_space(actions_range=[(-1, 1), (-1, 1)], bins=[5, 5])

    agent_params = [state_size, len(action_space), action_space, args.batch_size,
                    args.lr, args.gamma, args.memory_size, args.max_eps, args.min_eps,
                    args.eps_decay, args.target_update, args.hidden_layers_size, device]

    if args.model == 'dqn':
        agent = DQNAgent(*agent_params)
    elif args.model == 'ddqn':
        agent = DoubleDQNAgent(*agent_params)  # todo: check
    elif args.model == 'd3qn':
        agent = DuelingDDQNAgent(*agent_params)  # todo: check

    if args.train:
        train_params = [env, args.episodes, args.max_steps, args.learn_freq, paths]
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
        train_params = [env, params['episodes'], params['max_steps'], params['learn_freq'], paths]
        agent.train(*train_params, is_nni=True)


if __name__ == "__main__":
    main()
    # main_nni()
