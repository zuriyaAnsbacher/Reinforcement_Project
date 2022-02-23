import gym
import argparse
import torch

from Agents import DQNAgent
from utils import set_seed, quantize_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['dqn'], default='dqn')
    parser.add_argument('--train', default=True, help='train agent')
    # parser.add_argument('--file', default=None, help='weights file used in test')
    # parser.add_argument('--verbose', choices=[0, 1, 2, 3], default=0, help=' Verbose used in train '
    #                                                                        ' 0 (no plots, no logs, no video), '
    #                                                                        ' 1 (yes plots, no logs, no video),'
    #                                                                        ' 2 (yes plots, yes logs, no video), '
    #                                                                        ' 3 (yes plots, yes logs, yes video)')
    # parser.add_argument('--render', default=True, help='render video of the environment')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--memory_size', type=int, default=100000)  # todo: up to 1e6
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--episodes', type=int, default=800, help='number of episodes in train')
    parser.add_argument('--max_steps', type=int, default=10000, help='number of time steps in an episode (train)')
    parser.add_argument('--target_update', type=int, default=100, help='number of updates')
    parser.add_argument('--learn_freq', type=int, default=2, help='number of steps for agent weights update')
    parser.add_argument('--eps_decay', type=float, default=0.95)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--max_eps', type=float, default=1.0)
    parser.add_argument('--min_eps', type=float, default=0.01)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(device)

    # todo: add creating path for saving results
    # todo: decide how to do the train and the test (each X episode, check on test? (Ran) or if get 200 on train, check on test?)

    env = gym.make('LunarLanderContinuous-v2')
    set_seed(env, seed=0)

    state_size = 8  # state vector dim
    action_space = quantize_space(actions_range=[(-1, 1), (-1, 1)], bins=[5, 5])

    agent_params = [state_size, len(action_space), action_space, args.batch_size,
                    args.lr, args.gamma, args.memory_size, args.max_eps, args.min_eps,
                    args.eps_decay, args.target_update, device]

    if args.model == 'dqn':
        agent = DQNAgent(*agent_params)

    if args.train:
        train_params = [env, args.episodes, args.max_steps, args.learn_freq]
        agent.train(*train_params)


if __name__ == "__main__":
    main()
