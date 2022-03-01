import numpy as np
from statistics import mean
from itertools import product
import random
import torch
import glob
from gym.wrappers import Monitor
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt


def set_seed(env, seed):
    np.random.seed(seed)
    env.seed(seed)
    # env.action_space.seed(seed)  # todo: check (kfir+sharon code)
    torch.manual_seed(seed)
    random.seed(seed)
    # cudnn.deterministic = True # todo: check (kfir+sharon code)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def quantize_space(actions_range, bins):
    discrete_actions = product(*[np.linspace(start, end, num=num_of_splits) for
                                 (start, end), num_of_splits in zip(actions_range, bins)])
    return list(discrete_actions)


def create_plot(scores, model_name, save_path):
    avg = []
    for i in range(len(scores)):
        j = 0 if i <= 99 else i - 99
        avg.append(mean(scores[j:i+1]))

    fig, axis = plt.subplots()
    axis.clear()
    axis.plot(scores, 'c', label='Score', alpha=0.7)
    axis.plot(avg, 'orange', label='Average score (up to last 100 episodes)')
    axis.axhline(200, c='gray', label='Goal', alpha=0.7)
    axis.set_xlabel('Episodes')
    axis.set_ylabel('Scores')
    axis.legend(loc='lower right')
    plt.title(f'{model_name} Train')

    plt.savefig(save_path)
    plt.close()


def test(agent, env, paths, model_name, episode_num=100, steps_num=1000):
    file_name = ''.join([model_name, '_params', paths['args_set_num']])
    saved_trained_model = ''.join([paths['saved_model_path'], file_name, '.pth'])
    path_to_save_plot = ''.join([paths['saved_plots_path'], 'test_', file_name, '.png'])

    agent.load(saved_trained_model)
    agent.set_eval()

    scores_list = []
    for episode in range(1, episode_num + 1):
        state = env.reset()
        score = 0
        for step in range(1, steps_num + 1):
            action = agent.get_action(state, just_greedy=True)
            next_state, reward, done, _ = env.step(action)
            score += reward
            state = next_state
            if done:
                break

        scores_list.append(score)
        print(f'Episode: {episode}, score: {score:.02f}')

    plt.axhline(200, c='gray', label='Goal', alpha=0.7)
    plt.plot(scores_list, 'c', label='Score', alpha=0.7)
    plt.legend(loc='lower right')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.title(f'{model_name} Test. Average score: {mean(scores_list):.02f}')
    plt.savefig(path_to_save_plot)

    plt.close()
    env.close()


"""
Utility functions to enable video recording of gym environment 
and displaying it.
To enable video, just do "env = wrap_env(env)""
"""


def show_video(num):
    mp4list = glob.glob(f'video_{num}/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


def wrap_env(env, num):
    env = Monitor(env, f'./video_{num}', force=True)
    return env