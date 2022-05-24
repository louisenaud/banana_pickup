"""
Created by: louisenaud on 5/24/22 at 3:36 PM for banana_pickup.
"""
import os
import warnings
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment

from agent import Agent

warnings.simplefilter("ignore", UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env_path = "Banana.app"

env = UnityEnvironment(file_name=env_path)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]


def dqn_trainer(agent, n_episodes=100, print_range=10, eps_start=1.0, eps_end=0.01, eps_decay=0.995, early_stop=13,
                verbose=False):
    """Deep Q-Learning trainer.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        print_range (int): range to print partials results
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        early_stop (int): Stop training when achieve a defined score respecting 10 min n_episodes.
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=print_range)  # last 100 scores
    scores_mean = []
    eps = eps_start  # initialize epsilon
    for i in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        while True:
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done, i)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        if verbose:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)), end="")
            if i % print_range == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))

        if np.mean(scores_window) >= early_stop and i > 10:
            if verbose:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
            break

    return scores, i, np.mean(scores_window)

if __name__ == "__main__":


    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    env_info = env.reset(train_mode=True)[brain_name]

    agent = Agent(state_size=state_size, action_size=action_size, seed=199, nb_hidden=(64, 64),
                  learning_rate=0.001, memory_size=int(1e6), prioritized_memory=False, batch_size=64,
                  gamma=0.9, tau=0.03, small_eps=0.03, update_every=4)

    scores, episodes, last_avg_score = dqn_trainer(agent, n_episodes=500, early_stop=13, verbose=True)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Average Score')
    plt.xlabel('Episode #')
    plt.show()

    # prioritized memory
    # env_info = env.reset(train_mode=True)[brain_name]
    #
    # agent = Agent(state_size=state_size, action_size=action_size, seed=199, nb_hidden=(64, 64),
    #               learning_rate=0.001, memory_size=int(1e6), prioritized_memory=True, batch_size=64,
    #               gamma=0.9, tau=0.03, small_eps=0.03, update_every=4)
    #
    # scores, episodes, last_avg_score = dqn_trainer(agent, n_episodes=500, early_stop=13, verbose=True)
    #
    # # plot the scores
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(np.arange(len(scores)), scores)
    # plt.ylabel('Average Score')
    # plt.xlabel('Episode #')
    # plt.show()

    plt.plot(agent.losses)
    plt.title("Losses")
    plt.show()

    # save model
    model_path = "model.pt"
    agent.save_model(model_path)

    # load model
    model_path = "model.pt"
    agent.load_model(model_path)

    # test model
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    eps = 0.05
    while True:
        action = agent.act(state, eps)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break

    print("Score: {}".format(score))

    # close environment
    env.close()