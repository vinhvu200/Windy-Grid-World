import matplotlib.pyplot as plt
import numpy as np
from lib.envs.windy_gridworld import WindyGridworldEnv
from collections import defaultdict

env = WindyGridworldEnv()


def epsilon_greedy_policy(Q, state, nA, epsilon=0.05):
    '''

    :param Q: links state -> action value (dictionary)
    :param state: state character is in (int)
    :param nA: number of actions (int)
    :param epsilon: chance it will take a random move (float)
    :return:
    '''
    probs = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q[state])
    probs[best_action] += 1.0 - epsilon

    return probs

def Q_learning(episodes, learning_rate, discount=1.0):
    '''

    :param episodes: Number of episodes to run (int)
    :param learning_rate: How fast it will converge to a point (float [0, 1])
    :param discount: How much future events lose their value (float [0, 1])
    :return: x,y points to graph
    '''

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    move_cost = 0.05

    x = np.arange(episodes)
    y = np.zeros(episodes)

    for episode in range(episodes):
        state = env.reset()

        for step in range(10000):
            probs = epsilon_greedy_policy(Q, state, env.action_space.n)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            Q[state][action] -= move_cost
            next_state, reward, done, _ = env.step(action)

            td_target = reward + discount * np.amax(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += learning_rate * td_error

            if done:
                y[episode] = step
                break

            state = next_state

    return x, y

episodes = 300
learning_rate = 0.5

x, y = Q_learning(episodes, learning_rate)

fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='Episodes', ylabel='steps',
       title='Episodes vs steps')
ax.grid()

fig.savefig("Q_learning.png")

plt.show()
