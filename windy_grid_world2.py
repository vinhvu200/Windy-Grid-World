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


def q_learning_lambda(episodes, learning_rate, discount=1.0, _lambda=0.9):

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    e = defaultdict(lambda: np.zeros(env.action_space.n))
    x = np.arange(episodes)
    y = np.zeros(episodes)

    for episode in range(episodes):
        state = env.reset()

        probs = epsilon_greedy_policy(Q, state, env.action_space.n)
        action = np.random.choice(len(probs), p=probs)

        for step in range(10000):
            next_state, reward, done, _ = env.step(action)

            probs = epsilon_greedy_policy(Q, next_state, env.action_space.n)
            next_action = np.random.choice(len(probs), p=probs)

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]

            e[state][action] += 1

            for s in Q:
                for a in range(len(Q[s])):
                    Q[s][a] += learning_rate * td_error * e[s][a]
                    if next_action is best_next_action:
                        e[s][a] = discount * _lambda * e[s][a]
                    else:
                        e[s][a] = 0

            if done:
                y[episode] = step
                e.clear()
                break

            action = next_action
            state = next_state

    return x, y

episodes = 300
learning_rate = 0.5

x, y = q_learning_lambda(episodes, learning_rate)

fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='Episodes', ylabel='steps',
       title='Episodes vs steps')
ax.grid()

fig.savefig("Q_learning_lambda.png")

plt.show()