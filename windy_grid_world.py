import matplotlib.pyplot as plt
import numpy as np
from lib.envs.windy_gridworld import WindyGridworldEnv
from collections import defaultdict

env = WindyGridworldEnv()


def epsilon_greedy_policy(Q, state, nA, epsilon=0.05):
    probs = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q[state])
    probs[best_action] += 1.0 - epsilon

    return probs

def Q_learning(episodes, learning_rate, discount=1.0):

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

episodes = 200
learning_rate = 0.1
discount = 0.1

x, y = Q_learning(episodes, learning_rate, discount)

fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='Episodes', ylabel='steps',
       title='Episodes vs steps')
ax.grid()

fig.savefig("test.png")

plt.ion()

plt.show()
