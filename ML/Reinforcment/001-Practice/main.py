import gym 
import numpy as np

env = gym.make('FrozenLake-v0')

EPISODES = 25000
epsilon = 1
LEARNING_RATE = 0.1
DISCOUNT = 0.95

size = [env.observation_space.n, env.action_space.n]
q_table = np.zeros(shape=size)
print(q_table)

state = env.reset()

for i in range(1, EPISODES):
    done = False

    while not done:
        action = np.argmax(q_table[state])
        new_state, reward, done, _ = env.step(action)
        env.render()

        max_future_q = np.max(q_table[new_state])
        current_q = q_table[state, action]  
    
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[state, action] = new_q
        print(current_q)
        print(max_future_q)
        print(reward)
        print(new_q)

        
        state = new_state
env.close()

