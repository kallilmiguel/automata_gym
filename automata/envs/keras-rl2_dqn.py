#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:59:22 2020

@author: kallil
"""

import numpy as np
import gym
import csv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Embedding, LSTM, Dropout, Reshape
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from policy import CustomEpsGreedyQPolicy

case=1
last_actions=[0,1,10,11,12,13]

def q_possible():
    q_p = []
    mp = env.mapping(index=False)
    for i in range(len(env.possible_transitions())):
        q_p.append([mp[env.possible_transitions()[i]], q_values[env.actual_state][env.possible_transitions()[i]]])
    return q_p

with open('rp/case'+str(case)+'.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
reward = list(map(int, data[1]))
probabilities = list(map(float, data[2]))

env = gym.make('automata:automata-v0')

env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, last_action=last_actions, products=10, probs=probabilities)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Embedding(env.observation_space.n, 50, input_length=1))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.3))
model.add(Reshape((50,)))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = CustomEpsGreedyQPolicy(automataEnv=env, eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, gamma=0.999)
dqn.compile(Adam(lr=1e-4), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=100000, visualize=False, verbose=1)

q_values = dqn.compute_batch_q_values([0])
for i in range(1,env.observation_space.n):
    q_values = np.vstack((q_values, dqn.compute_batch_q_values([i])))

# After training is done, we save the final weights.
dqn.save_weights('renault_case{}_weigths.h5f'.format(case), overwrite=True)


# Finally, evaluate our algorithm for 5 episodes.
#dqn.test(env, nb_episodes=5, visualize=False)


#Caminho até B3
env.reset()
env.step(23)
env.step(5)
env.step(24)
env.step(6)
env.step(22)
env.step(4)


#Passando pelo bloco estendido e indo até B6
env.step(37)
env.step(13)
env.step(39)
env.step(15)
env.step(38)
env.step(14)
env.step(36)
env.step(12)
env.step(27)
env.step(9)

#Direto para B6
env.step(35)
env.step(11)
env.step(25)
env.step(7)
