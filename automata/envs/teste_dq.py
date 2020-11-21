#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:17:33 2020

@author: kallil
"""

import numpy as np
import gym
import random
import csv
import tensorflow as tf


case=1

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

env.reset("SM/Mesa.xml", rewards=reward, stop_crit=1, last_action=300, probs=probabilities)


print("Number of actions: %d" % env.action_space.n)
print("Number of states: %d" % env.observation_space.n)

action_size = env.action_space.n
state_size = env.observation_space.n

np.random.seed(123)
env.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape, LSTM, Dropout
from keras.optimizers import Adam
from policy import CustomEpsGreedyQPolicy

env.reset()

model_only_embedding = Sequential()
model_only_embedding.add(Embedding(state_size, action_size, input_length=1))
model_only_embedding.add(Reshape((action_size,)))
print(model_only_embedding.summary())

model = Sequential()
model.add(Embedding(state_size, 100, input_length=1))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.3))
model.add(Reshape((50,)))
model.add(Dense(action_size, activation='linear'))
print(model.summary())

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

memory = SequentialMemory(limit=50000, window_length=1)
policy = CustomEpsGreedyQPolicy(automataEnv=env, eps=.9)
dqn_only_embedding = DQNAgent(gamma=.999, model=model, nb_actions=action_size, memory=memory, 
                              nb_steps_warmup=500, target_model_update=1e-2, policy=policy, test_policy=policy)
dqn_only_embedding.compile(Adam(lr=1e-3), metrics=['mae'])
dqn_only_embedding.fit(env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=100, 
                       log_interval=10000, start_step_policy=policy)
q_values = dqn_only_embedding.compute_batch_q_values([0])
for i in range(1,state_size):
    q_values = np.vstack((q_values, dqn_only_embedding.compute_batch_q_values([i])))


#dqn_only_embedding.test(env, nb_episodes=5, visualize=False, verbose=1, nb_max_episode_steps=100, 
 #                     start_step_policy=policy)


    
    
    

#Caminho para o carro 0 até MI
env.reset()
env.step(21)
env.step(4)
env.step(22)
env.step(5)
env.step(19)
env.step(2)
#abaixo para B6
env.step(28)
env.step(8)
env.step(23)
env.step(6)

#retrabalho
env.step(27)
#segundo ciclo
env.step(18)
env.step(24)
env.step(7)

#direto
env.step(25)


#Caminho para carro 1 até o buffer MI
env.reset()
env.step(21)
env.step(4)
env.step(22)
env.step(5)
env.step(20)
env.step(3)
#abaixo para B6
env.step(29)
env.step(9)
env.step(23)
env.step(6)




