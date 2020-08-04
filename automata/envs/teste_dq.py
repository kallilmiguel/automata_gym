#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:17:33 2020

@author: kallil
"""

import numpy as np
import gym
import random

env = gym.make('automata:automata-v0')

env.reset("SM/CanaisD.xml",  [2,3,-1,-1,-1,-1], 1, 30)

# env=gym.make('Taxi-v3')
# env.reset()

print("Number of actions: %d" % env.action_space.n)
print("Number of states: %d" % env.observation_space.n)

action_size = env.action_space.n
state_size = env.observation_space.n

np.random.seed(123)
env.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from keras.optimizers import Adam
from policy import CustomEpsGreedyQPolicy

env.reset()
env.step(env.action_space.sample())[0]

model_only_embedding = Sequential()
model_only_embedding.add(Embedding(state_size, action_size, input_length=1))
model_only_embedding.add(Reshape((action_size,)))
print(model_only_embedding.summary())

model = Sequential()
model.add(Embedding(state_size, 10, input_length=1))
model.add(Reshape((10,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(action_size, activation='linear'))
print(model.summary())

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

memory = SequentialMemory(limit=50000, window_length=1)
policy = CustomEpsGreedyQPolicy(automataEnv=env, eps=.1)
dqn_only_embedding = DQNAgent(model=model_only_embedding, nb_actions=action_size, memory=memory, 
                              nb_steps_warmup=500, target_model_update=1e-2, policy=policy, test_policy=policy)
dqn_only_embedding.compile(Adam(lr=1e-3), metrics=['mae'])
dqn_only_embedding.fit(env, nb_steps=100000, visualize=False, verbose=1, nb_max_episode_steps=100, 
                       log_interval=10000, start_step_policy=policy)
q_values = dqn_only_embedding.compute_batch_q_values([0])
for i in range(1,state_size):
    q_values = np.vstack((q_values, dqn_only_embedding.compute_batch_q_values([i])))


dqn_only_embedding.test(env, nb_episodes=5, visualize=True, verbose=1, nb_max_episode_steps=100, 
                      start_step_policy=policy)