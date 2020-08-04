#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:52:09 2020

@author: kallil
"""

import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar

import gym
import tensorflow as tf
from tensorboardX import SummaryWriter

from keras import Model, Sequential
from keras.layers import Dense, Embedding, Reshape
from keras.optimizers import Adam

# env = gym.make('automata:automata-v0')

# env.reset("SM/MesaD.xml",  [-1,-1,-1,-1,-1,20,-1,-1,-0.25,-1,-1], 1, 60)


env = gym.make('Taxi-v3')
env.reset()
n_states = env.observation_space.n
n_actions = env.action_space.n

class Agent:
    def __init__(self, env, optimizer):
        
        # Initialize atributes
        self._state_size = env.observation_space.n
        self._action_size = env.action_space.n
        self._optimizer = optimizer
        
        self.experience_replay = deque(maxlen=2000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.9
        self.epsilon = 0.1
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    
    """
    Function act represents the exploration-exploitation policy, which is the 
    customized e-greedy function used in the Q-learning example
    """
    def act(self, state):
        # pt = np.array(env.possible_transitions())
        # controllable = np.array(env.controllable)
        # ptc = np.intersect1d(pt,controllable)
        # q_values = self.q_network.predict(state)
        # if ptc.size>0:
        #     if random.uniform(0,1) < self.epsilon:
        #         action = pt[env.possible_space.sample()]
        #     else:
        #         action = ptc[np.argmax(q_values[0,ptc])]
        # else:
        #     action = pt[env.possible_space.sample()]
            
        # return action
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)
        
        for state, action, reward, next_state, terminated in minibatch:
            
            target = self.q_network.predict(state)
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.q_network.fit(state, target, epochs=1, verbose=0)

optimizer = Adam(learning_rate=0.01)
agent = Agent(env, optimizer)

batch_size = 32
num_of_episodes = 100
timesteps_per_episode = 60
agent.q_network.summary()

writer = SummaryWriter()

for e in range(0, num_of_episodes):
    # Reset the enviroment
    state = env.reset()
    state = np.reshape(state, [1, 1])
    
    # Initialize variables
    reward = 0
    REWARD=0
    terminated = False
    
    bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for timestep in range(timesteps_per_episode):
        # Run Action
        action = agent.act(state)
        
        # Take action    
        next_state, reward, terminated, info = env.step(action) 
        next_state = np.reshape(next_state, [1, 1])
        agent.store(state, action, reward, next_state, terminated)
        
        REWARD+=reward
        state = next_state
        if terminated:
            agent.alighn_target_model()
            break
            
        if len(agent.experience_replay) > batch_size:
            agent.retrain(batch_size)
        
        if timestep%10 == 0:
            bar.update(timestep/10 + 1)
    
    
    print("reward: {}".format(REWARD))
    bar.finish()
    if (e + 1) % 10 == 0:
        print("**********************************")
        print("Episode: {}".format(e + 1))
        #env.render()
        print("**********************************")
