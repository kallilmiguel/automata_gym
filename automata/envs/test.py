#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 17:22:22 2019

@author: kallil
"""

import gym

env = gym.make('automata:automata-v0')

env2 = gym.make('Taxi-v3')

env.reset("SM/BufferD.xml", [-1, -1, -1, 10], 1, 11)


env2.reset()


env.render()

env.mapping()

env.step(0)
env.step(1)
env.step(2)
env.step(3)
env.step(4)
env.step(1)
env.step(2)
env.step(2)
env.step(3)

