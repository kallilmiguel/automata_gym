#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 17:22:22 2019

@author: kallil
"""

import gym

env = gym.make('automata:automata-v0')

env.reset("teste.xml")

env.render()

env.mapping()
env.step(0)
env.step(1)
env.step(2)
env.step(4)
env.step(4)
env.step(1)
env.step(2)
env.step(2)
env.step(3)

