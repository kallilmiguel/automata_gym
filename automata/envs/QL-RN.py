#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:49:07 2020

@author: kallil
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 11:22:34 2020

@author: kallil
"""

import gym
import random
import numpy as np
import time
from policy import CustomEpsGreedyQPolicy
import csv 


case=1
last_actions = [0,1,10,11]

def q_possible():
    q_p = []
    mp = env.mapping(index=False)
    for i in range(len(env.possible_transitions())):
        q_p.append([mp[env.possible_transitions()[i]], q_table[env.actual_state][env.possible_transitions()[i]]])
    return q_p

with open('rp/case'+str(case)+'.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
reward = list(map(int, data[1]))
probabilities = list(map(float, data[2]))

env = gym.make('automata:automata-v0')

env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, products=10, last_action=last_actions, probs=probabilities)

num_actions = env.action_space.n
num_states = env.observation_space.n

q_table = np.zeros([num_states, num_actions], dtype = np.float32)

lr = 0.1
gamma = 0.999
epsilon = 0.1


def choose_action(env):
    pt = np.array(env.possible_transitions())
    action = pt[env.possible_space.sample()]
    return action

def epsilon_greedy(env, epsilon, q_table, state): 
    action = -1
    pt = np.array(env.possible_transitions())
    uncontrollable = np.array(env.ncontrollable)
    ptu = np.intersect1d(pt,uncontrollable)
    probs = [[env.probs[i], ptu[i]] for i in range(len(ptu)) if env.probs[i]>0]
    
    while True:
    
        if(len(probs)>0):
            for i in range(len(probs)):
                pt = np.delete(pt, np.where(pt==ptu[i]))
            random.shuffle(probs)
            
            for i in range(len(probs)):
                if(np.random.uniform(0,1)<probs[i][0]):
                    return probs[i][1]
                
        if(pt.size>0):
            controllable = np.array(env.controllable)
            ptc = np.intersect1d(pt,controllable)   
        
            if ptc.size>0:
                if random.uniform(0,1) < epsilon:
                    action = pt[random.randint(0,pt.size-1)]
                else:
                    action = ptc[np.argmax(q_table[state,ptc])]
            else: 
                    action = pt[random.randint(0,pt.size-1)]
                
        if action !=-1:
            break
        
    return action
            

def update_q_table(q_table, state, action, next_state, reward):
    old = q_table[state, action]
    next_max = np.max(q_table[next_state])
            
    
    
    new_value = (1-lr)* old + lr * (reward + gamma*next_max)
    q_table[state,action] = new_value


    
episodes = 10000
start = time.time()
bad=0
good=0
for i in range(episodes):
    print(i)
    state = env.reset()
    done = False
    

    
    while not done:
        action=-1
        while action == -1:
            action = epsilon_greedy(env, epsilon, q_table, state)
        if(action>=0 and action<=3):
            bad+=1
        elif(action >=16 and action <=20):
            good+=1
        next_state, reward, done, info = env.step(action)    
        
        
        update_q_table(q_table, state, action, next_state, reward)
        
        state = next_state
    
        
end = time.time()
print("Execution time: {}".format(end-start))

#Testando decisões inteligentes
epsilon=0
episodes = 5
print("Teste Inteligente")
for i in range(episodes):
    state = env.reset()
    total_reward=0
    #env.render()
    
    done = False
    while not done:
        action=-1
        while action == -1:
            action = epsilon_greedy(env, epsilon, q_table, state)
        
        total_reward+=reward
        next_state, reward, done, info = env.step(action)
        update_q_table(q_table, state, action, next_state, reward)
        
        state = next_state
        
    print("Episode: {}, Total Reward: {}".format(i+1, total_reward))
    
#Testando decisões aleatórias
epsilon=1
episodes = 5
print("Teste Aleatório")
for i in range(episodes):
    state = env.reset()
    total_reward=0
    #env.render()
    
    done = False
    while not done:
        action=-1
        while action == -1:
            action = epsilon_greedy(env, epsilon, q_table, state)
        
        total_reward+=reward
        next_state, reward, done, info = env.step(action)
        update_q_table(q_table, state, action, next_state, reward)
        
        state = next_state
        
    print("Episode: {}, Total Reward: {}".format(i+1, total_reward))
        

env.step(14)
env.step(2)
