#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:49:07 2020

@author: kallil
"""

import gym
import random
import numpy as np
import time
from policy import CustomEpsGreedyQPolicy
import matplotlib.pyplot as plt
import csv 
import pandas as pd
import seaborn as sns

def choose_action(env):
    pt = np.array(env.possible_transitions())
    action = pt[env.possible_space.sample()]
    return action

def epsilon_greedy(env, epsilon, q_table, state): 
    action = -1
    pt = np.array(env.possible_transitions())
    uncontrollable = np.array(env.ncontrollable)
    ptu = np.intersect1d(pt,uncontrollable)
    probs = [[env.probs[ptu[i]], ptu[i]] for i in range(len(ptu)) if env.probs[ptu[i]]>0]
    
    if(len(probs)>0):
        for i in range(len(probs)):
            pt = np.delete(pt, np.where(pt==ptu[i]))
    
    while True:
    
        
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


def q_possible():
    q_p = []
    mp = env.mapping(index=False)
    for i in range(len(env.possible_transitions())):
        q_p.append([mp[env.possible_transitions()[i]], q_table[env.actual_state][env.possible_transitions()[i]]])
    return q_p


env = gym.make('automata:automata-v0')
cases=9
info_int=[]
info_rdn=[]
final_states_int=[]
final_states_rdn=[]
last_action=[0,1,10,11]

for k in range(cases):

    with open('rp/case'+str(k+1)+'.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    reward = list(map(int, data[1]))
    probabilities = list(map(float, data[2]))
    
    env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
    
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    
    q_table = np.zeros([num_states, num_actions], dtype = np.float32)
    
    lr = 0.5
    gamma = 0.999
    epsilon = 0.5
    
    
    
        
    episodes = 10000
    start = time.time()
    
    for i in range(episodes):
        print("Case: {}/{} ----- Episode: {}/{}".format(k+1,cases,i,episodes))
        state = env.reset()
        done = False
        
    
        
        while not done:

            action = epsilon_greedy(env, epsilon, q_table, state)


            next_state, reward, done, info = env.step(action)    
            
            
            update_q_table(q_table, state, action, next_state, reward)
            
            state = next_state
        
            
    end = time.time()
    print("\tExecution time: {}".format(end-start))
    

    #Testando decisões inteligentes
    epsilon=0
    episodes = 20
    rewards_int=[]
    print("\tTeste Inteligente")
    for i in range(episodes):
        state = env.reset()
        total_reward=0
        #env.render()
        
        done = False
        cars=0
        
        while not done:
            action=-1
            while action == -1:
                action = epsilon_greedy(env, epsilon, q_table, state)
            
            if(action in last_action):
                final_states_int.append((action,k+1))
                
           
            next_state, reward, done, info = env.step(action)
            #update_q_table(q_table, state, action, next_state, reward)
            total_reward+=reward
            
            state = next_state
        
        info_int.append((total_reward, 10*(k+1), "Supervisory+RL"))
        print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward))
        
    #Testando decisões aleatórias
    epsilon=1
    episodes = 20
    rewards_rdn=[]
    print("\tTeste Aleatório")
    for i in range(episodes):
        state = env.reset()
        total_reward=0
        #env.render()
        
        done = False
        cars=0
        count=0
        
        while not done:

            action = epsilon_greedy(env, epsilon, q_table, state)
            
            if(action in last_action):
                final_states_rdn.append((action,k+1))

            
            next_state, reward, done, info = env.step(action)
            #update_q_table(q_table, state, action, next_state, reward)
            total_reward+=reward
            
            state = next_state
        
        info_rdn.append((total_reward, 10*(k+1), "Supervisory"))
        print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward))

fsInt=[]
fsRdn=[]
for i in last_action:
    for j in range(cases):
        fsInt.append((final_states_int.count((i,j+1)), 10*(j+1), env.mapping()[i][1]))
        fsRdn.append((final_states_rdn.count((i,j+1)), 10*(j+1), env.mapping()[i][1]))

data = np.vstack((info_int, info_rdn))
data = pd.DataFrame(data, columns=["mean reward", "Fail Probability", "method"])
data.to_csv("data_sarsa.csv")
states_int = pd.DataFrame(fsInt, columns=["Number of occurrences","Fail Probability", "event"])
states_rdn = pd.DataFrame(fsRdn, columns=["Number of occurrences","Fail Probability", "event"])
states_int.to_csv("final_states_sarsa_int.csv")
states_rdn.to_csv("final_states_sarsa_random.csv")
intel = pd.read_csv("final_states_sarsa_int.csv")
randomic = pd.read_csv("final_states_sarsa_random.csv")
df = pd.read_csv("data_sarsa.csv")
plot = sns.lineplot(data=df, x="Fail Probability", y="mean reward", hue="method")
plot.set_title("Graph for reward by making 10 cars over probability of success entering block C - 90% not entering")


sint = sns.barplot(data=intel, x="Fail Probability", y="Number of occurrences", hue="event")
sint.set_title("Graph for number of occurrences of each event for the SARSA method")
srd = sns.barplot(data=randomic, x="Fail Probability", y="Number of occurrences", hue="event")
srd.set_title("Graph for number of occurrences of each event without the SARSA method")

