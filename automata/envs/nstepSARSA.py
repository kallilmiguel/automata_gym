#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 23:02:58 2020

@author: kallil
"""

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
import csv 


case=9
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
epsilon = 0.5


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
            
##NStepSarsa
episodes = 10000
n=50
start = time.time()
for i in range(episodes):
    print(i)
    A,S,R=[],[],[]
    state = env.reset()
    S.append(state)
    done = False
    T=1000
    t=0
    action=epsilon_greedy(env,epsilon, q_table, state)
    A.append(action)
    while True:
        if(t<T):
            s_n,r_n,done,_ = env.step(action)
            S.append(s_n)
            R.append(r_n)
            if(done):
                T=t+1
            else:
                action = epsilon_greedy(env, epsilon, q_table, state)
                A.append(action)
        tau = t-n+1
        if(tau>=0):
            G=0
            for i in range(tau+1, min(tau+n,T)):
                G+=(gamma**(i-tau-1))*R[i]
            if(tau+n<T):
                G = G+(gamma**n)*q_table[S[tau+n],A[tau+n]]
            q_table[S[tau],A[tau]] += lr*(G-q_table[S[tau],A[tau]])
     
            
        if(tau==T-1):
            break
        t+=1

    
        
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
        
        action = epsilon_greedy(env, epsilon, q_table, state)
        
        
        next_state, reward, done, info = env.step(action)
        total_reward+=reward
        
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
        action = epsilon_greedy(env, epsilon, q_table, state)
        
        
        next_state, reward, done, info = env.step(action)
        total_reward+=reward
        
        state = next_state
        
    print("Episode: {}, Total Reward: {}".format(i+1, total_reward))
        
env.reset()
env.step(14)
env.step(2)
