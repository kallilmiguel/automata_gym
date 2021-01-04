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
    
    pt = np.array(env.possible_transitions())
    uncontrollable = np.array(env.ncontrollable)
    ptu = np.intersect1d(pt,uncontrollable)
    probs = np.array([[ptu[i],env.probs[ptu[i]]] for i in range(len(ptu)) if env.probs[ptu[i]]>0])
    controllable = np.array(env.controllable)
    ptc = np.intersect1d(pt,controllable)
    
    actions = np.array([])
    if(probs.size>0):
        actions = np.append(actions, random.choices(probs[:,0],weights=probs[:,1]))

    
    if(len(probs)>0):
        for i in range(len(probs)):
            ptu = np.delete(ptu, np.where(ptu==probs[i,0]))
    if(ptu.size>0):
        actions=np.append(actions, ptu)
    
    if(ptc.size>0):
        if random.uniform(0,1) < epsilon:
            actions = np.append(actions, ptc[random.randint(0,ptc.size-1)])
        else:
            actions = np.append(actions,ptc[np.argmax(q_table[state,ptc])])
        
    return actions[random.randint(0,actions.size-1)].astype(np.int32)
    
            

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
xValues=[]

for k in range(cases):

    with open('testes/bad_A-cost/case'+str(k+1)+'.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    reward = list(map(int, data[1]))
    probabilities = list(map(float, data[2]))
    
    env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
    
    last_actions=last_action
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    
    q_table = np.zeros([num_states, num_actions], dtype = np.float32)
        
    #Alterar esse valor para aparecer no eixo x do gráfico
    xValues.append(-1*env.reward[0]) # MUDAR AQUI
    
    lr = 0.1
    gamma = 0.9
    epsilon = 0.5
    
    bad=[1011]
    #bad = [457,257,912,259,304,1226,888,1220,313,672]
    
    l=[]
    for i in env.transitions:
        if(i[2]==22 or i[2]==21 or i[2]==20 or i[2]==19):
            l.append(i[0])
    
        
    ##NStepSarsa
    episodes = 3000
    n=10
    start = time.time()
    for i in range(episodes):
        steps=0
        print("Case: {}/{} ----- Episode: {}/{}".format(k+1,cases,i,episodes))
        A,S,R=[],[],[]
        state = env.reset()
        S.append(state)
        done = False
        T=100000
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
                    print("---Steps:{}".format(steps))
                else:
                    action = epsilon_greedy(env, epsilon, q_table, s_n)
                    steps+=1
                    A.append(action)
            tau=t-n+1
            if(tau>=0):
                G=0
                for z in range(tau+1, min(tau+n,T)+1):
                    G+=(gamma**(z-tau-1))*R[z-1]
                if(tau+n<T):
                    G = G+(gamma**n)*q_table[S[tau+n],A[tau+n]]
                q_table[S[tau],A[tau]] += lr*(G-q_table[S[tau],A[tau]])
                # if(S[tau] in bad and i>100):
                #     print(S[tau])
                #     print(A[tau])
                #     print("Alô youtube")50005000
            if(tau==T-1):
                break
            t+=1
    
        
            
    end = time.time()
    print("Execution time: {}".format(end-start))
    

    #Testando decisões inteligentes
    epsilon=0
    episodes = 50
    
    print("\tTeste Inteligente")
    for i in range(episodes):
        state = env.reset()
        total_reward=0
        #env.render()
        
        done = False
        while not done:
            
            action = epsilon_greedy(env, epsilon, q_table, state)
            if(action in last_actions or action ==12 or action ==13):
                final_states_int.append((action,k+1))
            
#            if(state in l and (action!=20 and action !=22)):
 #              print("State:{}/Action:{}".format(state,action))
                
           
            next_state, reward, done, info = env.step(action)
            total_reward+=reward
            
            state = next_state
        
        info_int.append((total_reward, xValues[k], "Supervisory+RL"))
        print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward))
        
    #Testando decisões aleatórias
    epsilon=1
    episodes = 50
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
            
            if(action in last_actions or action==12 or action==13):
                final_states_rdn.append((action,k+1))

            
            next_state, reward, done, info = env.step(action)
            total_reward+=reward
            
            state = next_state
        
        info_rdn.append((total_reward, xValues[k], "Supervisory"))
        print("\t\tEpisode: {}, Total Reward: {}".format(i+1, total_reward))

# Alterar dataname para salvar diferentes bases de dados
dataname="bad_A" # MUDAR AQUI
reward_dataname=dataname+"_reward.csv"
occurrences_int_dataname=dataname+"_fsInt.csv"
occurrences_rnd_dataname=dataname+"_fsRnd.csv"

#nome do eixo x do gráfico
xlabel_name="bad_A Cost" # MUDAR AQUI

fsInt=[]
fsRdn=[]
last_actions.append(12)
last_actions.append(13)
for i in last_actions:
    for j in range(cases):
        fsInt.append((final_states_int.count((i,j+1)), xValues[j], env.mapping()[i][1]))
        fsRdn.append((final_states_rdn.count((i,j+1)), xValues[j], env.mapping()[i][1]))


data = np.vstack((info_int, info_rdn))
data = pd.DataFrame(data, columns=["mean reward",  xlabel_name, "method"])
data.to_csv(reward_dataname)
states_int = pd.DataFrame(fsInt, columns=["Number of occurrences",xlabel_name, "event"])
states_rdn = pd.DataFrame(fsRdn, columns=["Number of occurrences",xlabel_name, "event"])
states_int.to_csv(occurrences_int_dataname)
states_rdn.to_csv(occurrences_rnd_dataname)



intel = pd.read_csv(occurrences_int_dataname)
randomic = pd.read_csv(occurrences_rnd_dataname)
df = pd.read_csv(reward_dataname)
plot = sns.lineplot(data=df, x=xlabel_name, y="mean reward", hue="method")


sint = sns.barplot(data=intel, x=xlabel_name, y="Number of occurrences", hue="event")

srd = sns.barplot(data=randomic, x=xlabel_name, y="Number of occurrences", hue="event")

