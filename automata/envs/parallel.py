#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:21:12 2021

@author: kallil
"""

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
from numba import jit, cuda
import math



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

def training(q_table, env, episodes, n):
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
                data = np.ones(64)
                threadsperblock = 64
                blockspergrid = math.ceil(data.shape[0] / threadsperblock)
                nstep[threadsperblock, blockspergrid](q_table,tau,n,T,S,A,R,lr)
            if(tau==T-1):
                break
            t+=1

@cuda.jit
def nstep(q_table, tau, n, T, S, A, R, lr):
    G=0
    for z in range(tau+1, min(tau+n,T)+1):
        G+=(gamma**(z-tau-1))*R[z-1]
    if(tau+n<T):
        G = G+(gamma**n)*q_table[S[tau+n],A[tau+n]]
    q_table[S[tau],A[tau]] += lr*(G-q_table[S[tau],A[tau]])

env = gym.make('automata:automata-v0')
cases=9
info_int=[]
info_rdn=[]
final_states_int=[]
final_states_rdn=[]
last_action=[0,1,10,11]
xValues=[]

for k in range(cases):

    with open('testes/approve-A-90/case'+str(k+1)+'.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    reward = list(map(int, data[1]))
    probabilities = list(map(float, data[2]))
    
    env.reset("SM/Renault2.xml", rewards=reward, stop_crit=1, last_action=last_action, products=10, probs=probabilities)
    
    last_actions=last_action
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    
    q_table = np.zeros([num_states, num_actions], dtype = np.float32)
        
    #Alterar esse valor para aparecer no eixo x do gráfico
    xValues.append(100*env.probs[0])
    
    lr = 0.1
    gamma = 0.9
    epsilon = 0.5
    
    bad=[1011]
    #bad = [457,257,912,259,304,1226,888,1220,313,672]
    
    
    
        
    ##NStepSarsa
    episodes = 5000
    n=10
    
    start = time.time()
    
    training(q_table, env, episodes, n)   
            
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
directory="dados/"
dataname="approveA-90"
reward_dataname=directory+dataname+"_reward.csv"
occurrences_int_dataname=directory+dataname+"_fsInt.csv"
occurrences_rnd_dataname=directory+dataname+"_fsRnd.csv"
occurrences_redo_int = directory+dataname+"_redo.csv"


#nome do eixo x do gráfico
xlabel_name="Rejection Type 1 Probability"

fsInt=[]
fsRdn=[]
redo=[]
last_actions.append(12)
last_actions.append(13)
for i in last_actions:
    for j in range(cases):
        fsInt.append((final_states_int.count((i,j+1)), xValues[j], env.mapping()[i][1]))
        fsRdn.append((final_states_rdn.count((i,j+1)), xValues[j], env.mapping()[i][1]))
        if i==12 or i==13:
            redo.append((final_states_int.count((i,j+1)),xValues[j],env.mapping()[i][1],"Supervisory+RL"))
            redo.append((final_states_rdn.count((i,j+1)),xValues[j],env.mapping()[i][1],"Supervisory"))


        
list2Int=[]
redoInt=[]
redoRdn=[]
for i in range(0,9):
    list2Int.append((fsInt[i][1],fsInt[i][0], fsInt[i+9][0],fsInt[i+18][0],fsInt[i+27][0]))
    #redoInt.append((fsInt[i][1],fsInt[i+36][0],fsInt[i+45][0]))

list2Rdn=[]
for i in range(0,9):
    list2Rdn.append((fsRdn[i][1],fsRdn[i][0], fsRdn[i+9][0],fsRdn[i+18][0],fsRdn[i+27][0]))
    #redoRdn.append((fsRdn[i][1],fsRdn[i+36][0],fsRdn[i+45][0]))

data = np.vstack((info_int, info_rdn))
data = pd.DataFrame(data, columns=["mean reward",  xlabel_name, "method"])
states_int = pd.DataFrame(list2Int,columns=[xlabel_name,"Rejection Type 1","Rejection Type 2", "Approval Type 1","Approval Type 2"])
states_rdn = pd.DataFrame(list2Rdn, columns=[xlabel_name,"Rejection Type 1","Rejection Type 2", "Approval Type 1","Approval Type 2"])
redo = pd.DataFrame(redo, columns=["Number of Occurrences", xlabel_name, "Event", "Method"])
#redoRdn = pd.DataFrame(redoRdn, columns=[xlabel_name,"Rework Type 1","Rework Type 2"])

data.to_csv(reward_dataname)
states_int.to_csv(occurrences_int_dataname)
states_rdn.to_csv(occurrences_rnd_dataname)
redo.to_csv(occurrences_redo_int)
# redoRdn.to_csv(occurrences_redo_rdn)


intel = pd.read_csv(occurrences_int_dataname)
randomic = pd.read_csv(occurrences_rnd_dataname)
df = pd.read_csv(reward_dataname)
redo = pd.read_csv(occurrences_redo_int)
# redoRdn = pd.read_csv(occurrences_redo_rdn)


intel = intel.drop(["Unnamed: 0"],axis=1)
randomic = randomic.drop(["Unnamed: 0"], axis=1)
redo = redo.drop(["Unnamed: 0"], axis=1)
# redoRdn = redoRdn.drop(["Unnamed: 0"], axis=1)


plot = sns.lineplot(data=df, x=xlabel_name, y="mean reward", hue="method")

plt = intel.plot.bar(x=xlabel_name, stacked=True)
plt.set_xlabel(xlabel_name)
plt.set_ylabel("Number of Occurrences")

plt = randomic.plot.bar(x=xlabel_name, stacked=True)
plt.set_xlabel(xlabel_name)
plt.set_ylabel("Number of Occurrences")


sns.lineplot(x=xlabel_name, y="Number of Occurrences", style="Method", hue="Event", data=redo, markers=True)


# plt = redoInt.plot.bar(x=xlabel_name, stacked=True)
# plt.set_xlabel(xlabel_name)
# plt.set_ylabel("Number of Occurrences")

# plt = redoRdn.plot.line(x=xlabel_name, stacked=True)
# plt.set_xlabel(xlabel_name)
# plt.set_ylabel("Number of Occurrences")



