#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:25:46 2019

@author: kallil
"""
from IPython.display import Image, display
from XMLReader import parse
import pygraphviz as pgv

def plotSM(states, terminal, initial_state, actual_state, last_state, actions, transitions, width, height, prog, sep):
    
    G = pgv.AGraph(format='svg', directed=True, nodesep=sep)
    G.add_node(-1, shape='point')
    
    for i in states:
        if(i== initial_state):
            G.add_node(i)
            G.add_edge(-1, i)
            if(i in terminal):
                n = G.get_node(i)
                n.attr['shape'] = 'doublecircle'
            if(i == actual_state):
                n = G.get_node(i)
                n.attr['color'] = 'gold'
            elif(i==last_state):
                n = G.get_node(i)
                n.attr['color'] = 'purple'
        elif(i in terminal):
            G.add_node(i, shape='doublecircle')
            if(i == actual_state):
                n = G.get_node(i)
                n.attr['color'] = 'gold'
            elif(i==last_state):
                n = G.get_node(i)
                n.attr['color'] = 'purple'
        elif(i == actual_state):
            G.add_node(i, color='gold')
        elif(i==last_state):
            G.add_node(i, color='purple')
        else:
            G.add_node(i)
    
    for i in transitions:
        for j in actions:
            if(i[2] == j[0]):
                if(i[0] == last_state and i[1] == actual_state):
                    G.add_edge(i[0], i[1], label = str(j[1]), color = 'purple')
                else:
                    G.add_edge(i[0], i[1], label = str(j[1]))                    
                
        
    G.layout(prog=prog)
    G.draw('croppedAutomata.png')
    img = Image('croppedAutomata.png')
    
    img.width = width
    
    display(img)