#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:25:46 2019

@author: kallil
"""
from IPython.display import Image, display
from XMLReader import parse
import pygraphviz as pgv

from pygraphviz import *

def plotSM(states, terminal, initial_state, actual_state, actions, transitions, width, height):
    
    G = pgv.AGraph(directed=True)
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
        elif(i in terminal):
            G.add_node(i, shape='doublecircle')
            if(i == actual_state):
                n = G.get_node(i)
                n.attr['color'] = 'gold'
        elif(i == actual_state):
            G.add_node(i, color='gold')
        else:
            G.add_node(i)
    
    for i in transitions:
        for j in actions:
            if(i[2] == j[0]):
                G.add_edge(i[0], i[1], label = str(j[1]))
        
    G.layout()
    G.draw('auto.png')
    G.draw('auto.dot')
    img = Image('auto.png')
    
    img.width = width
    img.height = height
    
    display(img)