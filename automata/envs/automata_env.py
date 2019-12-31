import gym
from gym import error, spaces, utils
from gym.utils import seeding
import lxml.etree as et
import xml.dom.minidom
from XMLReader import parse
from pygv import plotSM
from IPython.display import display, Image

class automataEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
      ...
  def step(self, action):
      flag = True
      done = False
      for trans in self.transitions: 
        if(trans[0] == self.actual_state and trans[2] == action): 
            self.actual_state = trans[1]
            if(self.actual_state in self.terminal):
                done = True
            flag = False
      if(flag==True):
        print("Transição inválida")
      return self.actual_state, done
          
  def reset(self, filename):
    """
    Lê um arquivo XML obtido diretamente do software Supremica, e mapeia todos os estados 
    e transições obtidos a partir daquela estrutura de autômato.
    """
    self.actions, self.controllable, self.states, self.terminal, self.initial_state, self.transitions = parse(filename)
    self.observation_space = spaces.Discrete(len(self.states))
    self.action_space = spaces.Discrete(len(self.actions))
    self.actual_state = self.initial_state
    """
    TODO
    Criar todo o vetor de recompensas, que varia de acordo com o ambiente a ser simulado, 
    o ideal é criar uma lista na qual a recompensa do índice x seria relacionada à ação
    x. Após isso, a função step deve retornar também o valor da recompensa relacionada
    a ação que entrou como parâmetro naquela função.
    """
  def render(self, mode='human'):
    plotSM(self.states, self.terminal, self.initial_state,self.actual_state, self.actions, self.transitions, 800, 400)

  def mapping(self):
    """
    Informa para o usuário como estáo mapeados os eventos,
    qual id corresponde a qual label, etc
    """
    print(self.actions)