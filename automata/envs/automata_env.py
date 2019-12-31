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
      self.last_state = -1
  def step(self, action):
      flag = True
      done = False
      for trans in self.transitions: 
        if(trans[0] == self.actual_state and trans[2] == action): 
            self.last_state = self.actual_state
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
      """
      O tipo de layout ideal a ser usado depende do tamanho de cada autômato. Se for um tamanho muito grande,
      ou seja, acima de 100 estados, é recomendado utilizar "circo", embora não dê para visualizar muito bem
      no console, ao abrir a imagem é possível identificar os estados. Se o autômato for médio, entre 20 a 100 estados, 
      o recomendado é utilizar "dot", com um pouco de esforço é possível identificar as transições no console.
      Por fim, se o autômato tiver menos de 20 estados, o ideal é que seja renderizado com o programa "neato" ou "sfdp,
      sendo que considero o "sfdp" mais organizado para esse caso".
      """
      if(len(self.states) < 20):
          plotSM(self.states, self.terminal, self.initial_state,self.actual_state, self.last_state,
                 self.actions, self.transitions, 800, 400, prog='sfdp', sep=0.1)
          # padrão é "sfdp", porém pode ser utilizado "neato" também 
      elif(len(self.states) >= 20 and len(self.states) < 100):
          plotSM(self.states, self.terminal, self.initial_state,self.actual_state, self.last_state, 
                 self.actions, self.transitions, 800, 400, prog='dot', sep=0.1) 
      else:
          plotSM(self.states, self.terminal, self.initial_state,self.actual_state,self.last_state,
                 self.actions, self.transitions, 800, 400, prog='circo', sep=0.05)
          
    

  def mapping(self):
    """
    Informa para o usuário como estáo mapeados os eventos,
    qual id corresponde a qual label, etc
    """
    print(self.actions)