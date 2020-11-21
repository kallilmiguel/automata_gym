import gym
from gym import spaces
from pygv import plotSM
from XMLReader import parse



class automataEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
      self.reward=None
      self.probs=None
      self.products=None
      
  def step(self, action):
      flag = True
      done = False
      for trans in self.transitions: 
          if(trans[0] == self.actual_state and trans[2] == action): 
            self.last_state = self.actual_state
            self.actual_state = trans[1]
            flag = False
            if(self.stop_crit==0):
                if(self.actual_state in self.terminal and self.last_action == action):
                    done = True
                 
            elif(self.stop_crit==1):
                if(action in self.last_action):
                    self.counter+=1
                if(self.counter==self.products):
                    done=True
            else:
                self.counter+=1
                if(self.counter >= self.last_action):
                    done=True
            break
      if(flag==True):
        print("Transição inválida - {},{}".format(self.actual_state, action))
        return self.actual_state, 0, done, {"prob" : 1.0}
      return self.actual_state, self.reward[action], done, {"prob" : 1.0}
  
  """
            
  Aqui é passado como parâmetro da função o nome do arquivo, a lista de recompensas, o 
  critério de parada para o treinamento e a última ação a ser adotada.
  Se caso o critério de parada for 0, o treinamento encerrará o episódio quando o autômato estiver
  no estado terminal e ao mesmo tempo ocorrer o evento definido como o parâmetro last_action.
  Para stop_crit=1, last action se da pelos eventos que fazem com que um produto saia da linha de producao
  e products sao quantos produtos serao produzidos.
  Caso o critério de parada seja 2, o episódio se encerrará quando forem executados x passos, 
  sendo x definido por last_action.
  """
  def reset(self, filename='', rewards=0, stop_crit=0, last_action=0, products=None, probs = None):
    self.last_state = -1
    self.counter = 0
    """
    Lê um arquivo XML obtido diretamente do software Supremica, e mapeia todos os estados 
    e transições obtidos a partir daquela estrutura de autômato.
    """
    if(filename):    
        self.actions, self.controllable, self.ncontrollable, self.states, self.terminal, self.initial_state, self.transitions = parse(filename)
        self.observation_space = spaces.Discrete(len(self.states))
        self.action_space = spaces.Discrete(len(self.actions))
        self.stop_crit = stop_crit
        if(last_action):           
            self.last_action = last_action
        else:
            last_action = self.last_action
    self.actual_state = self.initial_state
    """
    TODO
    Criar todo o vetor de recompensas, que varia de acordo com o ambiente a ser simulado, 
    o ideal é criar uma lista na qual a recompensa do índice x seria relacionada à ação
    x. Após isso, a função step deve retornar também o valor da recompensa relacionada
    a ação que entrou como parâmetro naquela função.
    O vetor de recompensas é na verdade a recompensa obtida por cada ação realizada. Por
    exemplo, se houver 4 eventos, o vetor de recompensa terá 4 índices, uma recompensa
    para cada ação realizada.
    """
    if(products):
        self.products=products
    else:
        products=self.products
            
    if(rewards):
        self.reward = rewards
    else:
        rewards = self.reward
    if(probs):
        self.probs = probs
    else:
        probs = self.probs
    return self.actual_state
    
  def render(self, mode='human'):
      """
      O tipo de layout ideal a ser usado depende do tapossible_transitions = []
    for transition in env.transitions:
        if(env.actual_state == transition[0]):
            possible_transitions.append(transition[2])
            manho de cada autômato. Se for um tamanho muito grande,
      ou seja, acima de 100 estados, é recomendado utilizar "circo", embora não dê para visualizar muito bem
      no console, ao abrir a imagem é possível identificar os estados. Se o autômato for médio, entre 20 a 100 estados, 
      o recomendado é utilizar "dot", com um pouco de esforço é possível identificar as transições no console.
      Por fim, se o autômato tiver menos de 20 estados, o ideal é que seja renderizado com o programa "neato" ou "sfdp,
      sendo que considero o "sfdp" mais organizado para esse caso".
      """
      if(len(self.states) < 20):
          plotSM(self.states, self.terminal, self.initial_state,self.actual_state, self.last_state,
                 self.actions, self.controllable, self.transitions, 800, 400, prog='sfdp', sep=0.1)
          # padrão é "sfdp", porém pode ser utilizado "neato" também 
      elif(len(self.states) >= 20 and len(self.states) < 100):
          plotSM(self.states, self.terminal, self.initial_state,self.actual_state, self.last_state, 
                 self.actions, self.controllable, self.transitions, 800, 400, prog='dot', sep=0.1) 
      else:
          plotSM(self.states, self.terminal, self.initial_state,self.actual_state,self.last_state,
                 self.actions, self.controllable, self.transitions, 800, 400, prog='circo', sep=0.05)
          
    

  def mapping(self, index=True):
    """
    Informa para o usuário como estáo mapeados os eventos,
    qual id corresponde a qual label, etc
    """
    if(index==True):
        return self.actions
    mp=[]
    for i in range(len(self.actions)):
        mp.append(self.actions[i][1])
    return mp
    

  def possible_transitions(self):
      possible_transitions = []
      for transition in self.transitions:
          if(self.actual_state == transition[0]):
              possible_transitions.append(transition[2])
              
      self.possible_space = spaces.Discrete(len(possible_transitions))
                
      return possible_transitions