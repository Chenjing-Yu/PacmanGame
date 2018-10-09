#coding=utf-8

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
from capture import GameState

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               #first = 'attackAgent', second = 'defenseAgent'):
               first = 'attackAgent', second = 'attackAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class rootAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  initialFoodNum=0
  lastGameState=None
  lastPosition=None
  foodCarried=0
  currentX=0
  currentY=0

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)可用该方法计算两点之间的maze距离

    该方法最多执行15秒
    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    #print "ddd"
    '''
    Your initialization code goes here, if you need any.
    '''
  def chooseAction(self, gameState):
    """选择当前状态，可执行的value最大的action"""
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    #print gameState.getAgentState(self.index).getPosition()[0],"position"

    """计算Pacman自身携带的豆数"""
    if gameState.getAgentState(self.index).isPacman:
      Position=gameState.getAgentState(self.index).getPosition()
      self.currentX=int(Position[0])
      self.currentY=int(Position[1])
      #print self.currentX,self.currentY
      if self.lastGameState!=None:
        if self.lastGameState.hasFood(self.currentX,self.currentY):
          self.foodCarried+=1
    else:
      self.foodCarried=0
    self.lastGameState=gameState
    #if self.index==1:
      #print "bestActions=",bestActions

    return random.choice(bestActions)

  def evaluate(self, gameState, action):
    """计算value的方法，两个向量点乘a1*b1+a2*b2+...."""
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    #if self.index==1:
      #print action
      #print features
      #print weights
      #print "value=",features*weights
    return features * weights

  def getSuccessor(self, gameState, action):
    """传入当前状态，action。返回执行action后的下个状态。"""
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    #貌似是校准位置到地图网格
    #Finds the nearest grid point to a position (discretizes).
    if pos != nearestPoint(pos):
      #print "run in here,pos=",pos,"nearestPoint(pos)",nearestPoint(pos)
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def getFeatures(self, gameState, action):
    return
  def getWeights(self, gameState, action):
    return

class attackAgent(rootAgent):
  """进攻Agent类，继承自rootAgent"""

  #地图中按照是否连续，为food划分区域
  foodRegion=[]

  def registerInitialState(self, gameState):

    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    
    foodList=self.getFood(gameState).asList()
    """给地图上的food分区"""
    #循环得到foodRegion，列表嵌套列表再嵌套元组，每一个基本元素是一个food坐标元组
    while(len(foodList)!=0):
      food=foodList[0]
      region=self.findNeighbourFood(food,foodList)
      #print region
      self.foodRegion.append(region)
      foodList=list(set(foodList)-set(region))

  def findNeighbourFood(self,food1,foodList):
    """传入一个food坐标，返回与其相连的所有food，返回值是list嵌套坐标元组"""
    region=[]
    region+=[food1];
    foodList.remove(food1)
    for food2 in foodList:
      if food2[0]==food1[0] and (food2[1]==food1[1]+1 or food2[1]==food1[1]-1):
        region+=self.findNeighbourFood(food2,foodList)
      if food2[1]==food1[1] and (food2[0]==food1[0]+1 or food2[0]==food1[0]-1):
        region+=self.findNeighbourFood(food2,foodList)
    return region
    """结束分区"""

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    """吃豆"""
    #把矩阵转换成列表,每个元素都是一个food坐标，int
    foodList = self.getFood(successor).asList()
    #successorSocre=-剩余豆
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      #下一个位置的坐标
      myPos = successor.getAgentState(self.index).getPosition()
      #print myPos,"myPos"
      #到最近的豆的maze距离
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    """躲避对方防守怪物"""
    #getOpponents,获得对手的index
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    #print enemies
    #如果这个对手agent身份是怪物，且位置可见
    enemyGhost = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    enemyPosition= [a for a in enemies if a.getPosition() != None]
    #可见的对方吃豆人数量,通过设定该值，可让吃豆人在怪物状态下，顺路杀死对方吃豆人。
    features['numEnemy'] = len(enemyPosition)
    if len(enemyGhost) > 0:
      sumDistance=0
      for invader in enemyGhost:
        distance=self.getMazeDistance(myPos, invader.getPosition())
        sumDistance+=distance
      #离怪物的距离越远越好
      if sumDistance<5:
        features['sumDistance']=sumDistance
      #print features

    """不建议原地不动"""
    if action == Directions.STOP: features['stop'] = 1

    """防止原地打转"""
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    """回家放豆"""
    if self.foodCarried>0:
      disToHome=min([self.getMazeDistance(myPos, point) for point in [(16,1),(16,4),(16,7),(16,10),(16,14)]])
    
      if features['sumDistance']!=0:
        features['unload']=self.foodCarried*(50-disToHome)*10
      else:
        features['unload']=self.foodCarried*(50-disToHome)*0.1

    return features

  def getWeights(self, gameState, action):
    #吃豆的权重远大于寻找下一个最近豆,躲避怪物最重要，原地不动=等死
    return {'successorScore': 100, 'distanceToFood': -1,'numDefense':-1000,
            'sumDistance':200,'stop':-2000,'unload':1,'reverse':-25}


class defenseAgent(rootAgent):

  def registerInitialState(self, gameState):
 
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    #行动后，自身下一回合的位置
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)，我是怪物=1，我是吃豆人=0
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see计算我和可见的敌人的距离
    #getOpponents,获得对手的index
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    #如果这个对手agent身份是吃豆人，且位置可见
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    
    #可见的对方吃豆人数量
    features['numInvaders'] = len(invaders)
    #print "--------------",features['numInvaders']
    if len(invaders) > 0:
      #print "action",action
      
      #print "myPos=",myPos
      #print "a.getPosition=",invaders[0].getPosition()
      #获得自身到每个敌人的maze距离
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      #print "dists=",dists
      #自身到最近的敌人的maze距离
      features['invaderDistance'] = min(dists)
    #如果当前选择的动作是stop
    if action == Directions.STOP: features['stop'] = 1
    #gameState.getAgentState(self.index)返回：Ghost: (x,y)=(30.0, 12.0), South
    #rev就返回上一行，最后那个方向的反方向。North
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    #print "self.index",self.index
    #print "gameState\n",gameState
    #print "gameState.getAgentState(self.index)\n",gameState.getAgentState(self.index)
    #print "rev\n",rev

    #如果我当前选择的action是我运动的反方向
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    #改成+1000，怪物不敢杀吃豆人了……
    #return {'numInvaders': 1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
    #{可见的对方吃豆人数量，我是怪物1/吃豆人0，我离最近的敌方吃豆人的maze距离，我选的action是stop，如果我选的action和我运动方向相反}
    #{不希望发现对方吃豆人？？}

