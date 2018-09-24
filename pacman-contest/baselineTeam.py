# -*- coding:utf-8 -*-
# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  返回两个Agent组成队伍，传入firstIndex和secondIndex是Index number。布尔值isRed
  是选择队伍颜色，first/second不知道是啥，和capture.py文件有关系。

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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  客户端选择actions

  A base class for reflex agents that chooses score-maximizing actions
  """
  
  def registerInitialState(self, gameState):
    """
    注册游戏开始状态，每次游戏只执行一次。红0，橙2，蓝1，青3。
    """
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    #print "my index=",self.index
    #print "my start=",self.start
    #print "getNumAgents()",gameState.getNumAgents()
    #print "getAgentState(0)",gameState.getAgentState(0)
    #print "getAgentState(1)",gameState.getAgentState(1)
    #print "getAgentState(2)",gameState.getAgentState(2)
    #print "getAgentState(3)",gameState.getAgentState(3)
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    选择Q（s,a）值最高的actions.
    """
    #当前位置可执行的actions= ['Stop', 'North', 'South']
    actions = gameState.getLegalActions(self.index)
    #print "actions=",actions
    #if self.index==3:
    #  print "********my current position=",gameState.getAgentPosition(3)
    # You can profile your evaluation time by uncommenting these lines
    #可以通过下面一行代码配置evaluation time？这是啥？
    # start = time.time()

    #与上面的actions列表相对应，返回当前state各动作的values=[-2038, -2037, -2039]
    values = [self.evaluate(gameState, a) for a in actions]
    #print "values=", values
    #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    #找到最大value和其对应的action,有可能两个action的value值相等，此时两个都返回。
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    #if self.index==3:
    #  print "bestActions=",bestActions

    #我方还有多少豆没吃
    foodLeft = len(self.getFood(gameState).asList())
    #print "foodLeft=",foodLeft

    if foodLeft <= 2:
      #print "triger!index=",self.index
      #print "current position=",gameState.getAgentPosition(self.index)
      bestDist = 9999
      for action in actions:
        #行动后的下个状态
        successor = self.getSuccessor(gameState, action)
        #行动后自身的位置
        pos2 = successor.getAgentPosition(self.index)
        #计算从出生位置，到下一步的位置，之间的maze距离
        dist = self.getMazeDistance(self.start,pos2)
        #选择离出生位置最近的那个的那个点？？为逃跑做准备？？Yes
        #他这个agent没有什么时间该回家的算法，所以，一直吃，吃到还剩两个豆，才想到要回家。
        #而且，对方如果把18个豆带回家，游戏就结束了。。。。
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      #print "bestAction=",bestAction
      return bestAction
    #print "getNumAgents()",gameState.getNumAgents()
    #print "getAgentState(0)",gameState.getAgentState(0)
    #print "getAgentState(1)",gameState.getAgentState(1)
    #print "getAgentState(2)",gameState.getAgentState(2)
    #print "getAgentState(3)",gameState.getAgentState(3)
    #print "isOver()",gameState.isOver()
    #if self.index==0:
      #print "getAgentDistances(),0",gameState.getAgentDistances()
      #print "getInitialAgentPosition(0)",gameState.getInitialAgentPosition(0)
      #print "getInitialAgentPosition(1)",gameState.getInitialAgentPosition(1)
      #print "getInitialAgentPosition(2)",gameState.getInitialAgentPosition(2)
      #print "getInitialAgentPosition(3)",gameState.getInitialAgentPosition(3)
    #if self.index==2:
      #print "getAgentDistances(),2",gameState.getAgentDistances()


    #最终随机返回一个最优action
    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    返回值是一个gamestate（游戏地图的网格数据）
    """
    #print "gameState=\n",gameState
    successor = gameState.generateSuccessor(self.index, action)
    #print "self.index=",self.index
    #print "successor=\n",successor
    pos = successor.getAgentState(self.index).getPosition()
    #print "pos=",pos
    #如果该位置没有豆？？？ 
    #print "nearestPoint(pos)=",nearestPoint(pos)
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    #print "run in evaluate"
    features = self.getFeatures(gameState, action)
    #print "run after here"
    
    weights = self.getWeights(gameState, action)
    #print "weights=", weights
    #if self.index==3:
    #  print "action=",action
    #  print "features=", features
    #  print "features * weights=",features * weights
    #向量点乘的积，a1*b1+a2*b2,【>0则方向基本相同，<0则方向相反】用不上这个
    return features * weights

  def getFeatures(self, gameState, action):
    """
    返回当前state的feature计数器
    Returns a counter of features for the state
    """
    #print "run in getFeatures"
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    #print "features=",features['successorScore']
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    #print "run in getWeights"
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  进攻agent
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    #print "run in here"
    #print "my index=",self.index
    #print "current position=",gameState.getAgentPosition(self.index)
    #print "self.start=",self.start
    features = util.Counter()
    #print "my action=",action
    successor = self.getSuccessor(gameState, action)
    #把矩阵转换成列表
    foodList = self.getFood(successor).asList()
    #successorSocre=-剩余豆
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    #print features
    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      #下一个位置的坐标
      myPos = successor.getAgentState(self.index).getPosition()
      #print "successor position",myPos
      #到最近的豆的maze距离
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    #最后features中有两项
    #一项，successorScore=-剩余豆数量
    #一项，distanceToFood=到最近豆的距离
    return features

  def getWeights(self, gameState, action):
    #吃豆的权重远大于寻找下一个最近豆
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  防御agent
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

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
