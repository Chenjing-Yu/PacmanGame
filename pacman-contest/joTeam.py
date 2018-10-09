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
               first = 'TopAgent', second = 'BottomAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class GeneralAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  1. Three modes: attack, escape, defend
  attack: (current pacman/ghost) eat food
  escape: (current pacman) direct at home
  defend: (current ghost/pacman) try to catch the pacman on our side
  2. attack
  eat the nearest food
  3. escape *
  find a fast and safe path home
  4. defend
  go to catch the pacman when detecting its precise position (within 5 Manhattan distance)
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    # Sets if agent is on red team or not
    # if self.red:
    #   CaptureAgent.registerTeam(self, gameState.getRedTeamIndices())
    # else:
    #   CaptureAgent.registerTeam(self, gameState.getBlueTeamIndices())

    """self variables"""
    self.walls = gameState.getWalls()
    self.mapWidth = gameState.getWalls().width
    self.mapHeight = gameState.getWalls().height
    self.mapSize = self.mapWidth*self.mapHeight
    #agent的偏好，偏好行动范围
    self.favoredY = 0
    #index of ally 队友的index
    self.ally = self.index
    for i in self.getTeam(gameState):
      if self.index != i:
        self.ally = i
    #indexes of enemies
    self.enemies = self.getOpponents(gameState)
    #positions that are not walls on the other side敌方地盘的合法位置
    self.enemyPos = []
    #positions that are not walls on our side我方地盘的合法位置
    self.ourPos = []
    #escape goals -- the entrance positions on our side
    self.escapeGoals = []
    #corner depth: a map of position index and depth value, deeper corners have higher depths, 0 by default
    self.cornerDepth = util.Counter()
    self.beliefs={}
    #整张地图的可行动位置
    self.legalPositions = gameState.getWalls().asList(False)
    #地图中线的横坐标
    self.midWidth = gameState.data.layout.width / 2

    """initiate variables"""
    self.init(gameState)
    self.startBelief(gameState)

  def startBelief(self,gameState):
    '''游戏开始，初始化belief字典'''
    for enemy in self.enemies:
        self.beliefs[enemy] = util.Counter()
        self.beliefs[enemy][gameState.getInitialAgentPosition(enemy)] = 1.0

  def predictProbabilityOfPositionAfterAction(self,enemyIndex,gameState):
    new_belief = util.Counter()
    for p in self.legalPositions:
      # Get the new probability distribution.
      newPossitionPossibility = util.Counter()
      possiblePositions=[(p[0]+1,p[1]),(p[0]-1,p[1]),(p[0],p[1]),(p[0],p[1]+1),(p[0],p[1]-1)]
      for position in possiblePositions:
          if position in self.legalPositions:
              newPossitionPossibility[position] = 1.0
      newPossitionPossibility.normalize()
      for newPos, prob in newPossitionPossibility.items():
        new_belief[newPos] += prob * self.beliefs[enemyIndex][p]
    new_belief.normalize()
    self.beliefs[enemyIndex] = new_belief

  def computeProbabilityDistribution(self,enemyIndex,gameState):
    noisyDistance=gameState.getAgentDistances()[enemyIndex]
    myPos = gameState.getAgentPosition(self.index)
    new_belief = util.Counter()
    for p in self.legalPositions:
      trueManhattanDistance = util.manhattanDistance(myPos, p)
      confidence = gameState.getDistanceProb(trueManhattanDistance, noisyDistance)
      #排除不可能的概率
      if self.red:
        pac = p[0] < self.midWidth
      else:
        pac = p[0] > self.midWidth
      #1，对方不可能在范围5之内，因为他如果在，我就已经得到它得精确距离了。
      if trueManhattanDistance <= 5:
        new_belief[p] = 0.
      #2，通过敌方agent的身份，判断其不可能在地图左/右侧
      elif pac != gameState.getAgentState(enemyIndex).isPacman:
        new_belief[p] = 0.
      else:
        new_belief[p] = self.beliefs[enemyIndex][p] * confidence
    if new_belief.totalCount() == 0:
      self.beliefs[enemyIndex] = util.Counter()
      for p in self.legalPositions:
        self.beliefs[enemyIndex][p] = 1.0
      self.beliefs[enemyIndex].normalize()
    else:
      new_belief.normalize()
      self.beliefs[enemyIndex] = new_belief

  def getMostLikelyPosition(self,enemyIndex,gameState):
    self.predictProbabilityOfPositionAfterAction(enemyIndex,gameState)
    self.computeProbabilityDistribution(enemyIndex,gameState)
    print "the most likely position of enemy=",enemyIndex,"is",self.beliefs[enemyIndex].argMax()
    return self.beliefs[enemyIndex].argMax()

  def getMostLikelyManhattanDistance(self,enemyIndex,gameState):
    enemyPos=self.getMostLikelyPosition(enemyIndex,gameState)
    myPos = gameState.getAgentPosition(self.index)
    return util.manhattanDistance(myPos, enemyPos)

  def getMostLikelyMazeDistance(self,enemyIndex,gameState):
    enemyPos=self.getMostLikelyPosition(enemyIndex,gameState)
    myPos = gameState.getAgentPosition(self.index)
    print "myIndex=",self.index
    print "maze distance to enemy=",enemyIndex,"is",self.getMazeDistance(enemyPos,myPos)
    return self.getMazeDistance(enemyPos,myPos)


  def init(self, gameState):
    #escape goals
    x = self.mapWidth/2 - 1
    if not self.red:
      x = x + 1
    for y in range(1, self.mapHeight-1):
      if (x, y) in self.walls.asList(False):
        self.escapeGoals.append((x, y))

    #enemyPos, ourPos, cornerDepth
    for y in range(1, self.mapHeight-1):
      for x in range(1, self.mapWidth/2):
        if (x,y) in self.walls.asList(False):
          if self.red:
            self.ourPos.append((x,y))
          else:
            self.enemyPos.append((x,y))
            self.cornerDepth[(x,y)] = 0
      for x in range(self.mapWidth/2, self.mapWidth-1):
        if (x,y) in self.walls.asList(False):
          if self.red:
            self.enemyPos.append((x,y))
            self.cornerDepth[(x,y)] = 0
          else:
            self.ourPos.append((x,y))

    #cornerDepth
    # self.calculateDeadCorners(gameState)
    # distributions = [self.cornerDepth, ]
    # self.displayDistributionsOverPositions(distributions)
  #END of init

  def calculateDeadCorners(self, gameState):
    """
    calculate the dead corners and mark them with risk value.
    efault value is 0, which means not a dead corner,
    deeper dead corners are marked with higher value.
    self.cornerDepth: a map of position index and risk value
    """
    #log
    # print("rewards")
    # print(self.rewards)
    # print(len(self.rewards))
    # print(self.cornerDepth)
    # print(self.escapeGoals)

    marks = util.Counter() # map(position: mark), mark increases by 1 for each path
    for dot in self.enemyPos:
      visited = set()
      count = 0 # num of paths from the dot to escape goals
      aPath = [] # sequence of positions in any one path
      marks.clear()
      stack = util.Stack()
      stack.push((dot, [dot,]))
      while not stack.isEmpty():
        pos, path = stack.pop()
        visited.add(pos)
        if pos in self.escapeGoals:
          count = count + 1
          aPath = []
          for x, y in path:
            aPath.append((x,y))
            if (x,y) in marks:
              marks[(x,y)] = marks[(x,y)]+1
            else:
              marks[(x,y)] = 1
        else:
          successors = [(pos[0]+1, pos[1]), (pos[0]-1, pos[1]), (pos[0], pos[1]+1), (pos[0], pos[1]-1)]
          for next in successors:
            if next not in visited:
              if next in self.rewards:
                path.append(next)
                stack.push((next, path))
      if count > 0:
        depth = 0
        for pos, mark in marks:
          if mark == count:
            depth = depth + 1
        if depth > 1:
          for d in range(1, depth):
            self.cornerDepth[path[d-1]] = max(depth-d, self.cornerDepth[path[d-1]])

  def getEnemyPositions(self, gameState):
    positions = []
    for i in self.enemies:
      pos = gameState.getAgentPosition(i)
      if pos != None:
        positions.append((i, pos))
      else:
        self.getMostLikelyPosition(i,gameState)
    return positions

  def getEnemyDistances(self, myPos, gameState):
    """
    return the (index,distance)s of enemies
    """
    dists = []
    for i in self.enemies:
      pos = gameState.getAgentPosition(i)
      if pos == None:
        #dists.append((i, gameState.getAgentDistances()[i]))#todo: belief most likely distance
        dists.append((i, self.getMostLikelyMazeDistance(i,gameState)))
      else:
        dists.append((i, self.getMazeDistance(myPos, pos)))
    return dists

  def getNearestGhost(self, myPos, gameState):
    minDistance = 999999
    ghost = self.enemies[0]
    for enemy in self.enemies:
      if not gameState.getAgentState(enemy).isPacman:
        pos = gameState.getAgentPosition(enemy)
        if pos != None:
          dist = self.getMazeDistance(myPos, pos)
          if dist < minDistance:
            ghost = enemy
            minDistance = dist
        else:
          gameState.getAgentDistances()[enemy]#todo: belief most likely distance
    return (ghost, minDistance)

  def getNearestPacman(self, myPos, gameState):
    minDistance = 999999
    ghost = self.enemies[0]
    for enemy in self.enemies:
      if gameState.getAgentState(enemy).isPacman:
        pos = gameState.getAgentPosition(enemy)
        if pos != None:
          dist = self.getMazeDistance(myPos, pos)
          if dist < minDistance:
            ghost = enemy
            minDistance = dist
        else:
          gameState.getAgentDistances()[enemy]#todo: belief most likely distance
    return (ghost, minDistance)

  def getDistanceToAlly(self, myPos, gameState):
    """
    return distance to ally
    """
    return self.getMazeDistance(myPos, gameState.getAgentState(self.ally).getPosition())

  def getFavoredFoodDistance(self, myPos, food):
    return self.getMazeDistance(myPos, food) + abs(self.favoredY - food[1])

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    选择Q（s,a）值最高的actions.
    """
    actions = gameState.getLegalActions(self.index)
    #  print "********my current position=",gameState.getAgentPosition(3)
    # start = time.time()
    #score = self.getScore(gameState)
    #this agent
    myPos = gameState.getAgentPosition(self.index)
    isPacman = gameState.getAgentState(self.index).isPacman
    scaredTimer = gameState.getAgentState(self.index).scaredTimer
    carrying = gameState.getAgentState(self.index).numCarrying
    foodLeft = len(self.getFood(gameState).asList())
    carryLimit = 5
    #enemy
    enemyScaredTimer = min([gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies])
    enemyPositions = self.getEnemyDistances(myPos, gameState)
    minDistance = 999999
    for i, dist in enemyPositions:
      minDistance = min(minDistance, dist)

    mode = 'attack'
    #only 2 food left, just go back home
    if foodLeft <= 2:
      mode = 'escape'
    #enemy is in 5 steps
    if minDistance <= 5:
      #as a pacman, may escape depending on whether enemy is scared
      if isPacman:
        if enemyScaredTimer <= 5:
          mode = 'escape'
      #as a ghost, defend only if not so scared
      elif scaredTimer == 0:
        mode = 'defend'
      elif scaredTimer <= 5:
        mode = 'defend'#todo: follow?
    #enemy not visible
    elif carrying > carryLimit and enemyScaredTimer < 5:
      mode = 'escape'#todo: not just escape, but going towards middle, still eating food

#todo: escape mode, use A* to find the path, and take the first action
    values = [self.evaluate(gameState, a, mode) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction


    #最终随机返回一个最优action
    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    返回值是一个gamestate（游戏地图的网格数据）
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action, mode):
    """
    Computes a linear combination of features and feature weights
    """
    if mode == "attack":
      features = self.getAttackFeatures(gameState, action)
      weights = self.getAttackWeights(gameState, action)
    elif mode == "escape":
      features = self.getEscapeFeatures(gameState, action)
      weights = self.getEscapeWeights(gameState, action)
    elif mode == "defend":
      features = self.getDefendFeatures(gameState, action)
      weights = self.getDefendWeights(gameState, action)
    # print(features)
    # print(weights)

    return features * weights

  def getAttackFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    features['successorScore'] = -len(foodList) + self.getScore(successor)
    # Compute distances to the food in favored zone
    if len(foodList) > 0:
      minDistance = min([self.getMazeDistance(myPos, food)+abs(self.favoredY-food[1]) for food in foodList])
      features['distanceToFood'] = minDistance
      #min([self.getFavoredFoodDistance(myPos, food) for food in foodList])
    # compute distance to the nearest capsule
    capsules = self.getCapsules(successor)
    if len(capsules) > 0:
      minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
      features['distanceToCapsule'] = minDistance
      features['pickupCapsule'] = -len(capsules)
    # distance to the escape goals
    minDistance = min([self.getMazeDistance(myPos, home) for home in self.escapeGoals])
    features['distanceToEscape'] = minDistance
    #distance to ally
    if self.index == self.getTeam(gameState)[0]:
      features['distanceToAlly'] = self.getDistanceToAlly(myPos, gameState)
      # distance to the enemy
    features['distanceToEnemy'] = self.getNearestGhost(myPos, gameState)[1]
    # is it a dead corner? corner depth
      #todo
    # stop
    if action == Directions.STOP:
      features['stop'] = 1

    return features

  def getAttackWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1000, 'distanceToFood': -100, 'distanceToCapsule': -100, 'distanceToEscape': 0,
            'distanceToAlly': -100, 'distanceToEnemy': 500, 'stop': -1000, 'pickupCapsule': 1000}

  def getDefendFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
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

  def getDefendWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -100, 'stop': -100, 'reverse': -2}

  def getEscapeFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're safe
    features['isSafe'] = 1
    if myState.isPacman: features['isSafe'] = 0

    # Computes distance to ghosts we can see
    enemies = [successor.getAgentState(i) for i in self.enemies]
    chasers = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
    features['numChasers'] = len(chasers)

    if len(chasers) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in chasers]
      features['chaserDistance'] = min(dists)
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

    if action == rev: features['reverse'] = 1

    return features

  def getEscapeWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'numChasers': -1000, 'isSafe': 2000, 'chaserDistance': -1000, 'stop': -1000, 'reverse': -1}

class TopAgent(GeneralAgent):

  def registerInitialState(self, gameState):
    GeneralAgent.registerInitialState(self, gameState)
    self.favoredY = gameState.data.layout.height

# Leeroy Bottom Agent - favors pellets with a lower y
class BottomAgent(GeneralAgent):

  def registerInitialState(self, gameState):
    GeneralAgent.registerInitialState(self, gameState)
    self.favoredY = 0
