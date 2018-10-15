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
    #agent preference area in the map(top/bottom)
    self.favoredY = 0
    #index of ally 
    self.ally = self.index
    for i in self.getTeam(gameState):
      if self.index != i:
        self.ally = i
    #indexes of enemies
    self.enemies = self.getOpponents(gameState)
    #positions that are not walls on the other side
    self.enemyPos = []
    #positions that are not walls on our side
    self.ourPos = []
    #escape goals -- the entrance positions on our side
    self.escapeGoals = []
    #corner depth: a map of position index and depth value, deeper corners have higher depths, 0 by default
    self.cornerDepth = util.Counter()
    self.beliefs={}
    #legal Positions in whole map
    self.legalPositions = gameState.getWalls().asList(False)
    # middele coordinate in the map
    self.midWidth = gameState.data.layout.width / 2
    self.lastTurnFoodList=self.getFoodYouAreDefending(gameState).asList()
    #a list of enemy's (index, pos, isPacman, scaredTimer), fetched in each chooseAction
    self.enemyInfo = []
    self.Timer = 300

    """initiate variables"""
    self.init(gameState)
    self.startBelief(gameState)

  def startBelief(self,gameState):
    '''game start，initialize belief dictionary'''
    for enemy in self.enemies:
      self.beliefs[enemy] = util.Counter()
      self.beliefs[enemy][gameState.getInitialAgentPosition(enemy)] = 1.0

  def getExpectedValue(self,enemyIndex,gameState):
    new_belief = util.Counter()
    for p in self.legalPositions:
      # Get the new probability distribution.
      newPossitionPossibility = util.Counter()
      possiblePositions=[(p[0]+1,p[1]),(p[0]-1,p[1]),(p[0],p[1]),(p[0],p[1]+1),(p[0],p[1]-1)]
      for position in possiblePositions:
        if position in self.legalPositions:
          newPossitionPossibility[position] = 1.0
      newPossitionPossibility.normalize()
      #MDP
      for newPos, prob in newPossitionPossibility.items():
        new_belief[newPos] += prob * self.beliefs[enemyIndex][p]
    new_belief.normalize()
    self.beliefs[enemyIndex] = new_belief

  def computeProbabilityDistribution(self,enemyIndex,gameState):
    noisyDistance=gameState.getAgentDistances()[enemyIndex]
    myPos = gameState.getAgentPosition(self.index)
    new_belief = util.Counter()

    for pos in self.legalPositions:
      trueManhattanDistance = util.manhattanDistance(myPos, pos)
      confidence = gameState.getDistanceProb(trueManhattanDistance, noisyDistance)
      #exclude some impossible positions
      if self.red:
        pac = pos[0] < self.midWidth
      else:
        pac = pos[0] > self.midWidth
      #1，if enemy in 5 distance, we have got his accurate position
      if trueManhattanDistance <= 5:
        new_belief[pos] = 0.
      #2，judge by enemy's status, he cannot in left/right of map
      elif pac != gameState.getAgentState(enemyIndex).isPacman:
        new_belief[pos] = 0.
      else:
        new_belief[pos] = self.beliefs[enemyIndex][pos] * confidence

    if new_belief.totalCount() == 0:
      self.beliefs[enemyIndex] = util.Counter()
      for pos in self.legalPositions:
        self.beliefs[enemyIndex][pos] = 1.0
      self.beliefs[enemyIndex].normalize()
    else:
      new_belief.normalize()
      self.beliefs[enemyIndex] = new_belief

  def getMostLikelyPosition(self,enemyIndex,gameState):
    self.getExpectedValue(enemyIndex,gameState)
    self.computeProbabilityDistribution(enemyIndex,gameState)
    #print "the most likely position of enemy=",enemyIndex,"is",self.beliefs[enemyIndex].argMax()
    currentFoodList = self.getFoodYouAreDefending(gameState).asList()
    isAttacking = False # is there any pacman attacking our food
    status = self.getEnemyStatusList(gameState) # status (true for pacman, false for ghost) list of enemies
    numOfAttackingEnemy = 0
    enemyInd = self.enemies[0]
    for enemy in status:
      if enemy[1]:
        isAttacking = True
        numOfAttackingEnemy += 1
        enemyInd = enemy[0]
    # if enemey come in to our area
    if isAttacking: 
      #print "-----------------------attacking-----------------"
      #if num of enemy is one, then his index can be locked
      if numOfAttackingEnemy == 1:
        #print"----------------------number of enemy is 1", update his beleif dictionary
        if len(currentFoodList) < len(self.lastTurnFoodList):
          #print "==========================",list(set(self.lastTurnFoodList)-set(currentFoodList))
          #print currentFoodList
          #print self.lastTurnFoodList
          foodBeEaten = list(set(self.lastTurnFoodList)-set(currentFoodList))[0]
          newBel = util.Counter()
          newBel[foodBeEaten] = 1.0
          self.beliefs[enemyInd] = newBel
          #print "---------------------one food have been eated---------------"
    return self.beliefs[enemyIndex].argMax()

  def getMostLikelyManhattanDistance(self,enemyIndex,gameState):
    enemyPos=self.getMostLikelyPosition(enemyIndex,gameState)
    myPos = gameState.getAgentPosition(self.index)
    return util.manhattanDistance(myPos, enemyPos)

  def getMostLikelyMazeDistance(self,enemyIndex,gameState):
    enemyPos=self.getMostLikelyPosition(enemyIndex,gameState)
    myPos = gameState.getAgentPosition(self.index)
    #print "myIndex=",self.index
    #print "maze distance to enemy=",enemyIndex,"is",self.getMazeDistance(enemyPos,myPos)
    return self.getMazeDistance(enemyPos,myPos)


  def init(self, gameState):
    '''calculate map data at game initialization'''
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
        new_belief = util.Counter()
        #get enemy accurate position, update belief
        new_belief[pos] = 1.0
        self.beliefs[i] = new_belief
      else:
        pos=self.getMostLikelyPosition(i,gameState)
        positions.append((i,pos))


    return positions

  def getEnemyDistances(self, myPos, gameState):
    """
    return the (index,distance)s of enemies
    """
    dists = []
    for i, pos, isPacman, scaredTimer in self.enemyInfo:
      dists.append(self.getMazeDistance(myPos, pos))
    return dists


  def getEnemyInfo(self, gameState):
    """
    return a list of enemies with info of (index, position, isPacman, scaredTimer)
    position won't be none
    """
    info = []
    for i in self.enemies:
      pos = gameState.getAgentPosition(i)
      if pos != None:
        new_belief = util.Counter()
        #get enemy accurate position, update belief
        new_belief[pos] = 1.0
        self.beliefs[i] = new_belief
      else:
        pos = self.getMostLikelyPosition(i, gameState)
      info.append((i, pos, gameState.getAgentState(i).isPacman, gameState.getAgentState(i).scaredTimer))
    self.enemyInfo = info

  def getNearestGhost(self, myPos, gameState):
    minDistance = 999999
    ghost = self.enemies[0]
    for enemy in self.enemies:
      if not gameState.getAgentState(enemy).isPacman:
        pos = gameState.getAgentPosition(enemy)
        if pos != None:
          dist = self.getMazeDistance(myPos, pos)
        else:
          dist = self.getMostLikelyMazeDistance(enemy, gameState)
        if dist < minDistance:
          ghost = enemy
          minDistance = dist
    return (ghost, minDistance)

  def getNearestPacman(self, myPos, gameState):
    minDistance = 999
    pacman = self.enemies[0]
    for enemy in self.enemies:
      if gameState.getAgentState(enemy).isPacman:
        pos = gameState.getAgentPosition(enemy)
        if pos != None:
          dist = self.getMazeDistance(myPos, pos)
        else:
          dist = self.getMostLikelyMazeDistance(enemy, gameState)
        if dist < minDistance:
          pacman = enemy
          minDistance = dist
    return (pacman, minDistance)

  def getDistanceToAlly(self, myPos, gameState):
    """
    return distance to ally
    """
    return self.getMazeDistance(myPos, gameState.getAgentState(self.ally).getPosition())

  def allyIsPacman(self,gameState):
    #print gameState.getAgentState(self.ally).isPacman
    return gameState.getAgentState(self.ally).isPacman

  def getEnemyStatusList(self,gameState):
    '''return enemyIndex and if his status is pacman，[(0,ture),(2,true)]'''
    status=[]
    status.append((self.enemies[0],gameState.getAgentState(self.enemies[0]).isPacman))
    status.append((self.enemies[1],gameState.getAgentState(self.enemies[1]).isPacman))
    return status

  # def enemyAttacking(self,gameState):
  #   status=self.getEnemyStatusList(gameState)
  #   for enemy in status:
  #     if enemy[1]:
  #       #print "EnemyAttacking"
  #       return True
  #   return False


  def getFavoredFoodDistance(self, myPos, food):
    return self.getMazeDistance(myPos, food) + abs(self.favoredY - food[1])

  def chooseMode(self,gameState, myPos):
    # start = time.time()
    """food list"""
    self.foodList = self.getFood(gameState).asList() #food (positions) left to eat
    foodLeft = len(self.foodList)
    if foodLeft>0:
      nearestFoodDistance=min([self.getMazeDistance(myPos, food)for food in self.foodList])
    else:
      nearestFoodDistance=9999
    carryLimit = 5 #limit of food to carry, retreat when carrying more food
    #score = self.getScore(gameState) #current score
    """this agent"""
    isPacman = gameState.getAgentState(self.index).isPacman #is pacman or ghost
    myScaredTimer = gameState.getAgentState(self.index).scaredTimer #left time of scared status (0 if not scared)
    carrying = gameState.getAgentState(self.index).numCarrying #number of food this agent is carrying
    """enemy information"""
    enemyAttacking = False #true if there's an enemy which is pacman
    minDistance = 999999 #distance to the nearest enemy (not only ghost)
    enemyScaredTimer = 0 #nereast enemy's scared timer
    for eIndex, ePos, ePacman, eScaredTimer in self.enemyInfo:
      #get minimum distance to enemy and the enemy's scaredTimer
      dist = self.getMazeDistance(myPos, ePos)
      if dist < minDistance:
        minDistance = dist
        enemyScaredTimer = eScaredTimer
      if ePacman:
        enemyAttacking = True

    #set agent mode，default=attack
    mode = 'attack'

    #two food left, just go home
    if foodLeft <= 2:
      return 'goHome'

    if self.Timer <= min([self.getMazeDistance(myPos, p) for p in self.escapeGoals]) + 5 and carrying > 0:
      return 'goHome'

    if(enemyAttacking):
      #if i am not scared
      if(myScaredTimer==0):
        #if i am a ghost
        if(not isPacman):
          return "defend"
        #if i am a pacman
        else:
          #if my ally is a ghost
          if(not self.allyIsPacman(gameState)):
            pass
          #if i and ally are both pacman
          else:
            #if i carring more food than ally
            if(carrying>gameState.getAgentState(self.ally).numCarrying):
              return 'goHome'#"goHome"
            if(carrying==gameState.getAgentState(self.ally).numCarrying):
              if (self.index>self.ally):
                return 'goHome'#"goHome"
      #if i am scared, just be a pacman and eat foot
      else:
        mode="attack"

    #if enemy in 5 distance
    if minDistance <= 5:
      #if i am a pac man 
      if isPacman:
        if enemyScaredTimer <= 5:
          if self.getNearestGhost(myPos,gameState)[1]>nearestFoodDistance+1:
            return 'attack'
          else:
            mode = 'retreat'
      #as a ghost, defend only if not so scared
      elif myScaredTimer == 0:
        mode = 'defend'
      elif myScaredTimer < 5:
        mode = 'defend'#todo: follow?


    #if enemy distance > 5
    #if i carring many food
    elif carrying > carryLimit and enemyScaredTimer < 5:
      #if safe, keep eating food
      if self.getNearestGhost(myPos,gameState)[1]>nearestFoodDistance+1:
        return 'attack'
      #else,go home.
      else:
        mode = 'retreat'#"goHome"#todo: not just escape, but going towards middle, still eating food
    return mode

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest value.
    """
    self.Timer = self.Timer - 1
    self.getEnemyInfo(gameState) # refresh enemyInfo, DO NOT DELETE
    actions = gameState.getLegalActions(self.index)
    myPos = gameState.getAgentPosition(self.index)
    mode = self.chooseMode(gameState, myPos) 

    #print "index=",self.index,"mode=",mode,"Position=",myPos

    """A star for escape and retreat"""
    if mode == "goHome":
      return self.astar(gameState, myPos, mode, self.escapeGoals)
    if mode == "retreat":
      capsules = self.getCapsules(gameState)
      return self.astar(gameState, myPos, mode, self.escapeGoals+capsules)

    """values for attack and defend"""
    values = [self.evaluate(gameState, a, mode) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    #update the foodList i monitoring
    currentFoodList = self.getFoodYouAreDefending(gameState).asList()
    self.lastTurnFoodList=list(currentFoodList)
    return random.choice(bestActions)

  def astar(self, gameState, myPos, mode, goals):
    open = util.PriorityQueue()
    closed = set()
    paths = {myPos: [], }
    open.push(myPos, 0)
    escapePath = []
    while open:
      pos = open.pop()
      actions = paths[pos]
      cost = len(actions)
      if pos in goals:
        escapePath = paths[pos]
        break
      if pos not in closed:
        closed.add(pos)
        successors = self.legalSuccessors(pos)
        for nextpos, nextaction in successors:
          nextActions = actions + [nextaction]
          nextCost = cost + self.heuristic(gameState, nextpos, mode)
          if nextpos not in closed:
            if nextpos in paths:
              if nextCost < len(paths[nextpos]) + self.heuristic(gameState, nextpos, mode):
                open.update(nextpos, nextCost)
                paths[nextpos] = nextActions
            else:
              open.push(nextpos, nextCost)
              paths[nextpos] = nextActions

    #print "escapeAction() time",time.time()-t1
    if len(escapePath) == 0:
      #print "no way to escape"
      
      return Directions.STOP
    else:
      return escapePath[0]

  def legalSuccessors(self, pos):
    x = pos[0]
    y = pos[1]
    successors = []
    if (x+1, y) in self.legalPositions:
      successors.append(((x+1, y), Directions.EAST))
    if (x-1, y) in self.legalPositions:
      successors.append(((x-1, y), Directions.WEST))
    if (x, y-1) in self.legalPositions:
      successors.append(((x, y-1), Directions.SOUTH))
    if (x, y+1) in self.legalPositions:
      successors.append(((x, y+1), Directions.NORTH))
    return successors

  def heuristic(self, gameState, mypos, mode):
    t1=time.time()
    goalDist = min([self.getMazeDistance(mypos, p) for p in self.escapeGoals]) # min distance to escape goals
    enemyDist = 999999 # min distance to the nearest ghost
    for enemy, pos, isPacman, scaredTimer in self.enemyInfo:
      if not isPacman and scaredTimer < 2:
        enemyDist = min(enemyDist, self.getMazeDistance(mypos, pos))

    #print "heuristic() time",time.time()-t1
    if mode == 'goHome': # directly home
      if enemyDist == 0:
        return 999 #dead
      else:
        return goalDist + 10.0/((enemyDist-1)*2+0.1)
    else: # retreat: going back (escape goals or capsule) tending to eat food along the way
      capsules = self.getCapsules(gameState)
      if len(capsules) > 0:
        goalDist = min(goalDist, [self.getMazeDistance(mypos, p) for p in capsules])
      # pickupfood = 0
      # if mypos in self.foodList:
      #   pickupfood = 1
      if enemyDist == 0:
        return 999 #dead
      else:
        return goalDist + 10.0/((enemyDist-1)*2+0.1) #+ 1.0/(pickupfood+1.0)


  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    return successor is a gameState
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
    elif mode == "goHome":
      features = self.getGoHomeFeatures(gameState, action)
      weights = self.getGoHomeWeights(gameState, action)
      #print "features * weights",features * weights
    elif mode == "defend":
      features = self.getDefendFeatures(gameState, action)
      weights = self.getDefendWeights(gameState, action)
    elif mode == "retreat":
      features = self.getRetreatFeatures(gameState, action)
      weights = self.getRetreatWeights(gameState, action)


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
    enemy = self.getNearestGhost(myPos, gameState);
    enemyScaredTimer = gameState.getAgentState(enemy[0]).scaredTimer
    if enemyScaredTimer <= 5:
      features['distanceToEnemy'] = 100.0/(enemy[1]+0.1)
    # is it a dead corner? corner depth
    if(len(gameState.getLegalActions(self.index)) <= 2):
      features['deadCorner'] = 1
    else:
      features['deadCorner'] = 0
    # stop
    if action == Directions.STOP:
      features['stop'] = 1
    # move history

    return features

  def getAttackWeights(self, gameState, action):
    return {'successorScore': 30000, 'distanceToFood': -100, 'distanceToCapsule': -20, 'distanceToEscape': 5,
            'distanceToAlly': 50, 'distanceToEnemy': -5, 'stop': -1000, 'pickupCapsule': 1000, 'deadCorner': -10}

  def getDefendFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0
    features['dead']=1
    if myPos==successor.getInitialAgentPosition(self.index):features['dead']=0
    ghostIndex,ghostDis=self.getNearestGhost(myPos,successor)
    if myState.isPacman and ghostDis==1 and successor.getAgentState(ghostIndex).scaredTimer==0:
      features['dead']=0
    if successor.getAgentState(ghostIndex).scaredTimer>3:
      if myState.isPacman: features['onDefense'] = 2
    # Computes distance to invaders we can see
    #getOpponents index
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    #if enemy is a pacman and visible
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    #num of enemy who is a pacman and visible
    features['numInvaders'] = len(invaders)
    #print "--------------",features['numInvaders']
    if len(invaders) > 0:
      #distance to each enemy
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      #distance to nearest enemy
      features['invaderDistance'] = min(dists)
    else:
      features['invaderDistance'] = self.getNearestPacman(myPos, successor)[1]
    #if i choose "stop" as action
    if action == Directions.STOP: features['stop'] = 1
    #get the reverse direction of current moving
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getDefendWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'numInvaders': -99999999, 'onDefense': 100, 'invaderDistance': -100, 'stop': -200, 'reverse': -2,'dead':99999}

  def getGoHomeFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()


    ## Computes whether we're safe
    features['isSafe'] = 1
    if myState.isPacman: features['isSafe'] = 0

    successor = self.getSuccessor(gameState, action)
    #gameStae after take action
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    #distance from myPos to the nearest middle line
    distanceFromStart = min([self.distancer.getDistance(myPos, (self.midWidth, i))
                             for i in range(gameState.data.layout.height)
                             if (self.midWidth, i) in self.legalPositions])
    #disctance from i to nearest "ghost" of enemy
    enemyIndex,minDis=self.getNearestGhost(myPos,successor)
    #print enemyIndex,minDis
    if(gameState.getAgentState(enemyIndex).scaredTimer>0):
      features["mindDis"]=0
      features["critical"]=0
    else:
      features["mindDis"]=minDis
      features['distanceFromStart']=distanceFromStart
      if minDis>2:
        features["critical"]=0
      else:
        features["critical"]=1


    return features

  def getGoHomeWeights(self, gameState, action):
    return {'isSafe':999,'minDis': 1, 'distanceFromStart': -2,'critical':-999}

  def getRetreatFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    ## Computes whether we're safe
    features['isSafe'] = 1
    if myState.isPacman: features['isSafe'] = 0

    ## Computes distance to ghosts we can see
    #enemies = [successor.getAgentState(i) for i in self.enemies]
    #chasers = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
    #features['numChasers'] = len(chasers)

    #if len(chasers) > 0:
    #  dists = [self.getMazeDistance(myPos, a.getPosition()) for a in chasers]
    #  features['chaserDistance'] = min(dists)
    #if action == Directions.STOP: features['stop'] = 1
    #rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

    #if action == rev: features['reverse'] = 1

    successor = self.getSuccessor(gameState, action)
    #gameState after take this action
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    #distacne from myPos to nearest Middle Line
    distanceFromStart = min([self.getMazeDistance(myPos, pos) for pos in self.escapeGoals])
    #if "Capsules" >0
    if len(self.getCapsules(gameState))>0:
      distanceFromCapsuls=min([self.getMazeDistance(myPos, p) for p in self.getCapsules(gameState)])
      #if "Capsules" is nearer than middle line, go to eat it
      distanceFromStart=min(distanceFromStart,distanceFromCapsuls)
    #distance from i to nearest "ghost" of enemy
    enemyIndex,minDis=self.getNearestGhost(myPos,successor)
    features['distanceFromStart']=distanceFromStart
    if(gameState.getAgentState(enemyIndex).scaredTimer>0):
      features["mindDis"]=0
      features["critical"]=0
    else:
      features["mindDis"]=minDis
      if minDis>2:
        features["critical"]=0
      else:
        features["critical"]=1

    #战略撤退，查看到最近food的距离，越小越好
    features["foodDist"] = min([self.getMazeDistance(myPos, food)for food in self.foodList])
    #战略撤退，下一步吃豆，给很高的权重去吃豆
    successorFoodList = self.getFood(successor).asList()
    features["pickupFood"] = len(self.foodList) - len(successorFoodList)
    #print "action",action
    #print features
    return features

  def getRetreatWeights(self, gameState, action):
    return {'isSafe':500,'minDis': 1, 'distanceFromStart': -2,'critical':-1000, 'pickupFood': 200}


class TopAgent(GeneralAgent):

  def registerInitialState(self, gameState):
    GeneralAgent.registerInitialState(self, gameState)
    self.favoredY = gameState.data.layout.height

# Leeroy Bottom Agent - favors pellets with a lower y
class BottomAgent(GeneralAgent):

  def registerInitialState(self, gameState):
    GeneralAgent.registerInitialState(self, gameState)
    self.favoredY = 0