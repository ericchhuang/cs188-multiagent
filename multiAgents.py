# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        currentFood = currentGameState.getFood()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0
        for ghost in currentGameState.getGhostStates():
            ghostDistance = util.manhattanDistance(newPos, ghost.getPosition())
            if ghostDistance <= 1:
                score -= 1000000
                
        minFood = 500
        for food in newFood.asList():
            foodDistance = util.manhattanDistance(newPos, food)
            if foodDistance < minFood:
                minFood = foodDistance
        score -= 5*minFood
        
        if len(currentFood.asList()) > len(newFood.asList()):
            score += 500
        if len(newFood.asList()) == 0:
            score += 10000
        return score + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        legalMoves = gameState.getLegalActions(0)
        chosenIndex = 0
        maxValue = float('-inf')
        for i, action in enumerate(legalMoves):
            newValue = self.value(1, gameState.generateSuccessor(0, action), self.depth)
            if maxValue < newValue:
                chosenIndex = i
                maxValue = newValue
        return legalMoves[chosenIndex]
    
    def value(self, agentIndex, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(agentIndex, gameState, depth)
        if agentIndex != 0:
            return self.minValue(agentIndex, gameState, depth)
        
    def maxValue(self, agentIndex, gameState, depth):
        v = float('-inf')
        newIndex = (agentIndex+1)%gameState.getNumAgents()

        for action in gameState.getLegalActions(agentIndex):
            v = max(v, self.value(newIndex, gameState.generateSuccessor(agentIndex, action), depth))
        return v
    
    def minValue(self, agentIndex, gameState, depth):
        v = float('inf')
        newIndex = (agentIndex+1)%gameState.getNumAgents()
        newDepth = depth
        if newIndex == 0:
            newDepth = depth-1
            
        for action in gameState.getLegalActions(agentIndex):
            v = min(v, self.value(newIndex, gameState.generateSuccessor(agentIndex, action), newDepth))
        return v
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        chosenIndex = 0
        maxValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for i, action in enumerate(legalMoves):
            newValue = self.value(1, gameState.generateSuccessor(0, action), self.depth, alpha, beta)
            if maxValue < newValue:
                chosenIndex = i
                maxValue = newValue
            alpha = max(alpha, newValue)
        return legalMoves[chosenIndex]
    
    def value(self, agentIndex, gameState, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(agentIndex, gameState, depth, alpha, beta)
        if agentIndex != 0:
            return self.minValue(agentIndex, gameState, depth, alpha, beta)
        
    def maxValue(self, agentIndex, gameState, depth, alpha, beta):
        v = float('-inf')
        newIndex = (agentIndex+1)%gameState.getNumAgents()

        for action in gameState.getLegalActions(agentIndex):
            v = max(v, self.value(newIndex, gameState.generateSuccessor(agentIndex, action), depth, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def minValue(self, agentIndex, gameState, depth, alpha, beta):
        v = float('inf')
        newIndex = (agentIndex+1)%gameState.getNumAgents()
        newDepth = depth
        if newIndex == 0:
            newDepth = depth-1
            
        for action in gameState.getLegalActions(agentIndex):
            v = min(v, self.value(newIndex, gameState.generateSuccessor(agentIndex, action), newDepth, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        chosenIndex = 0
        maxValue = float('-inf')
        for i, action in enumerate(legalMoves):
            newValue = self.value(1, gameState.generateSuccessor(0, action), self.depth)
            if maxValue < newValue:
                chosenIndex = i
                maxValue = newValue
        return legalMoves[chosenIndex]
    
    def value(self, agentIndex, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(agentIndex, gameState, depth)
        if agentIndex != 0:
            return self.minValue(agentIndex, gameState, depth)
        
    def maxValue(self, agentIndex, gameState, depth):
        v = float('-inf')
        newIndex = (agentIndex+1)%gameState.getNumAgents()

        for action in gameState.getLegalActions(agentIndex):
            v = max(v, self.value(newIndex, gameState.generateSuccessor(agentIndex, action), depth))
        return v
    
    def minValue(self, agentIndex, gameState, depth):
        v = 0
        n = 0
        newIndex = (agentIndex+1)%gameState.getNumAgents()
        newDepth = depth
        if newIndex == 0:
            newDepth = depth-1
            
        for action in gameState.getLegalActions(agentIndex):
            v = v + self.value(newIndex, gameState.generateSuccessor(agentIndex, action), newDepth)
            n = n + 1
        return v/n

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: If ghost is within 1 tile, decrement score by large number. Otherwise, decrement score by distance to ghost. Decrement score by 25 * distance to closest food. Decrement score by 500 * number of pellets left. Decrement score by 5000 * number of capsules left. If a state has 0 pellets, increment by a large number. 
    """
    "*** YOUR CODE HERE ***"
    curFood = currentGameState.getFood()
    curPos = currentGameState.getPacmanPosition()
    curGhostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = 0
    for ghost in curGhostStates:
        ghostDistance = util.manhattanDistance(curPos, ghost.getPosition())
        score += ghostDistance
        if ghostDistance <= 1:
            score -= 1000000

    minFood = 5000
    for food in curFood.asList():
        foodDistance = util.manhattanDistance(curPos, food)
        if foodDistance < minFood:
            minFood = foodDistance
    score -= 25*minFood
    
    score -= len(curFood.asList())*500
    score -= len(capsules)*5000
    if len(curFood.asList()) == 0:
        score += 5000000
    return score
    
# Abbreviation
better = betterEvaluationFunction
