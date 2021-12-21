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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currentPosition = currentGameState.getPacmanPosition()
        
        currentFood = currentGameState.getFood()
        
        maze = currentGameState.getWalls()
        
        lengthMax = maze.height - 2 + maze.width - 2
        
        points = 0
        
        if currentFood[newPos[0]][newPos[1]]:
            points += 10
            
        distanceNextFood = float("inf")
        
        for food in newFood.asList():
            distanceFood = manhattanDistance(newPos, food)
            distanceNextFood = min([distanceNextFood, distanceFood])
            
        distanceNextGhost = float("inf")
        
        for ghost in successorGameState.getGhostPositions():
            distanceGhost = manhattanDistance(newPos, ghost)
            distanceNextGhost = min([distanceNextGhost, distanceGhost])
            
        if distanceNextGhost < 2:
            points -= 500
            
        points = points + 1.0 / distanceNextFood + distanceNextGhost/lengthMax
        
        return points
            
        

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
        def maxi(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            value = float("-inf")
            
            for action in state.getLegalActions():
                value = max(value, mini(state.generateSuccessor(0, action), depth, 1))
            
            return value
        
        def mini(state, depth, index):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            value = float("inf")
            
            if index == state.getNumAgents()-1:
                for action in state.getLegalActions(index):
                    value = min(value, maxi(state.generateSuccessor(index, action), depth + 1))
            else:
                for action in state.getLegalActions(index):
                    value = min(value, mini(state.generateSuccessor(index, action), depth, index + 1))
                    
            return value
        
        value = float("-inf")
        
        act = Directions.STOP
        
        for action in gameState.getLegalActions():
            tempRes = mini(gameState.generateSuccessor(0, action), 0, 1)
            
            if tempRes > value:
                value = tempRes
                act = action
                
        return act
            
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBeta(0, 0, gameState, float("-inf"), float("inf"))[0]
        util.raiseNotDefined()
        
    def alphaBeta(self, depth, index, gameState, alpha, beta):
        if index >= gameState.getNumAgents():
                depth += 1
                index = 0
                
        if depth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        bestScore, bestAction = None, None
        
        if index == 0: 
            for action in gameState.getLegalActions(index): 
               
                nextGameState = gameState.generateSuccessor(index, action)
                score = self.alphaBeta(depth, index + 1, nextGameState, alpha, beta)[1]
                
                if bestScore is None or score > bestScore:
                    bestScore = score
                    bestAction = action
                
                alpha = max(alpha, score)
                
                if alpha > beta:
                    break
        else:  
            for action in gameState.getLegalActions(index):
                nextGameState = gameState.generateSuccessor(index, action)
                score = self.alphaBeta(depth, index + 1, nextGameState, alpha, beta)[1]
                
                if bestScore is None or score < bestScore:
                    bestScore = score
                    bestAction = action
                
                beta = min(beta, score)
               
                if beta < alpha:
                    break
        
        if bestScore is None:
            return None, self.evaluationFunction(gameState)
        return bestAction, bestScore

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
        def maxi(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
        
            value = float("-inf")
        
            for action in state.getLegalActions():
                value = max(value, expecti(state.generateSuccessor(0, action), depth, 1))
        
            return value
    
        def expecti(state, depth, index):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            value = 0

            if index == state.getNumAgents()-1:
                for action in state.getLegalActions(index):
                    value += maxi(state.generateSuccessor(index, action), depth + 1)
            else:
                for action in state.getLegalActions(index):
                    value += expecti(state.generateSuccessor(index, action), depth, index + 1)

            return value / len(state.getLegalActions(index))


        act = Directions.STOP
        value = float("-inf")

        for action in gameState.getLegalActions():
            tempRes = expecti(gameState.generateSuccessor(0, action), 0, 1)

            if tempRes > value:
                value = tempRes
                act = action

        return act
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    maze = currentGameState.getWalls()
    posPowerUp = currentGameState.getCapsules()
    x,y = currentGameState.getPacmanPosition()
    posFood = currentGameState.getFood()
    
    lengthMax = maze.height - 2 + maze.width - 2
    
    distanceToFood = []
    distanceToPowerUp = []
    
    for food in posFood.asList():
        distanceToFood.append(manhattanDistance((x,y), food))
    
    for powerUp in posPowerUp:
        distanceToPowerUp.append(manhattanDistance((x,y), powerUp))
    
    points = 0
    
    for ghost in currentGameState.getGhostStates():
        distanceToGhost = manhattanDistance((x,y), ghost.configuration.getPosition())
        
        if distanceToGhost < 2:
            if ghost.scaredTimer != 0:
                points += 1000.0 / (distanceToGhost + 1)
            else:
                points -= 1000.0 / (distanceToGhost + 1)
                
    if min(distanceToPowerUp + [100.0]) < 5:
        points += 500.0 / min(distanceToPowerUp)
    
    for powerUpX, powerUpY in posPowerUp:
        if(powerUpX == x) & (powerUpY == y):
            points += 600.0
            
    return points + 1.0 / min(distanceToFood + [100.0]) - len(distanceToFood) * 10.0

    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
