from hrr import hrri, convolve, LTM
import numpy as np
import random
import time
import math

from rl import *
from agent import RlAgent
from utils import display_progress, train, plot

# Calculate the optimal trace for the given goal
def optimalTraces(size, goal, discountFactor):
    trace = np.zeros(size)
    for i in range(1, size//2 + 1):
        trace[-i] = trace[i] = discountFactor*trace[i-1] - 1
    return np.array([np.roll(trace, goal + 1), np.roll(trace, goal - 1)])

class MazeAgent(RlAgent):

    def __init__(self, rl, mazeSize, goal):
        super(MazeAgent, self).__init__(rl)
        self.__size = mazeSize
        self.__goal = goal

    def onBegin(self, state, startState = None):
        if startState() is not None:
            state(startState)
        else:
            state(random.randrange(self.__size))

    def onStep(self, state):
        if self.stepCount() > 0:
            self.reward(-1.0, absorb=False)
        action = self.predict(f"s{state()}", ("left", "right"))
        if action == 0:
            state((state() - 1) % self.__size)
        else:
            state((state() + 1) % self.__size)

    def onFinish(self, state):
        if state() == self.__goal and self.stepCount() > 0:
            self.reward(0.0)
        return self.stepCount()

    def isAtGoal(self, state):
        return state() == self.__goal

    def error(self):
        states = [f"s{i}" for i in range(self.__size)]
        traces = self.rl().traces(states, ("left", "right"))
        oTraces = optimalTraces(self.__size, self.__goal, self.rl().discountFactor())
        return np.mean(np.delete(np.square(traces - oTraces), self.__goal, axis=1))

    def plot(self, title):
        l, r = ([self.rl().maxValue(f"s{i}", a) for i in range(self.__size)] for a in ("left", "right"))
        plot(title, MAZE_SIZE, (l, r), ("Left", "Right"))

STEP_LIMIT = 100

# Maze Settings
MAZE_SIZE = 20
GOAL      = 10

# Agent Settings
HRR_SIZE   = 256 # 64
LEARN_RATE = 0.05
EPSILON    = 0.2
DISCOUNT   = 0.95
COPY       = 10

# The agents (Q-learning, double Q-learning, deep Q-learning, double/deep Q-learning)
standardAgents = []
for hidden in ([], [HRR_SIZE//2]):
    for nQlearning in (1, 2):
        ql = NQLearningNetwork(nQlearning, HRR_SIZE, hiddenLayers=hidden, learnRate=LEARN_RATE, discountFactor=DISCOUNT, eligibilityFactor = 0.1, copyFrequency=COPY)
        standardAgents.append(MazeAgent(ql, MAZE_SIZE, GOAL))

train("Q-learning Maze Agent", standardAgents[0], maxEpochs=10000, epsilon=EPSILON, simLimit=STEP_LIMIT, useBest = False)
