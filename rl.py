from inspect import signature, Parameter
import numpy as np
import keras
import random

from hrr import hrri, convolve, LTM
        
class HolographicNeuralNetwork:
    
    def __init__(self, *layers, verbose = False, separateTarget = False):
        # Long-term memory
        self.__ltm = LTM(layers[0])
        
        # Neural networks
        self.__modelTarget = self.__modelPredict = self.createModel(layers)
        if separateTarget:
            self.__modelTarget = self.createModel(layers)

        # Options
        self.__verbose = verbose
        
        # Optimizations
        self.__lastPrediction = None
        self.__backup = None
        
    def backup(self):
        self.__backup = self.__modelPredict.get_weights()
        
    def restore(self):
        self.__modelPredict.set_weights(self.__backup)
        
    def createModel(self, layers):
        model = keras.models.Sequential()
        for i in range(1, len(layers)):
            model.add(keras.layers.Dense(layers[i], input_shape=(layers[i-1],), activation="linear", use_bias=False))
        model.compile(loss=keras.losses.mean_squared_error, optimizer="adam")
        return model
    
    def copyTargetWeights(self):
        weights = self.__modelPredict.get_weights()
        self.__modelTarget.set_weights(weights)
        
    def predict(self, state, actions):
        inputs = np.array([self.encode(f"{state}*{a}") for a in actions])
        values = self.__modelPredict.predict(inputs, verbose = self.__verbose)
        self.__lastPrediction = (inputs, values)
        return values
    
    def targetValues(self, state, actions):
        inputs = np.array([self.encode(f"{state}*{a}") for a in actions])
        return self.__modelTarget.predict(inputs, verbose = self.__verbose)
    
    def value(self, state, action):
        inputs = np.array([self.encode(f"{state}*{action}")])
        return self.__modelPredict.predict(inputs, verbose = self.__verbose)[0]
    
    def averageValue(self, state, action):
        return np.average(self.value(state, action))
    
    def maxValue(self, state, action):
        return np.max(self.value(state, action))
    
    def minValue(self, state, action):
        return np.min(self.value(state, action))
    
    def fit(self, stateAction, target):
        self.__modelPredict.fit(np.array([stateAction]), np.array([target]), verbose = self.__verbose)
        
    def encode(self, value):
        return self.__ltm.encode(value)
        
    def decode(self, string):
        return self.__ltm.decode(value)
    
    def lastPrediction(self):
        return self.__lastPrediction
    
    def clearLastPrediction(self):
        self.__lastPrediction = None
        
    def traces(self, states, actions):
        return np.array([[self.averageValue(s, a) for s in states] for a in actions])        
        
class NQLearningNetwork(HolographicNeuralNetwork):
    
    def __init__(self, nQlearning, hrrSize, hiddenLayers = [], learnRate = 1, discountFactor = 0.5, eligibilityFactor = 0.0, copyFrequency = 10, verbose = False):
        super(NQLearningNetwork, self).__init__(hrrSize, *hiddenLayers, nQlearning, verbose=verbose)
        self.__isDeep = len(hiddenLayers) > 0
        self.__nQlearning = nQlearning
        self.__learnRate = learnRate
        self.__discountFactor = discountFactor
        self.__eligibilityFactor = eligibilityFactor
        self.__copyFrequency = copyFrequency
        self.__nUpdates = 0
        self.__defaultReward = 0
        self.__reward = self.__defaultReward
        self.__prevAction = None
        self.__eligibilityTrace = np.zeros(hrrSize)
        
    def policy(self, values, epsilon = 0.0):
        """Determine the action to take"""
        if epsilon > 0 and random.random() <= epsilon:
            return (True, random.randrange(len(values)))
        return (False, np.argmax(np.average(values, axis=1)))
    
    def update(self, state, prevPrediction, targetValues, eligibilityTrace, action):
        toUpdate = random.randrange(self.__nQlearning)
        maxAction = np.argmax(targetValues[:,toUpdate])
        if self.__nQlearning > 1:
            qValue = np.average(np.delete(targetValues[maxAction], toUpdate))
        else:
            qValue = targetValues[maxAction][0]
        target = prevPrediction[1][action]
        delta = self.__reward + self.__discountFactor*qValue - target[toUpdate]
        target[toUpdate] += self.__learnRate*delta
        self.fit(eligibilityTrace, target)
        self.__reward = self.__defaultReward
        self.__nUpdates += 1
        if self.__isDeep and self.__nUpdates % self.__copyFrequency == 0:
            self.copyTargetWeights()
        
    def predict(self, state, actions, epsilon = 0.0, learn = False):
        """Predict the action to take. If learning is enabled, updates previous prediction"""
        prevPrediction, prevAction = self.lastPrediction(), self.__prevAction
        values = super(NQLearningNetwork, self).predict(state, actions)
        isRandom, self.__prevAction = self.policy(values, epsilon)
        if learn and prevPrediction:
            self.__eligibilityTrace *= self.__eligibilityFactor
            self.__eligibilityTrace += prevPrediction[0][prevAction] # previous HRR
            targetValues = self.targetValues(state, actions) if self.__isDeep else values
            self.update(state, prevPrediction, targetValues, self.__eligibilityTrace, prevAction)
        if isRandom:
            self.__eligibilityTrace *= 0.0
        return self.__prevAction
    
    def reward(self, reward, absorb = True):
        """Reward the Q-learning agent"""
        if absorb:
            toUpdate = random.randrange(self.__nQlearning)
            target = self.lastPrediction()[1][self.__prevAction]
            target[toUpdate] = reward
            self.fit(self.lastPrediction()[0][self.__prevAction], target)
        else:
            self.__reward = reward
    
    def newEpisode(self):
        """Reset and prepare for the next episode"""
        self.clearLastPrediction()
        self.__reward = self.__defaultReward
        self.__prevAction = None
        self.__eligibilityTrace *= 0.0
        
    def discountFactor(self):
        return self.__discountFactor