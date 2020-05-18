from inspect import signature, Parameter

class State:
    def __init__(self, value = None):
        self.__value = value
        
    def __call__(self, *args):
        if len(args) == 1:
            self.__value = args[0]
        elif len(args) > 1:
            self.__value = args
        return self.__value
    
class StateMap:
    def __init__(self, *methods):
        self.__states = {}
        self.__methods = {}
        self.initFromSignature(*methods)
        
    def initFromSignature(self, *methods):
        for method in methods:
            params = signature(method).parameters
            for key, param in params.items():
                if key not in self.__states:
                    if param.default != Parameter.empty:
                        self[key] = param.default
        
    def mapSignature(self, method):
        if method not in self.__methods:
            params = signature(method).parameters
            args = [self[key] for key in params]
            self.__methods[method] = args
        return self.__methods[method]
    
    def invoke(self, method, *args, **kwargs):
        args = list(args)
        kwarguments = {}
        params = signature(method).parameters
        for key, param in params.items():
            if key in kwargs:
                self[key] = kwargs[key]
            else:
                if len(args):
                    self[key] = args.pop(0)
                elif param.default != Parameter.empty:
                    self[key] = param.default
            kwarguments[key] = self[key]
        return method(**kwarguments)
    
    def __getitem__(self, key):
        if key not in self.__states:
            self.__states[key] = State()
        return self.__states[key]
    
    def __setitem__(self, key, value):
        if key not in self.__states:
            self.__methods.clear()
            self.__states[key] = State()
        self.__states[key](value)       
        
# A standard reinforcement learning agent
class RlAgent:
    
    def __init__(self, rl):
        self.__rl = rl
        self.__terminated = False
        self.__stepCount = 0
        self.__epoch = 0
        self.__learnEnabled = False
        self.__stateMap = StateMap(
            self.onBegin, self.onStep, self.onFinish
        )
    
    def predict(self, state, actions, epsilon = None):
        if not self.__learnEnabled:
            epsilon = 0
        elif epsilon is None:
            epsilon = self.__stateMap["epsilon"]()
        return self.__rl.predict(state, actions, epsilon=epsilon, learn=self.__learnEnabled)
    
    def reward(self, value, absorb = True):
        if not self.__learnEnabled:
            return False
        self.__rl.reward(value, absorb=absorb)
    
    def train(self, *args, epsilon = 0, simLimit = None, **kwargs):
        self.__learnEnabled = True
        self.__stateMap["epsilon"] = epsilon
        result = self.run(epsilon, simLimit, *args, **kwargs)
        self.__learnEnabled = False
        return result
    
    def test(self, *args, simLimit = None, **kwargs):
        result = self.run(0, simLimit, *args, **kwargs)
        return result
    
    def run(self, epsilon, simLimit, *args, **kwargs):
        self.newEpisode()
        self.__stateMap.invoke(self.onBegin, *args, **kwargs)
        goalArgs = self.__stateMap.mapSignature(self.isAtGoal)
        stepArgs = self.__stateMap.mapSignature(self.onStep)
        while not self.__terminated and not self.isAtGoal(*goalArgs):
            if self.canContinue(simLimit):
                self.onStep(*stepArgs)
                self.__stepCount += 1
            else:
                self.terminate()
        args = self.__stateMap.mapSignature(self.onFinish)
        return self.onFinish(*args)
    
    def terminate(self):
        self.__terminated = True
        args = self.__stateMap.mapSignature(self.onTerminate)
        self.onTerminate(*args)
    
    def canContinue(self, simLimit):
        return not simLimit or self.__stepCount < simLimit
    
    def newEpisode(self):
        self.__rl.newEpisode()
        self.__terminated = False
        self.__stepCount = 0
        self.__epoch += 1
        
    def epoch(self):
        return self.__epoch
    
    def rl(self):
        return self.__rl
    
    def stepCount(self):
        return self.__stepCount
    
    def wasTerminated(self):
        return self.__terminated
    
    # Methods to override
    
    def isAtGoal(self):
        """Indicate if the agent has made it to the goal state"""
        return False
    
    def onBegin(self):
        """Indicates the beginning of an episode. Include any extra parameters"""
        pass
    
    def onStep(self):
        """Perform a step in the episode. Return True to finish"""
        return True
    
    def onTerminate(self):
        """Invoked upon early termination"""
    
    def onFinish(self):
        """Indicates the end of an episode"""
        pass