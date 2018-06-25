import tensorflow as tf
import numpy as np

class Ai:
    """
    This class implement the ai player consisting of value networks,
    policy networks and a search strategy.
    It can play 2048 games and also train it self after several games.
    """
    def __init__(self,para,name):
        self._name = name
        self._para = para

    @property
    def name(self):
        return self._name
    @property
    def para(self):
        return self._para
        
    def value_net(self):
        pass

    def policy_net(self):
        pass

    def move(self,board):
        value_net()
        policy_net()
        search()

    def search(self,board):
