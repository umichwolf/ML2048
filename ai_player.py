# import tensorflow as tf
import copy
import numpy as np
from board import Board

class Ai:
    """
    This class implement the ai player consisting of value networks,
    policy networks and a search strategy.
    It can play 2048 games and also train itself after several games.
    """
    def __init__(self,name,para,search_depth=3):
        self._name = name
        self._para = para
        self._virtual_board = Board(para)
        self._search_depth = search_depth
        self._best_value = 0
        self._best_move = None
        self._current_value = 0
        self._current_move = None

    @property
    def name(self):
        return self._name
    @property
    def para(self):
        return self._para

    def value_net(self,board):
        return np.sum(board)

    def policy_net(self,board):
        move_list = []
        for move in ['a','w','s','d']:
            self._virtual_board.load_board(board)
            if self._virtual_board.move(move,quiet=1):
                move_list.append(move)
        return move_list

    def move(self,board):
        self._best_move = None
        self.search(board,self._search_depth)
        return self._best_move

    def search(self,board,depth,current_value=0):
        if depth  == 0:
            if current_value > self._best_value:
                self._best_move = self._current_move
                self._best_value = current_value
            return 1
        for move in self.policy_net(board)[:2]:
            if depth == self._search_depth:
                self._current_move = move
            self._virtual_board.load_board(board)
            self._virtual_board.move(move,quiet=1)
            next_value = current_value + self.value_net(self._virtual_board)
            print(move,next_value)
            self._virtual_board.next()
            board_tmp = copy.deepcopy(self._virtual_board)
            self.search(board_tmp,depth-1,next_value)
