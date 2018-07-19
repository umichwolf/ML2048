import tensorflow as tf
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
        self._best_value = -1
        self._best_move = None
        self._current_value = 0
        self._current_move = None
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        if self._model() == 0:
            raise ValueError

    @property
    def name(self):
        return self._name
    @property
    def para(self):
        return self._para

    def _model(self):
        size = self._para['size']
        with self._graph.as_default():
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,size,size,1],name='features')
            y = tf.placeholder(dtype=tf.float32,
                shape=[None,1],name='labels')
            W_conv1 = tf.Variable(tf.truncated_normal(shape=[2,2,1,5],
                stddev=1e-2))
            b_conv1 = tf.Variable(tf.constant(0.1,shape=[5]))
            conv1 = tf.nn.relu(tf.nn.conv2d(input=x,
                filter=W_conv1,
                strides=[1,1,1,1],
                padding='SAME',
                name='conv1'
                ) + b_conv1)
            W_conv2 = tf.Variable(tf.truncated_normal(shape=[2,2,5,10],
                stddev=1e-2))
            b_conv2 = tf.Variable(tf.constant(0.1,shape=[10]))
            conv2 = tf.nn.relu(tf.nn.conv2d(input=conv1,
                filter=W_conv2,
                strides=[1,1,1,1],
                padding='SAME',
                name='conv2'
                ) + b_conv2)
            flat = tf.reshape(conv2,shape=[-1,4*4*10])
            dropout1 = tf.nn.dropout(x=flat,
                keep_prob=0.7)
            dense1 = tf.layers.dense(inputs=dropout1,
                units=20,
                activation = tf.nn.relu,
                name='dense1'
                )
            dropout2 = tf.nn.dropout(x=dense1,
                keep_prob=0.7)
            dense2 = tf.layers.dense(inputs=dropout2,
                units=1,
                activation = tf.nn.relu,
                name='dense2'
                )
            print(dense2)
            loss = tf.losses.mean_squared_error(labels=y,
                predictions=dense2
                )
            self._sess.run(tf.global_variables_initializer())

    def _predict(self,board):
        feed_dict = {'features:0':board}
        prediction = self._graph.get_tensor_by_name('dense2/Relu:0')
        result=self._sess.run(prediction,feed_dict=feed_dict)
        return result

    def _fit(self,board):
        pass

    def value_net(self,board):
        board = np.array(board,dtype=np.float32)
        board = np.reshape(board,(1,4,4,1))
        return self._predict(board)

    def policy_net(self,board):
        move_list = []
        for move in ['a','w','s','d']:
            self._virtual_board.load_board(board)
            if self._virtual_board.move(move,quiet=1):
                move_list.append(move)
        return move_list

    def move(self,board):
        self._best_move = None
        self._best_value = -1
        self.search(board,self._search_depth)
        print(self._best_move)
        return self._best_move

    def search(self,board,depth,current_value=0):
        if depth  == 0:
            print(self._current_move,current_value)
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
            self._virtual_board.next()
            board_tmp = copy.deepcopy(self._virtual_board)
            self.search(board_tmp,depth-1,next_value)
