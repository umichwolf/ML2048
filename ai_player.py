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
    def __init__(self,path='./tmp/'):
        self._path = path
        self._name = None
        self._para = None
        self._move_list = ['a','w','s','d']
        self._search_depth = None
        self._intuition_depth = None
        self._best_value = -1
        self._best_move = None
        self._current_value = 0
        self._current_move = None
        self._policy_graph = tf.Graph()
        self._value_graph = tf.Graph()
        self._policy_sess = tf.Session(graph=self._policy_graph)
        self._value_sess = tf.Session(graph=self._value_graph)

    @property
    def name(self):
        return self._name
    @property
    def para(self):
        return self._para
    @property
    def search_depth(self):
        return self._search_depth
    @property
    def intuition_depth(self):
        return self._intuition_depth

    def get_params(self):
        return {
            'name': self._name,
            'para': self._para,
            'search_depth' = self._search_depth,
            'intuition_depth' = self._intuition_depth}

    def _build(self,name,para,search_depth,intuition_depth):
        self._name = name
        self._para = para
        self._search_depth = search_depth
        self._intuition_depth = intuition_depth
        self._virtual_board = Board(para)
        try:
            self._value_model()
            self._policy_model()
        except:
            print('Model Initialization Fails!')
        else:
            print('Model Generated!')

    def new(self,name,para,search_depth,intuition_depth):
        ckpt_exists = tf.train.checkpoint_exists(self._path + name + '_value')
        if ckpt_exists:
            print('''The Name Has Been Used! Do you want to
            1. load it
            2. change a name
            3. overwrite it ''')
            choice = input('Please select: ') == '1':
            if choice = '1':
                self._load(name)
            if choice = '2':
                self._new(input('New name: '),search_depth,intuition_depth)
        self._build(name,para,search_depth,intuition_depth)

    def load(self,name):
        params = np.load(self._path + name + '_value.npy')
        self._build(**params)
        try:
            saver = tf.train.Saver()
            saver.restore(self._value_sess,self._path + name + '_value.ckpt')
            saver.restore(self._policy_sess,self._path + name + '_policy.ckpt')
        except:
            print('Restoring Variables Fails!')

    def save(self):
        saver = tf.train.Saver()
        saver.save(self._value_sess,self._path + name + '_value.ckpt')
        saver.save(self._policy_sess,self._path + name + '_policy.ckpt')
        np.save(self.get_params(),self._path + self._name + '_value.npy')

    def _base_model(self,x):
        size = self._para['size']
        global_step = tf.Variable(0,trainable=False,name='global')
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
        flat = tf.reshape(conv2,shape=[-1,size*size*10])
        dropout1 = tf.nn.dropout(x=flat,
            keep_prob=0.7)
        dense1 = tf.layers.dense(inputs=dropout1,
            units=20,
            activation = tf.nn.relu,
            name='dense1'
            )
        dropout2 = tf.nn.dropout(x=dense1,
            keep_prob=0.7)
        return dropout2

    def _value_model(self):
        size = self._para['size']
        with self._value_graph.as_default():
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,size,size,1],name='features')
            y = tf.placeholder(dtype=tf.float32,
                shape=[None,1],name='scores')
            dense2 = tf.layers.dense(inputs=self._base_model(x),
                units=1,
                activation = tf.nn.relu,
                name='dense2'
                )
            loss = tf.losses.mean_squared_error(labels=y,
                predictions=dense2
                )
            optimizer = tf.train.AdamOptimizer(learning_rate=10)
            train_op = optimizer.minimize(loss=loss,
                global_step=global_step,
                name='Train')
            self._sess.run(tf.global_variables_initializer())

    def _policy_model(self):
        size = self._para['size']
        with self._policy_graph.as_default():
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,size,size,1],name='features')
            y = tf.placeholder(dtype=tf.float32,
                shape=[None,1],name='labels')
            dense2 = tf.layers.dense(inputs=self._base_model(x),
                units=4,
                activation = tf.sigmoid,
                name='dense2'
                )
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                logits=dense2
                )
            optimizer = tf.train.AdamOptimizer(learning_rate=10)
            self._sess.run(tf.global_variables_initializer())
            train_op = optimizer.minimize(loss=loss,
                global_step=global_step,
                name='Train')
            self._sess.run(tf.global_variables_initializer())

    def _predict_policy(self,board):
        feed_dict = {'features:0':board}
        prediction = self._policy_model.get_tensor_by_name('dense2/Relu:0')
        result=self._sess.run(prediction,feed_dict=feed_dict,graph=self._policy_graph)
        order = np.argsort(result)
        move_sequence = [self._move_list[order[idx]] for idx in len(order)]
        return move_sequence

    def _predict_value(self,board):
        feed_dict = {'features:0':board}
        prediction = self._value_model.get_tensor_by_name('dense2/Relu:0')
        result=self._sess.run(prediction,feed_dict=feed_dict,graph=self._value_graph)
        return result

    def _fit_value_net(self,data,scores,batch_size,n_iter):
        with self._policy_graph.as_default():
            train_op = self._value_graph.get_operation_by_name('Train')
            for idx in range(n_iter):
                rand_list = np.random.randint(len(data),size=batch_size)
                feed_dict = {'features:0':data[rand_list,:],
                             'scores:0':scores[rand_list]}
                if idx % 1000 == 1:
                    print('iter: {0:d}, loss: {1:.4f}'.format(
                        idx, self._sess.run(loss,feed_dict)))
                self._sess.run(train_op,feed_dict)

    def _fit_policy_net(self,data,labels,batch_size,n_iter):
        indices = [self._move_list.index(label) for label in labels]
        indices = np.array(indices)
        with self._policy_graph.as_default():
            train_op = self._value_graph.get_operation_by_name('Train')
            for idx in range(n_iter):
                rand_list = np.random.randint(len(data),size=batch_size)
                feed_dict = {'features:0':data[rand_list,:],
                             'labels:0':indices[rand_list]}
                if idx % 1000 == 1:
                    print('iter: {0:d}, loss: {1:.4f}'.format(
                        idx, self._sess.run(loss,feed_dict)))
                self._sess.run(train_op,feed_dict)

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
