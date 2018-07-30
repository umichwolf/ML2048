import tensorflow as tf
import pickle
import copy
import numpy as np
from board import Board

class Ai:
    """
    This class implement the ai player consisting of value networks,
    policy networks and a search strategy.
    It can play 2048 games and also train itself after several games.
    """
    def __init__(self,path='./tmp/',game_path='./game_archives/'):
        self._keep_prob = 0.7
        self._path = path
        self._game_path = game_path
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
        tf.reset_default_graph()
        self._value_graph = tf.Graph()
        tf.reset_default_graph()
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

    def __del__(self):
        self._policy_sess.close()
        self._value_sess.close()

    def get_params(self):
        return {
            'name': self._name,
            'para': self._para,
            # 'keep_prob': self._keep_prob,
            'search_depth': self._search_depth,
            'intuition_depth': self._intuition_depth}

    def _build(self,name,para,search_depth,intuition_depth):
        self._name = name
        self._para = para
        self._search_depth = search_depth
        self._intuition_depth = intuition_depth
        self._virtual_board = Board(para)
        # try:
        self._value_model()
        self._policy_model()
        # except:
         #   print('Model Initialization Fails!')
        # else:
         #   print('Model Generated!')

    def new(self,name,para):
        ckpt_exists = tf.train.checkpoint_exists(self._path + name + '_value')
        if ckpt_exists:
            print('''The Name Has Been Used! Do you want to
            1. load it
            2. change a name
            3. overwrite it ''')
            choice = input('Please select: ')
            if choice == '1':
                self.load(name)
                return 1
            if choice == '2':
                self._new(input('New name: '),search_depth,intuition_depth)
        search_depth = eval(input('Search Depth: '))
        intuition_depth = eval(input('Intuition Depth: '))
        self._build(name,para,search_depth,intuition_depth)

    def load(self,name):
        with open(self._path + name + '_value.pkl','rb') as f:
            params = pickle.load(f)
        self._build(**params)
        try:
            with self._policy_graph.as_default():
                saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(self._policy_sess,self._path + name + '_policy')
            with self._value_graph.as_default():
                saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(self._value_sess,self._path + name + '_value')
        except:
            print('Restoring Variables Fails!')

    def save(self):
        name = self._name
        with self._policy_graph.as_default():
            saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.save(self._policy_sess,self._path + name + '_policy')
        with self._value_graph.as_default():
            saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.save(self._value_sess,self._path + name + '_value')
        with open(self._path + name + '_value.pkl','wb') as f:
            pickle.dump(self.get_params(),f)

    def _base_model(self,x,keep_prob,normalizer):
        size = self._para['size']
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
        dropout1 = tf.scalar_mul(normalizer,tf.nn.dropout(x=flat,
            keep_prob=keep_prob))
        dense1 = tf.layers.dense(inputs=dropout1,
            units=20,
            activation = tf.nn.relu,
            name='dense1'
            )
        dropout2 = tf.nn.dropout(x=dense1,
            keep_prob=keep_prob) * normalizer
        return dropout2

    def _value_model(self):
        size = self._para['size']
        with self._value_graph.as_default():
            global_step = tf.Variable(0,trainable=False,name='global')
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,size,size,1],name='features')
            y = tf.placeholder(dtype=tf.float32,
                shape=[None],name='scores')
            keep_prob = tf.placeholder(shape=[],
                dtype=tf.float32,name='keep_prob')
            normalizer = tf.placeholder(shape=[],dtype=tf.float32,
                name='normalizer')
            dense2 = tf.layers.dense(
                inputs=self._base_model(x,keep_prob,normalizer),
                units=1,
                activation = tf.nn.relu)
            loss = tf.losses.mean_squared_error(labels=tf.reshape(y,[-1,1]),
                predictions=dense2)
            optimizer = tf.train.AdamOptimizer(learning_rate=1)
            train_op = optimizer.minimize(loss=loss,
                global_step=global_step)
            tf.add_to_collection('output',dense2)
            tf.add_to_collection('output',train_op)
            tf.add_to_collection('output',loss)
            init_op = tf.global_variables_initializer()
        self._value_sess.run(init_op)

    def _policy_model(self):
        size = self._para['size']
        with self._policy_graph.as_default():
            global_step = tf.Variable(0,trainable=False,name='global')
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,size,size,1],name='features')
            y = tf.placeholder(dtype=tf.uint8,
                shape=[None],name='labels')
            keep_prob = tf.placeholder(shape=[],
                dtype=tf.float32,name='keep_prob')
            normalizer = tf.placeholder(shape=[],dtype=tf.float32,
                name='normalizer')
            dense2 = tf.layers.dense(
                inputs=self._base_model(x,keep_prob,normalizer),
                units=4
                )
            onehot_labels = tf.one_hot(indices=y,depth=4)
            loss = tf.reduce_mean(
                tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                    logits=dense2))
            optimizer = tf.train.AdamOptimizer(learning_rate=1)
            train_op = optimizer.minimize(loss=loss,
                global_step=global_step)
            tf.add_to_collection('output',dense2)
            tf.add_to_collection('output',train_op)
            tf.add_to_collection('output',loss)
            init_op = tf.global_variables_initializer()
        self._policy_sess.run(init_op)

    def _predict_policy(self,board):
        feed_dict = {'features:0':board,
            'keep_prob:0': 1,
            'normalizer:0': self._keep_prob}
        prediction,_,_ = self._policy_graph.get_collection('output')
        result = self._policy_sess.run(prediction,feed_dict=feed_dict)
        order = np.argsort(result)[::-1]
        move_sequence = [self._move_list[order[0,idx]]
            for idx in range(len(order[0]))]
        return move_sequence

    def _predict_value(self,board):
        feed_dict = {'features:0':board,
            'keep_prob:0': 1,
            'normalizer:0': self._keep_prob}
        prediction,_,_ = self._value_graph.get_collection('output')
        result = self._value_sess.run(prediction,feed_dict=feed_dict)
        return result[0,0]

    def _fit_value_net(self,data,scores,batch_size,n_iter):
        with self._value_graph.as_default():
            _,train_op,loss = self._value_graph.get_collection('output')
        for idx in range(n_iter):
            rand_list = np.random.randint(len(scores),size=batch_size)
            feed_dict = {'features:0':data[rand_list,:],
                         'scores:0':scores[rand_list],
                         'keep_prob:0': self._keep_prob,
                         'normalizer:0': 1.}
            if idx % 100 == 1:
                print('iter: {0:d}, loss: {1:.4f}'.format(
                    idx, self._value_sess.run(loss,feed_dict)))
            self._value_sess.run(train_op,feed_dict)

    def _fit_policy_net(self,data,labels,batch_size,n_iter):
        indices = [self._move_list.index(label) for label in labels]
        indices = np.array(indices)
        with self._policy_graph.as_default():
            _,train_op,loss = self._policy_graph.get_collection('output')
        for idx in range(n_iter):
            rand_list = np.random.randint(len(labels),size=batch_size)
            feed_dict = {'features:0':data[rand_list,:],
                         'labels:0':indices[rand_list],
                         'keep_prob:0': self._keep_prob,
                         'normalizer:0': 1}
            if idx % 100 == 1:
                print('iter: {0:d}, loss: {1:.4f}'.format(
                    idx, self._policy_sess.run(loss,feed_dict)))
            self._policy_sess.run(train_op,feed_dict)

    def _load_game(self,filename):
        gamedata = []
        with open(self._game_path+filename,'r') as f:
            for line in f:
                gamedata.append(eval(line))
        with open(self._game_path+filename+'.para','r') as f:
            para = eval(f.read())
        return gamedata,para

    def _game_type_is(self,para):
        return True if self._para == para else False

    def _convert_board(self,board):
        size = self._para['size']
        data = np.reshape(board,(-1,size,size,1))
        return data

    def _log_board(self,board):
        data = np.array(board,dtype=np.float32)
        data = np.ma.log2(data).filled(0)
        return data

    def learn(self,batch_size,filenames):
        p_labels = []
        v_scores = []
        data = []
        for filename in filenames:
            print(filename)
            gamedata,para = self._load_game(filename)
            if not self._game_type_is(para):
                print(filename+' Data Type Not Match!')
                continue
            p_labels.extend([gamedata[idx][-1] for idx in range(len(gamedata))])
            v_scores.extend([gamedata[idx+self._intuition_depth][:-1].count(0)
                for idx in range(len(gamedata)-self._intuition_depth)])
            data.extend([gamedata[idx][:-1] for idx in range(len(gamedata))])
        n_iter = len(data)
        data = self._log_board(data)
        data = self._convert_board(data)
        v_scores = np.array(v_scores)
        self._fit_policy_net(data,p_labels,batch_size,n_iter)
        self._fit_value_net(data,
            v_scores,batch_size,n_iter)

    def predict_value(self,board):
        board = self._log_board(board)
        board = self._convert_board(board)
        return self._predict_value(board)

    def predict_policy(self,board):
        data = self._log_board(board)
        data = self._convert_board(data)
        move_list = []
        for move in self._predict_policy(data):
            self._virtual_board.load_board(board)
            tag = self._virtual_board.move(move,quiet=1)
            if tag == 1:
                move_list.append(move)
        return move_list

    def move(self,board):
        self._best_move = None
        self._best_value = -1
        self.search(board,self._search_depth)
        print(self._best_move)
        return self._best_move

    def search(self,board,depth,current_value=0):
        move_list = self.predict_policy(board)[:2]
        if move_list == [] or depth == 0:
            print(self._current_move,current_value)
            if current_value > self._best_value:
                self._best_move = self._current_move
                self._best_value = current_value
            return 1
        for move in move_list:
            if depth == self._search_depth:
                self._current_move = move
            self._virtual_board.load_board(board)
            self._virtual_board.move(move)
            next_value = current_value + self.predict_value(self._virtual_board)
            self._virtual_board.next()
            board_tmp = copy.deepcopy(self._virtual_board)
            self.search(board_tmp,depth-1,next_value)

def main():
    end_flag = 0
    while end_flag == 0:
        order = input('''
Choose the option from the list:
    1. Build a new ai player
    2. Load an ai player
    3. Train the ai player
    4. Exit
    ''')
        if order == '1':
            name = input('name: ')
            size = eval(input('size: '))
            ratio = eval(input('odd of 2(between 0 and 1): '))
            para = {'size': size, 'odd_2': ratio}
            ai_player = Ai()
            ai_player.new(name,para)
        if order == '2':
            ai_player = Ai()
            ai_player.load(input('name: '))
        if order == '3':
            filenames = input('game name: ')
            filenames = filenames.split()
            batch_size = eval(input('epochs: '))
            ai_player.learn(batch_size,filenames)
            save_order = input('Do you want to save it? (y/n) ')
            if save_order == 'y':
                ai_player.save()
        if order == '4':
            end_flag = 1

if __name__ == '__main__':
    main()
