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
        self._search_width = None
        self._intuition_depth = None
        self._total_games = 0
        self._learned_games = 0
        self._score_list_size = 5
        self._best_score_list = [0] * self._score_list_size
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
    def search_width(self):
        return self._search_width
    @property
    def intuition_depth(self):
        return self._intuition_depth
    @property
    def total_games(self):
        return self._total_games
    @property
    def learned_games(self):
        return self._learned_games
    @property
    def best_score_list(self):
        return self._best_score_list
    @property
    def score_list_size(self):
        return self._score_list_size

    def __del__(self):
        self._policy_sess.close()
        self._value_sess.close()

    def get_params(self):
        return {
            'name': self._name,
            'para': self._para,
            # 'keep_prob': self._keep_prob,
            'search_depth': self._search_depth,
            'search_width': self._search_width,
            'total_games': self._total_games,
            'learned_games': self._learned_games,
            'best_score_list': self._best_score_list,
            'score_list_size': self._score_list_size,
            'intuition_depth': self._intuition_depth}

    def insert_cache_queue(self,score):
        self._total_games += 1
        for i in range(self._score_list_size):
            if score >= self._best_score_list[i]:
                self._learned_games += 1
                self._best_score_list.insert(i,score)
                self._best_score_list.pop()
                position = self._cache_game_list.pop()
                self._cache_game_list.insert(i,position)
                return position
        return -1

    def _build(self,name,para,search_depth,search_width,intuition_depth,
        total_games,learned_games,best_score_list,score_list_size,load=False):
        self._name = name
        self._para = para
        self._search_width = search_width
        self._search_depth = search_depth
        self._total_games = total_games
        self._score_list_size = score_list_size
        self._best_score_list = best_score_list
        self._cache_game_list = list(range(self._score_list_size))[::-1]
        self._learned_games = learned_games
        self._intuition_depth = intuition_depth
        self._virtual_board = Board(para)
        if load == False:
            try:
                self._value_model()
                self._policy_model()
            except:
                print('Model Initialization Fails!')
            else:
                print('Model Generated!')
        if load == True:
            try:
                with self._policy_graph.as_default():
                    saver = tf.train.import_meta_graph(
                        self._path + name + '_policy.meta')
                saver.restore(self._policy_sess,self._path + name + '_policy')
                with self._value_graph.as_default():
                    saver = tf.train.import_meta_graph(
                        self._path + name + '_value.meta')
                saver.restore(self._value_sess,self._path + name + '_value')
            except:
                print('Restoring Variables Fails!')
            else:
                print('Model Restored!')


    def new(self,name):
        ckpt_exists = tf.train.checkpoint_exists(self._path + name + '_value')
        if ckpt_exists:
            print('''The Name Has Been Used! Do you want to
            1. load it
            2. change a name
            3. overwrite it
            ''')
            choice = input('Please select: ')
            if choice == '1':
                self.load(name)
                return 1
            if choice == '2':
                self.new(input('New name: '))
        para = {
                    'size': eval(input('Size: ')),
                    'odd_2': eval(input('Odd of 2(between 0 and 1): '))
                }
        list_length = eval(input('Score List Length: '))
        params = {
                    'name': name,
                    'para': para,
                    # 'keep_prob': self._keep_prob,
                    'search_depth': eval(input('Search Depth: ')),
                    'search_width': eval(input('Search Width: ')),
                    'total_games': 0,
                    'score_list_size': list_length,
                    'best_score_list': [0]*list_length,
                    'learned_games': 0,
                    'intuition_depth': eval(input('Intuition Depth: '))
                  }
        self._build(**params)

    def load(self,name):
        with open(self._path + name + '_params.pkl','rb') as f:
            params = pickle.load(f)
        self._build(**params,load=True)

    def save(self):
        name = self._name
        with self._policy_graph.as_default():
            saver = tf.train.Saver()
        saver.save(self._policy_sess,self._path + name + '_policy')
        with self._value_graph.as_default():
            saver = tf.train.Saver()
        saver.save(self._value_sess,self._path + name + '_value')
        with open(self._path + name + '_params.pkl','wb') as f:
            pickle.dump(self.get_params(),f)

    def _batch_norm(self,x,depth):
        # Batch norm for convolutional maps
        beta = tf.Variable(tf.constant(0.0, shape=[depth]), name='beta',trainable=True)
        gamma = tf.Variable(tf.constant(0.0, shape=[depth]), name='gamma',trainable=True)
        with tf.variable_scope('bn'):
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            # batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
            normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
            return normed

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
            conv1 = tf.layers.conv2d(
                inputs = x,
                filters = 10,
                kernel_size = 2,
                padding = 'same',
                activation = tf.nn.relu
            )
            conv2 = tf.layers.conv2d(
                inputs = conv1,
                filters = 10,
                kernel_size = 3,
                padding = 'same',
                activation = tf.nn.relu
            )
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
            dense2 = tf.layers.dense(
                inputs=dropout2,
                units=1,
                activation = tf.nn.relu)
            loss = tf.losses.mean_squared_error(labels=tf.reshape(y,[-1,1]),
                predictions=dense2)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
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
            batch1 = self._batch_norm(x,1)
            conv1 = tf.layers.conv2d(
                inputs = batch1,
                filters = 10,
                kernel_size = 2,
                padding = 'same',
                activation = tf.nn.relu
            )
            batch2 = self._batch_norm(conv1,10)
            conv2 = tf.layers.conv2d(
                inputs = batch2,
                filters = 10,
                kernel_size = 3,
                padding = 'same',
                activation = tf.nn.relu
            )
            batch3 = self._batch_norm(conv2,10)
            flat = tf.reshape(batch3,shape=[-1,size*size*10])
            dropout1 = tf.scalar_mul(normalizer,tf.nn.dropout(x=flat,
                keep_prob=keep_prob))
            dense1 = tf.layers.dense(inputs=dropout1,
                units=20,
                activation = tf.nn.relu,
                name='dense1'
                )
            dropout2 = tf.nn.dropout(x=dense1,
                keep_prob=keep_prob) * normalizer
            dense2 = tf.layers.dense(
                inputs=dropout2,
                units=4)
            onehot_labels = tf.one_hot(indices=y,depth=4)
            loss = tf.reduce_mean(
                tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                    logits=dense2))
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
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
            if idx % 10 == 1:
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
            if idx % 10 == 1:
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

    def learn(self,batch_size,filenames,quiet=0):
        p_labels = []
        v_scores = []
        data = []
        for filename in filenames:
            print(filename)
            gamedata,para = self._load_game(filename)
            if not self._game_type_is(para):
                print(filename+' Data Type Not Match!')
                continue
            length = len(gamedata)
            p_labels.extend([gamedata[idx][-1] for idx in range(length)])
            # n_zeroes = [gamedata[idx][:-1].count(0) for idx in range(length)]
            # ma_zeros = [np.mean(n_zeroes[i:min(i+self._intuition_depth,length)])
             #   for i in range(length)]
            max_value = [max(gamedata[-1][:-1])] * length
            # n_zeroes = n_zeroes[self._intuition_depth:]
            # n_zeroes.extend([0]*self._intuition_depth)
            v_scores.extend(max_value)
            data.extend([gamedata[idx][:-1] for idx in range(length)])
        n_iter = len(data)
        data = self._log_board(data)
        data = self._convert_board(data)
        v_scores = self._log_board(v_scores)
        if quiet == 0:
            choose_net = input(
            '''Which net do you want to train:
                1. policy
                2. value
                3. both
                ''')
        else:
            choose_net = 3
        if choose_net != '2':
            self._fit_policy_net(data,p_labels,batch_size,n_iter)
        if choose_net != '1':
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
        if self._para != board.para:
            print('The ai player cannot play the game of the given parameters!')
            return 0
        self._best_move = None
        self._current_move = None
        self._best_value = -1
        self._current_value = 0
        self.search(board,self._search_depth)
        # print(self._best_move)
        return self._best_move

    def search(self,board,depth):
        if depth == self._search_depth:
            move_list = self.predict_policy(board)[:4]
        else:
            move_list = self.predict_policy(board)[:1]
        if move_list == []:
            self._current_value += 0
        if depth == 0:
            self._current_value += self.predict_value(board)
            return 1
        for move in move_list:
            for idx in range(self._search_width):
                if depth == self._search_depth:
                    if self._best_value < self._current_value:
                        self._best_move = self._current_move
                    self._current_value = 0
                    self._current_move = move
                self._virtual_board.load_board(board)
                self._virtual_board.move(move)
                self._virtual_board.next()
                board_tmp = copy.deepcopy(self._virtual_board)
                self.search(board_tmp,depth-1)

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
            ai_player.new(name)
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
