# This module defines the classes of machines that can be used
# to solve the games.
import numpy as np
from joblib import Parallel, delayed
import tensorflow as tf
import Mcnn

class tfRF2L:
    """
    This is class constructs a 2-layer net with cos and sin nodes
    in the hidden layer. The weights in the first layer is
    initialized using random Gaussian features.
    Layerwise training can be applied.
    """
    def __init__(self,n_old_features,
        n_components,Lambda,Gamma,classes,
        loss_fn='log loss',log=False):
        self._d = n_old_features
        self._N = n_components
        self._Lambda = Lambda
        self._Gamma = Gamma
        self._classes = classes
        self._loss_fn = loss_fn
        self.log = log
        self._total_iter = 0
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        self._model_fn()

    @property
    def d(self):
        return self._d
    @property
    def N(self):
        return self._N
    @property
    def Lambda(self):
        return self._Lambda
    @property
    def Gamma(self):
        return self._Gamma
    @property
    def classes(self):
        return self._classes
    @property
    def loss_fn(self):
        return self._loss_fn
    @property
    def total_iter(self):
        return self._total_iter

    def _model_fn(self):
        d = self._d
        N = self._N
        Lambda = self._Lambda
        Gamma = self._Gamma
        n_classes = len(self._classes)
        loss_fn = self._loss_fn

        with self._graph.as_default():
            global_step_1 = tf.Variable(0,trainable=False,name='global1')
            global_step_2 = tf.Variable(0,trainable=False,name='global2')
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,d],name='features')
            y = tf.placeholder(dtype=tf.uint8,
                shape=[None],name='labels')

            with tf.name_scope('RF_layer'):
                initializer = tf.random_normal_initializer(
                    stddev=tf.sqrt(Gamma))

                trans_layer = tf.layers.dense(inputs=x,units=N,
                    use_bias=False,
                    kernel_initializer=initializer,
                    name='Gaussian')

                cos_layer = tf.cos(trans_layer)
                sin_layer = tf.sin(trans_layer)
                concated = tf.concat([cos_layer,sin_layer],axis=1)
                RF_layer = tf.div(concated,tf.sqrt(N*1.0))
                tf.summary.histogram('inner weights',
                    self._graph.get_tensor_by_name('Gaussian/kernel:0'))

            logits = tf.layers.dense(inputs=RF_layer,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=Lambda),
                units=n_classes,name='Logits')
            tf.add_to_collection("Probab",logits)
            tf.summary.histogram('outer weights',
                self._graph.get_tensor_by_name('Logits/kernel:0'))

            probab = tf.nn.softmax(logits, name="softmax")
            tf.add_to_collection("Probab",probab)

            # hinge loss only works for binary classification.
            regularizer = tf.losses.get_regularization_loss(scope='Logits')
            # if n_classes == 2:
            #     loss_hinge = tf.losses.hinge_loss(labels=y,
            #         logits=logits,
            #         loss_collection="loss") + regularizer
            #     loss_ramp = (tf.max(loss_hinge,1)
            #         + regularizer)
            #     tf.add_to_collection("loss",loss_ramp)
            onehot_labels = tf.one_hot(indices=y, depth=n_classes)
            loss_log = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=logits)
            reg_log_loss = tf.add(tf.reduce_mean(loss_log),regularizer)
            tf.add_to_collection('Loss',reg_log_loss)

            merged = tf.summary.merge_all()
            tf.add_to_collection('Summary',merged)
            self._sess.run(tf.global_variables_initializer())

        if self.log:
            summary = self._sess.run(merged)
            self._train_writer.add_summary(summary)

    def predict(self,data):
        with self._graph.as_default():
            feed_dict = {'features:0':data}
            logits,probab = tf.get_collection('Probab')
            predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "indices": tf.argmax(input=logits,axis=1),
                # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                # `logging_hook`.
                "probabilities": probab}
        results = self._sess.run(predictions,feed_dict=feed_dict)
        classes = [self._classes[index] for index in results['indices']]
        probabilities = results['probabilities']
        return classes,probabilities

    def score(self,data,labels):
        pred,_ = self.predict(data)
        accuracy = np.sum(pred==labels) / 100
        return accuracy

    def fit(self,data,labels,mode='layer 2',
        batch_size=1,n_iter=1000):
        indices = [self._classes.index(label) for label in labels]
        indices = np.array(indices)
        with self._graph.as_default():
            in_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                'Gaussian')
            out_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                'Logits')
            # if self._n_classes == 2:
            #     h_loss,r_loss,l_loss = tf.get_collection('loss')
            # else:
            #     l_loss = tf.get_collection('loss')
            # if self._loss_fn == 'hinge loss':
            #     loss = h_loss
            # elif self._loss_fn == 'log loss':
            #     loss = l_loss
            loss = tf.get_collection('Loss')[0]
            global_step_1 = self._graph.get_tensor_by_name('global1:0')
            global_step_2 = self._graph.get_tensor_by_name('global2:0')
            merged = tf.get_collection('Summary')[0]
            if mode == 'layer 2':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=10)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50.,
                 #   l2_regularization_strength=0.)
                # optimizer = tf.train.AdamOptimizer(learning_rate=10.)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_2,
                    var_list=out_weights
                )
                # self._sess.run(tf.global_variables_initializer())
            if mode == 'layer 1':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=10)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50.,
                #    l2_regularization_strength=0.)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50,
                #     l2_regularization_strength=0.)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_1,
                    var_list=in_weights
                )
                # self._sess.run(tf.global_variables_initializer())
            if mode == 'over all':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=10)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50.,
                 #   l2_regularization_strength=0.)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50,
                #     l2_regularization_strength=0.)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_1,
                    )
                # self._sess.run(tf.global_variables_initializer())
            if self.log:
                self._train_writer = tf.summary.FileWriter('tmp',
                    tf.get_default_graph())

        for idx in range(n_iter):
            rand_list = np.random.randint(len(data),size=batch_size)
            feed_dict = {'features:0':data[rand_list,:],
                         'labels:0':indices[rand_list]}
            if idx % 10 == 1:
                if self.log:
                    print('iter: {0:d}, loss: {1:.4f}'.format(
                        idx, self._sess.run(loss,feed_dict)))
                    summary = self._sess.run(merged)
                    self._train_writer.add_summary(summary,self._total_iter)
            self._sess.run(train_op,feed_dict)
            self._total_iter += 1

    def get_params(self,deep=False):
        params = {
            'n_old_features': self._d,
            'n_components': self._N,
            'Lambda': self._Lambda,
            'Gamma': self._Gamma,
            'classes': self._classes,
            'loss_fn': self._loss_fn
        }
        return params

    def __del__(self):
        self._sess.close()
        print('Session is closed.')

class SVM(tfRF2L):
    def __init__(self,classes,gamma=1,no_features=10,alpha=10**(-5)):
        super().__init__(n_old_features=16,
            n_components=no_features,
            Lambda=alpha,
            Gamma=gamma,
            classes=classes,
            loss_fn='log loss',
            log=False)

    def train(self,X,Y):
        X_features = self.rbf_features.fit_transform(X)
        self.partial_fit(X_features,Y,self.classes)
        score = self.score(X_features,Y)
        return score

    def test(self,X):
        X = X.reshape(1,-1)
        X_features = self.rbf_features.transform(X)
        proba = self.predict_proba(X_features)
        ans = dict()
        for idx in range(len(self.classes)):
            ans[self.classes_[idx]] = proba[0,idx]
        return ans

class CNN:
    def __init__(self):
        self.flag = 1

    def test(self,X):
        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
            model_fn=Mcnn.cnn_model_fn, model_dir="/tmp/2048_model")

        predict_input_fn = tf.numpy_input_fn(
            x={"x": X},
            num_epochs=1,
            shuffle=None,
        )
        proba = mnist_classifier.predict(
            predict_input_fn
        )


        print(proba)
        return proba
