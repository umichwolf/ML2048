# This module defines the classes of machines that can be used
# to solve the games.
from sklearn import svm
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
import numpy as np

class SVM(SGDClassifier):
    def __init__(self,classes,gamma=1,no_features=10,alpha=10**(-5)):
        SGDClassifier.__init__(self)
        self.gamma = gamma
        self.no_features = no_features
        self.classes = classes
        self.loss = "modified_huber"
        self.alpha = alpha
# Establish the random kernel when the machine is
# built for the first time. Then it will be a constant
# along with the machine.
# Parameter gamma is usually of the size of the reciprocal
# of the averaged 2 norm square of difference between
# samples.
        self.rbf_features = RBFSampler(gamma=gamma,
                                       n_components=no_features)

    def train(self,X,Y):
        X_features = self.rbf_features.fit_transform(X)
        self.partial_fit(X_features,Y,self.classes)
        score = self.score(X_features,Y)
        return score

    def test(self,X):
        X = X.reshape(1,-1)
        X_features = self.rbf_features.fit_transform(X)
        proba = self.predict_proba(X_features)
        ans = dict()
        for idx in range(len(self.classes)):
            ans[self.classes_[idx]] = proba[0,idx]
        return ans

class KSVM(svm.SVC):
    def __init__(self,classes,C=1,gamma='auto',kernel='rbf'):
        svm.SVC.__init__(self,C=C,gamma=gamma,kernel=kernel,probability=True)
        self.classes = classes

    def train(self,X,Y):
        self.fit(X,Y)

    def test(self,X):
        X = X.reshape(1,-1)
        proba = self.predict_proba(X)
        ans = dict()
        for idx in range(len(self.classes)):
            ans[self.classes_[idx]] = proba[0,idx]
        return ans
