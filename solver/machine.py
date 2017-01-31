# This module defines the classes of machines that can be used
# to solve the games.
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

class SVM(SGDClassifier):
    def __init__(self):
        SGDClassifier.__init__(self,gamma=1,no_feature=10)
# Establish the random kernel when the machine is
# built for the first time. Then it will be a constant
# along with the machine.
# Parameter gamma is usually of the size of the reciprocal
# of the averaged 2 norm square of difference between
# samples.
        self.rbf_features = RBFSampler(gamma=gamma,
                                       random_state=no_feature)

    def train(self,X,Y)
        X_features = self.rbf_features.fit_transform(X)
        self.parital_fit(X_features,Y)

    def test(self,X)
        X_features = self.rbf_features.fit_transform(X)
        ans = self.predict_proba(X)
        return ans
