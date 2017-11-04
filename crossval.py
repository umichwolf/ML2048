from solver import Msolver
import numpy as np
import Minput
from sklearn.model_selection import train_test_split

# Read training data from output file
filename = "data_train.txt"
data = Minput.read_training_data(filename)
X = data["X"]
Y = data["Y"]

Y_hv = Y
for idx in range(len(Y)):
    if Y[idx] == "w" or Y[idx] == "s":
        Y_hv[idx] = "v"
    else:
        Y_hv[idx] = "h"

mgr = Msolver.Solver()
X = mgr.cleanup(X)
X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y_hv,test_size=0.4,random_state=0)

bestscore = 0
for gamma in np.arange(-2,-1,0.3):
    for alpha in np.arange(-3,-2,0.3):
        game_info = {"type":"2048",
                     "size":4,
#                     "moves":["w","a","s","d"]
                     "moves":["h","v"]
                    }
        machine_info = {"type":"SVM",
                        "gamma":10**float(gamma),
                        "no_features":500,
                        "alpha":10**float(alpha)
                       }
        mgr.new(game_info,machine_info,"hv")
        train_score = mgr.machines[0]["machine"].train(X_train,Y_train)
        print train_score
        X_test = mgr.machines[0]["machine"].rbf_features.fit_transform(X_test)
        score = mgr.machines[0]["machine"].score(X_test,Y_test)
        if score > bestscore:
            bestgamma = gamma
            bestalpha = alpha
            bestscore = score
        mgr.remove(0)
        print alpha,gamma
print "bestgamma=",bestgamma
print "bestalpha=",bestalpha
print "bestscore=",bestscore
