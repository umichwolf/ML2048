from solver import Msolver
import numpy as np
import Minput
import random

mgr = Msolver.Solver()
filename = "data_train.txt"
data_list = Minput.read_training_data(filename)
X = data_list["X"]
Y = data_list["Y"]
Y_hv = Y
for idx in range(len(Y)):
    if Y[idx] == "w" or Y[idx] == "s":
        Y_hv[idx] = "v"
    else:
        Y_hv[idx] = "h"
game_info = {"type":"2048",
             "size":4,
             "moves":["h","v"]}
machine_info = {"type":"SVM",
                "gamma":10**(-1.4),
                "no_features":500,
                "alpha":10**(-2.45)}
X = mgr.cleanup(X)
mgr.new(game_info,machine_info,"hv")
score = mgr.machines[0]["machine"].train(X,Y)
print score
#mgr.save(0)
