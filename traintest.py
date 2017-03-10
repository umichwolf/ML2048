from solver import Msolver
import numpy as np
import Minput

# Read training data from output file
filename = "data_train.txt"
data = Minput.read_training_data(filename)
X = data["X"]
Y = data["Y"]

game_info = {"type":"2048",
             "size":4,
             "moves":["w","a","s","d"]}
machine_info = {"type":"SVM",
                "gamma":10**(-5),
                "no_features":500,
                "alpha":10**(-1.9)}

mgr = Msolver.Solver()
#mgr.load('SVM','train1')
mgr.new(game_info,machine_info,"better")
data_list = mgr.machines[0]["machine"].cleanup(X)
mgr.machines[0]["machine"].train(X,Y)
mgr.save(0)
