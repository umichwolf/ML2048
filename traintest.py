from solver import Msolver
import numpy as np
import Minput

# Read training data from output file
filename = "data_train.txt"
data = Minput.read_training_data(filename)
X = data["X"]
Y = data["Y"]
mgr = Msolver.Solver()
X = mgr.cleanup(X)
game_info = {"type":"2048",
             "size":4,
             "moves":["w","a","s","d"]}
machine_info = {"type":"SVM",
                "gamma":10**(-1.4),
                "no_features":500,
                "alpha":10**(-2.45)}
mgr.new(game_info,machine_info,"better")
score = mgr.machines[0]["machine"].train(X,Y)
print score
mgr.save(0)
