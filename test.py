from solver import Msolver
import numpy as np
import Minput
import random

filename = "data_train.txt"
data = Minput.read_training_data(filename)
X = data["X"]
Y = data["Y"]
mgr = Msolver.Solver()
game_info = {"type":"2048",
             "size":4,
             "moves":["w","a","s","d"]}
machine_info = {"type":"SVM",
                "gamma":0.002,
                "no_features":100,
                "alpha":10**(-20)}
mgr.new(game_info,machine_info,"dbg1")
Xlen = len(X)
no_samples = int(0.1*Xlen)
print Xlen
samples = random.sample(xrange(Xlen),no_samples)
Xsamples = [X[i] for i in samples]
print np.linalg.norm(Xsamples)/np.sqrt(no_samples)
