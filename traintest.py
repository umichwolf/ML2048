from solver import Msolver
import numpy as np
import Minput
from sklearn.model_selection import train_test_split

# Read training data from output file
filename = "data_train.txt"
data = Minput.read_training_data(filename)
X = data["X"]
Y = data["Y"]
mgr = Msolver.Solver()
X = mgr.cleanup(X)
X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=0.4,random_state=0)
game_info = {"type":"2048",
             "size":4,
             "moves":["w","a","s","d"]}
machine_info = {"type":"KSVM",
                "C":1,
                "gamma":10**(-1.6)}
mgr.new(game_info,machine_info,"better")
mgr.machines[0]["machine"].train(X,Y)
print mgr.machines[0]["machine"].n_support_
score = mgr.machines[0]["machine"].score(X,Y)
print score
#mgr.save(0)
