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

bestscore = 0
for gamma in np.arange(-1.7,-1.4,0.05):
    for alpha in np.arange(-2.7,-2.4,0.05):
        game_info = {"type":"2048",
                     "size":4,
                     "moves":["w","a","s","d"]}
        machine_info = {"type":"SVM",
                        "gamma":10**float(gamma),
                        "no_features":500,
                        "alpha":10**float(alpha)}
        mgr.new(game_info,machine_info,"better")
        score = mgr.machines[0]["machine"].train(X,Y)
        if score > bestscore:
            bestgamma = gamma
            bestalpha = alpha
            bestscore = score
        mgr.remove(0)
        print alpha,gamma
print "bestgamma=",bestgamma,"\n"
print "bestalpha=",bestalpha,"\n"
print "bestscore=",bestscore,"\n"
