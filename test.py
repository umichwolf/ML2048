import Msolver
import numpy as np

# Initialize parameters
X = np.array([[0,1],[1,0],[1,1]])
Y = ["a","b","c"]
game_info = {"type":"2048",
             "size":4,
             "moves":["c","a","b"]}
machine_info = {"type":"SVM",
                "gamma":1,
                "no_features":3}

# Initialize Msolver manager
mgr = Msolver.Solver()

# Build a new machine
mgr.new(game_info,machine_info,"test1")
print mgr.machines[0]

# Train and test the new machine
# mgr.machines[0]["machine"].train(X,Y)
# print mgr.machines[0]["machine"].classes_
# ans = mgr.machines[0]["machine"].test([[1,2]])
# print ans

# Show the machines
mgr.show("l")

# Save and load the machine
mgr.save(0)
mgr.load("SVM","test1")
mgr.show()

