# The solver of games is used to build, train,
# test and manage the machines for games.
from sklearn import preprocessing
import machine
import pickle
import numpy as np

class Solver:
    def __init__(self,pwd="./machines/"):
        self.pwd = pwd
        self.machines = list()

# Build a new machine.
    def new(self,game_info,machine_info,machine_name):
        if machine_info["type"] == "SVM":
            newm = machine.SVM(game_info["moves"],
                               machine_info["gamma"],
                               machine_info["no_features"],
                               machine_info["alpha"])
            item = {"name":machine_name,
                    "game_type":game_info["type"],
                    "game_size":game_info["size"],
                    "machine_type":machine_info["type"],
                    "gamma":machine_info["gamma"],
                    "no_features":machine_info["no_features"],
                    "alpha":machine_info["alpha"],
                    "machine":newm}
            self.machines.append(item)
            return len(self.machines)-1
        elif machine_info["type"] == "KSVM":
            newm = machine.KSVM(C=machine_info["C"],
                                gamma=machine_info["gamma"])
            item = {"name":machine_name,
                    "game_type":game_info["type"],
                    "game_size":game_info["size"],
                    "machine_type":machine_info["type"],
                    "gamma":machine_info["gamma"],
                    "C":machine_info["C"],
                    "machine":newm}
            self.machines.append(item)
            return len(self.machines)-1
        else:
            print "The machine type is not found."
            return -1
 
# Show the list of machine
    def show(self,l_s="s"):
        if len(self.machines)==0:
            print "No machines have been loaded or built yet!"
        if l_s=="l":
            for idx in range(len(self.machines)):
                print idx,"\n",self.machines[idx]
        else:
            for idx in range(len(self.machines)):
                print idx," ",self.machines[idx]["name"]

# Save a machine. The names of machines within the same type
# cannot duplicate.
    def save(self,midx):
        if type(midx) != int:
            print "Machine index must be a nonnegative integer."
            return 0
        elif midx >= len(self.machines):
            print "Machine index is out of range."
            return 0
        else:
            item = self.machines[midx]
            machine_type = item["machine_type"]
            machine_name = item["name"]
            pkl_filename = self.pwd + machine_type + "-" + machine_name + ".pkl"
            pkl_file = open(pkl_filename,"wb+")
            ans = pickle.dump(item,pkl_file)
            pkl_file.close()
            return ans

# Load a machine, and return its number.
    def load(self,machine_type,machine_name):
        pkl_filename = self.pwd + machine_type + "-" + machine_name + ".pkl"
        pkl_file = open(pkl_filename,'rb')
        item = pickle.load(pkl_file)
        pkl_file.close()
        self.machines.append(item)
        return len(self.machines)

# Remove a machine from solver
    def remove(self,index = -1):
        if type(index) != int:
            print "Machine index must be a nonnegative integer."
            return 0
        elif index >= len(self.machines):
            print "Machine index is out of range."
            return 0
        else:
            del self.machines[index]
            return 1

# Clean up the data before training a machine
    def cleanup(self,X):
        X = np.array(X)
        X[X==0] = 1
        X = np.log2(X)
#        X = preprocessing.scale(X)
        return X
