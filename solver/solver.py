# The solver of games is used to build, train,
# test and manage the machines for games.
import machine

class Solver:
    def __init__(self,pwd):
        self.pwd = pwd
        self.machines = list()

# Build a new machine.
    def new(self,game_info,machine_info,neworold):
        if neworold == "old":
            self.machines[old] = load("old")
        elif machine_info["type"] == "SVM":
            newm = machine.SVM(game_info["moves"],
                               machine_info["gamma"],
                               machine_info["no_features"])
            item = {"name":neworold,
                    "game_type":game_info["type"],
                    "game_size":game_info["size"],
                    "machine_type":machine_info["type"],
                    "gamma":machine_info["gamma"],
                    "no_features":machine_info["no_features"],
                    "machine_type":machine_info["type"],
                    "machine":newm}
            self.machines.append(item)

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

## Save a machine
#    def save(self):
#        save(self)
#
## Remove a machine from solver
#    def remove(self):


