# The solver of games is used to build, train,
# test and manage the machines for games.
import machine

class Solver:
    def __init__(self, pwd):
        dict.__init__(self)
        self.pwd = pwd
        self.machines = dict()

# Build a new machine.
    def new(self, gameinfo, machine_type, neworold):
        if neworold = "old"
            self.machines[old] = load("old")
        if machine_type = "type":
            self.machines[gameinfo] = machine.machine_type(gameinfo)

# Show the list of machine
    def show(self):
        print self.machine

# Save a machine
    def save(self):
        save(self)

# Remove a machine from solver
    def remove(self):


