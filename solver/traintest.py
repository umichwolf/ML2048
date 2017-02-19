import solver
import numpy as np

# Read training data from output file
try:
    f = open("output.txt","r")
    dataset = list()
    for line in f:
        row = line.split()
        if row != []:
            dataset.append(row)
finally:
    f.close()

data_list = list()
move_list = list()
for idx in range(len(dataset)):
    if len(dataset[idx])==17:
        move = dataset[idx].pop(16)
    else:
        move = "f"
    if move in ["w","a","s","d"]:
        move_list.append(move)
        data_list.append(map(int,dataset[idx]))

game_info = {"type":"2048",
             "size":4,
             "moves":["w","a","s","d"]}
machine_info = {"type":"SVM",
                "gamma":0.01,
                "no_features":20}

mgr = solver.Solver()
mgr.new(game_info,machine_info,"train1")
data_list = np.array(data_list)
data_list[data_list==0] = 1
data_list = np.log(data_list)/np.log(2)
mgr.machines[0]["machine"].train(data_list,move_list)
mgr.save(0)
