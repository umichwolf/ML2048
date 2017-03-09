from solver import Msolver
import numpy as np

# Read training data from output file
try:
    f = open("data_train.txt","r")
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
                "gamma":0.002,
                "no_features":1000,
                "alpha":10**(-20)}

mgr = Msolver.Solver()
#mgr.load('SVM','train1')
mgr.new(game_info,machine_info,"better")
data_list = mgr.machines[0]["machine"].cleanup(data_list)
mgr.machines[0]["machine"].train(data_list,move_list)
mgr.save(0)

### check classification accuracy of machine 
#score = 0
#for idx in range(len(move_list)):
#    pre_list = mgr.machines[0]["machine"].test([data_list[idx,:]])
#    move_value = 0
#    for key,value in pre_list.items():
#        if value > move_value:
#            move = key
#    print move,' ',move_list[idx]
#    if move==move_list[idx]:
#        score = score + 1
#print score,' ',len(move_list)
