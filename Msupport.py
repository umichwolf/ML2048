#Some support functions

import sys
import re
import random

#select a best played game among several
def gamebetter(list1,list2):
    if list2 == None:
        return True
    elif max(list1) > max(list2):
        return True
    elif sum(list1) > sum(list2):
        return True
    else:
        return False


#print out an order 
def dictprint(order):
    print("")
    for key, value in order.items():
        if value != None:
            print(key, ": ", value)
            
#get parameters from a list         
def parametrize(list):
    dict = {
        "numofplay": None,
        "outputfile": None,
        "method": None,
        "numperselect":None,
        "machinetype": None,
        "machinename": None
   }
    
    order_dict = {
        "numofplay": "np:",
        "outputfile": "o:",
        "method": "m:",
        "numperselect": "ns:",
        "machinetype": "t:",
        "machinename": "n:"
    }
    for key,value in order_dict.items():
        #print key,value
        for i in range(1,len(list)):
            #print value,list[i]
            if value in list[i]:
                dict[key] = (re.findall(value+'(.*)',list[i]))[0]
    
    #print dict
    
    return dict
    
    
#get parameters from an order   
def get_para(label,list):
    for i in range(1,len(list)):
        print(label,list[i])
        if re.findall(label+'(.*?)(?: |$)',list[i]) != []:
            return re.findall(label+'(.*?)(?: |$)',list[i])
        
#weighted random choice from http://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
def weighted_choice(choices):
   total = sum(w for c, w in choices.items())
   r = random.uniform(0, total)
   upto = 0
   for c, w in choices.items():
      if upto + w >= r:
         return c
      upto += w
   assert False, "Shouldn't get here"    
