##### This is the main code #####

import sys
import Msupport
import Minit
import Minput
import Mwork
from solver import Msolver

Minit.machine_init()


request = 1
while request:
    order = Minput.input()
    solvermgr = Msolver.Solver()
    Mwork.work(order,solvermgr)
    print("Do you have another request?")
    request = Minput.requestcheck()
    
    
    
print("Thank you! Bye-Bye!")

#Moutput.output()
