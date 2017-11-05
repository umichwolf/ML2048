#This file will initialize the machine.
import sys
import Msupport

def machine_init():
    try:
        ini_fo = open("ini.txt",'r')
        ini_file = ini_fo.read()
        print("ini.txt successfully loaded.")
        print(ini_file)
    except:
        print("ini.txt is missing or broken. Machine shut down.")
        sys.exit(0)
        
    try:
        help_fo = open("help.txt",'r')
        help_file = help_fo.read()
        print("help.txt successfully loaded.")
        print(help_file)
    except:
        print("help.txt is missing or broken. Machine shut down.")
        sys.exit(0)
