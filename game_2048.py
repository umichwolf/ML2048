#This is the game "2048".

import copy
import random
import Msupport
from solver import Msolver

#Define Game Class
class Game(list):
#Initialization
    def __init__(self, parameter):
        list.__init__(self)
        self.para = {
            "size": 4,
            "inc": 1,
            "ratio": [0.5,0.5],
            "choice": random.choice             
        }
                
        self.zero_entry = list()
        for i in range(0,self.para["size"]):
            row = list()
            for j in range(0,self.para["size"]):
                row.append(0)
            self.append(row)
        
#Update zero entries list
    def update_zero_entry(self):
        self.zero_entry = list()
        for i in range(0,self.para["size"]):
            for j in range(0,self.para["size"]):
                if self[i][j] == 0:
                    row = [i,j]
                    self.zero_entry.append(row)

#Update all entries to be zero
    def allzero(self):
        for i in range(0,self.para["size"]):
            for j in range(0,self.para["size"]):
                self[i][j] = 0
                

#Tell a game end or not
    def gameend(self):
        tag = 0
        if self.zero_entry == []:
            tag = 1
            for j in range(0,self.para["size"]):
                for i in range(0,self.para["size"]-1):
                    if self[i][j] == self[i+1][j]:
                        tag = 0
                        break
                if tag == 0:
                    break
            if tag == 1:
                for i in range(0,self.para["size"]):
                    for j in range(0,self.para["size"]-1):
                        if self[i][j] == self[i][j+1]:
                            tag = 0
                            break
                    if tag == 0:
                        break
        return tag

                    
#Generate next number               
    def next(self):
        self.update_zero_entry()
        #print self.zero_entry
        temp = (self.para["choice"])(self.zero_entry)
        rand = random.uniform(0,1)
        #print "rand: ",rand
        i = 0
        while i <= len(self.para["ratio"]):
            if rand <= (self.para["ratio"])[i]:
                self[temp[0]][temp[1]] = 2**(i+1)
                #print self[temp[0]][temp[1]]
                break
            i = i + 1
            rand = rand - (self.para["ratio"])[i]
        self.update_zero_entry()
#Find the max entry
    def find_max(self):
        max_entry = {'row':-1,'col':-1,'value':-1}
        for i in range(0,self.para["size"]):
            for j in range(0,self.para["size"]):
                if max_entry['value'] <= self[i][j]:
                    max_entry['row'] = i
                    max_entry['col'] = j
                    max_entry['value'] = self[i][j]
        return max_entry
#Print the game
    def printout(self):
        l = len(str((self.find_max())['value']))
        for i in range(0,self.para["size"]):
            print '|',
            for j in range(0,self.para["size"]):
                fill = l - len(str(self[i][j]))
                space = ' '* fill
                print space + str(self[i][j]) + '|',
            print '\n'
#Basic operations          
    def move_up(self):
        for j in range(0,self.para["size"]):
            temp = list()
            for i in range(0,self.para["size"]):
                if self[i][j] != 0:
                    temp.append(self[i][j])
            i = 0
            while i < len(temp)-1:
                if temp[i] == temp[i+1]:
                    temp[i] = temp[i]*2
                    temp[i+1] = 0
                else: i = i+1
            temp1 = list()
            for i in range(0,len(temp)):
                if temp[i] != 0:
                    temp1.append(temp[i])
            for i in range(0,self.para["size"]):
                if i < len(temp1):
                    self[i][j] = temp1[i]
                else: self[i][j] = 0    
                            
    def move_down(self):
        for j in range(0,self.para["size"]):
            temp = list()
            for i in range(0,self.para["size"]):
                   temp.append(self[i][j])
            for i in range(0,self.para["size"]):
                   self[self.para["size"]-i-1][j] = temp[i]
        self.move_up()
        for j in range(0,self.para["size"]):
            temp = list()
            for i in range(0,self.para["size"]):
                   temp.append(self[i][j])
            for i in range(0,self.para["size"]):
                   self[self.para["size"]-i-1][j] = temp[i]
        
    def move_left(self):
        for i in range(0,self.para["size"]):
            temp = list()
            for j in range(0,self.para["size"]):
                if self[i][j] != 0:
                    temp.append(self[i][j])
            j = 0
            while j < len(temp)-1:
                if temp[j] == temp[j+1]:
                    temp[j] = temp[j]*2
                    temp[j+1] = 0
                else: j = j+1
            temp1 = list()
            for j in range(0,len(temp)):
                if temp[j] != 0:
                    temp1.append(temp[j])
            for j in range(0,self.para["size"]):
                if j < len(temp1):
                    self[i][j] = temp1[j]
                else: self[i][j] = 0
                
    def move_right(self):
        for i in range(0,self.para["size"]):
            temp = list()
            for j in range(0,self.para["size"]):
                   temp.append(self[i][j])
            for j in range(0,self.para["size"]):
                   self[i][self.para["size"]-j-1] = temp[j]
        self.move_left()
        for i in range(0,self.para["size"]):
            temp = list()
            for j in range(0,self.para["size"]):
                   temp.append(self[i][j])
            for j in range(0,self.para["size"]):
                   self[i][self.para["size"]-j-1] = temp[j]
                   
class Game_play(Game):
    def __init__(self, order, solvermgr):
        self.move_dict={
            'w':1,
            's':2,
            'a':3,
            'd':4
            }
        self.method_dict = {
            "naive": self._2048_naivesolver,
            "ml": self._2048_mlsolver
            }
        para = Msupport.parametrize(order["para"])
        machine = Msupport.parametrize(order["machine"])
        self.mgr = solvermgr
        if (para)["method"] == 'ml':
            n_machine = solvermgr.load(machine["machinetype"],machine["machinename"])
        self.machine = solvermgr.machines[n_machine-1]["machine"]
        parameter = list()
        Game.__init__(self,parameter)
        self.data = list()
        self.finaldata = list()
        
    #countempty elements
    def countempty(self):
        a = 0
        for i in range(0,self.para["size"]):
            for j in range(0,self.para["size"]):
                if self[i][j] == 0:
                    a = a + 1
        return a
    
    
    def data_foutput(self,filename):
        try:
            fo = open(filename,'w')
            #print self.data
            for onegame in self.data:
                for onestep in onegame:
                    for i in onestep:
                        print >> fo, i,' ',
                    print >> fo, '\n'
            fo.close()
        except:
            print "Output to ",filename," failed."

    def finaldata_foutput(self,filename):
        try:
            fo = open(filename,'w')
            #print self.data
            for onegame in self.finaldata:
                for i in onegame:
                    print >> fo, i,' ',
                print >> fo, '\n'
            fo.close()
        except:
            print "Output to ",filename," failed."

            
    #Find greatest corner
    def greatestcorner(self):
        a = [-1,-1]
        if self[0][0] == self.find_max()['value']:
            a[0] = 0
            a[1] = 0
        elif self[0][self.para["size"]-1] == self.find_max()['value']:
            a[0] = 0
            a[1] = self.para["size"] - 1
        elif self[self.para["size"]-1][0] == self.find_max()['value']:
            a[0] = self.para["size"] - 1
            a[1] = 0
        elif self[self.para["size"]-1][self.para["size"]-1] == self.find_max()['value']:
            a[0] = self.para["size"] - 1
            a[1] = self.para["size"] - 1
        return a
    #Check possiblity of moving greatest entry to the corner
    def _2048_naivesolver(self):
        cornerscore = {'w':0,'a':0,'s':0,'d':0}
        spacescore = {'w':0,'a':0,'s':0,'d':0}
        
        tempgame = list()
        
        #print "got herer!!!!!!!!!!!!!!"
        for i in range(0,self.para["size"]):
            tempgame.append(self[i][:])
        #print tempgame
        self.move_up()
        #print "get here naivesolver!!!1"
        if tempgame[:] != self[:]:
            if self.greatestcorner()[1] != -1: cornerscore['w'] = 1
            spacescore['w'] = self.countempty()
        else: spacescore['w'] = 0
        for i in range(0,self.para["size"]):
            for j in range(0,self.para["size"]):
                 self[i][j] = tempgame[i][j]
        
        self.move_left()
        if tempgame[:] != self[:]:
            if self.greatestcorner()[1] != -1: cornerscore['a'] = 1
            spacescore['a'] = self.countempty()
        else: spacescore['a'] = 0
        for i in range(0,self.para["size"]):
            for j in range(0,self.para["size"]):
                 self[i][j] = tempgame[i][j]

        
        self.move_down()
        if tempgame[:] != self[:]:
            if self.greatestcorner()[1] != -1: cornerscore['s'] = 1
            spacescore['s'] = self.countempty()
        else: spacescore['s'] = 0
        for i in range(0,self.para["size"]):
            for j in range(0,self.para["size"]):
                 self[i][j] = tempgame[i][j]

        
        self.move_right()
        if tempgame[:] != self[:]:
            if self.greatestcorner()[1] != -1: cornerscore['d'] = 1
            spacescore['d'] = self.countempty()
        else: spacescore['d'] = 0
        for i in range(0,self.para["size"]):
            for j in range(0,self.para["size"]):
                 self[i][j] = tempgame[i][j]
        #print tempgame
        #print "= = = = ="
        #self.printout()
                
        movelist = list()
        a = -1
        for i in ['w','a','s','d']:
            b = cornerscore[i]*self.para["size"]*self.para["size"]+spacescore[i]
            if b > a:
                movelist = [i]
                a = b
            elif b==a:
                movelist.append(i)
        if len(movelist) > 1:
            move = random.choice(movelist)
        else:
            move = movelist[0]
    #variable control
        #print cornerscore
        #print spacescore
        return move

    #########plaey a whole game and record the process ######
            
    def gameplay(self,choicefunc):
        #print "here i am "
        onegame = list()
        self.allzero()
        self.next()
        self.next()
        tag = 0
        while tag == 0:
            onestep = self[0][:]
            for i in range(1,self.para["size"]):
                onestep = onestep + self[i][:]
            #self.printout()
            move = choicefunc()
            #print "get here gameplay!!!!!!!!!!"
            onestep.append(move)
            onegame.append(onestep)
            #print move
            if  move == 'a':
                self.move_left()
            elif move == 's':
                self.move_down()
            elif move == 'd':
                self.move_right()
            else: self.move_up()
                
            self.next()
            tag = self.gameend()
            
        self.printout()
        
        onestep = self[0][:]
        for i in range(1,self.para["size"]):
            onestep = onestep + self[i][:]
        #onestep.append(None)
        onegame.append(onestep)
        return onegame,onestep
        

    def _2048_mlsolver(self):   
        tempgame = list()
        for i in range(0,self.para["size"]):
            tempgame.extend(self[i][:])
        tempgame = self.mgr.cleanup(tempgame)
        movelist = self.machine.test(tempgame)    
        while True:
            move = Msupport.weighted_choice(movelist)
            if self.validate_move(move):
                break
            movelist.pop(move)
        return move
        
    #play only one game but collect all data
    def oneplay(self,para_dict):
        #print "get here oneplay"
        onegame, gamefinal = self.gameplay(self.method_dict[para_dict["method"]])
        #print "get here oneplay"
        self.data.append(onegame)
        self.finaldata.append(gamefinal)
        try:
            self.data_foutput(para_dict["outputfile"])
        except:
            print "Output to ",para_dict["data_foutputfile"]," failed."
    
    #play multiple games but only collect the final step data
    def play(self,para_dict):
        for i in range(int(para_dict["numofplay"])):
            onegame, gamefinal = self.gameplay(self.method_dict[para_dict["method"]])
            self.finaldata.append(gamefinal)        
        try:
            self.finaldata_foutput(para_dict["outputfile"])
        except:
            print "Output to ",para_dict["outputfile"]," failed."

    #return a best game data among n many of them
    def selectone(self,para_dict):
        gameselected = None
        finalselected = None
        for i in range(int(para_dict["numperselect"])):
            curgame, curfinal = self.gameplay(self.method_dict[para_dict["method"]])
            if Msupport.gamebetter(curfinal,finalselected) == True:
                gameselected = curgame
                finalselected = curfinal
        return gameselected, finalselected
                
    def playselect(self,para_dict):
        for i in range(int(para_dict["numofplay"])):
            onegame, gamefinal = self.selectone(para_dict)
            self.data.append(onegame)
            self.finaldata.append(gamefinal)        
        try:
            self.finaldata_foutput(para_dict["outputfile"])
            self.data_foutput("output.txt")
        except:
            print "Output to ",para_dict["outputfile"]," failed."   

    def validate_move(self,move):
        tempgame = list()
        for i in range(0,self.para["size"]):
            tempgame.append(self[i][:])
        if move == 'a':
            self.move_left()
        elif move == 's':
            self.move_down()
        elif move == 'd':
            self.move_right()
        elif move == 'w':
            self.move_up()    
        else:
            print "illegal input!"
        if tempgame[:] != self[:]:
            for i in range(0,self.para["size"]):
                for j in range(0,self.para["size"]):
                     self[i][j] = tempgame[i][j]
            return True
        else: return False
    
#The following is the main program.

if __name__ == '__main__':
    game_play = Game_play(1)
    #game_play.naiveplay()
    #game_play.data_foutput("output.txt")
    
    
