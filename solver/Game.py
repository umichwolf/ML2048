#This is the game "2048".
import numpy as np
import solver

#Define the size of the game
import random
def GAMESIZE():
        return 4
#Define Game Class
class Game(list):
#Initialization
        def __init__(self, GAMESIZE, RATIO):
                list.__init__(self)
                self.size = GAMESIZE
                self.ratio = RATIO
                self.ratiolist = [4]
                for i in range(0,RATIO):
                        self.ratiolist.append(2)
                #Initialize our zero entries list
                self.zero_entry = list()
                for i in range(0,GAMESIZE):
                        row = list()
                        for j in range(0,GAMESIZE):
                                row.append(0)
                        self.append(row)
                
#Update zero entries list
        def update_zero_entry(self):
                self.zero_entry = list()
                for i in range(0,self.size):
                        for j in range(0,self.size):
                                if self[i][j] == 0:
                                        row = [i,j]
                                        self.zero_entry.append(row)
#Generate next number                                
        def next(self):
                self.update_zero_entry()
                temp = random.choice(self.zero_entry)
                self[temp[0]][temp[1]] = random.choice(self.ratiolist)
                self.update_zero_entry()
#Find the max entry
        def find_max(self):
                max_entry = {'row':-1,'col':-1,'value':-1}
                for i in range(0,self.size):
                        for j in range(0,self.size):
                                if max_entry['value'] <= self[i][j]:
                                        max_entry['row'] = i
                                        max_entry['col'] = j
                                        max_entry['value'] = self[i][j]
                return max_entry
#Print the game
        def printout(self):
                l = len(str((self.find_max())['value']))
                for i in range(0,self.size):
                        print '|',
                        for j in range(0,self.size):
                                fill = l - len(str(self[i][j]))
                                space = ' '* fill
                                print space + str(self[i][j]) + '|',
                        print '\n'
#Basic operations               
        def move_up(self):
                for j in range(0,self.size):
                        temp = list()
                        for i in range(0,self.size):
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
                        for i in range(0,self.size):
                                if i < len(temp1):
                                        self[i][j] = temp1[i]
                                else: self[i][j] = 0        
                                                        
        def move_down(self):
                for j in range(0,self.size):
                        temp = list()
                        for i in range(0,self.size):
                               temp.append(self[i][j])
                        for i in range(0,self.size):
                               self[self.size-i-1][j] = temp[i]
                self.move_up()
                for j in range(0,self.size):
                        temp = list()
                        for i in range(0,self.size):
                               temp.append(self[i][j])
                        for i in range(0,self.size):
                               self[self.size-i-1][j] = temp[i]
                
        def move_left(self):
                for i in range(0,self.size):
                        temp = list()
                        for j in range(0,self.size):
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
                        for j in range(0,self.size):
                                if j < len(temp1):
                                        self[i][j] = temp1[j]
                                else: self[i][j] = 0
                                
        def move_right(self):
                for i in range(0,self.size):
                        temp = list()
                        for j in range(0,self.size):
                               temp.append(self[i][j])
                        for j in range(0,self.size):
                               self[i][self.size-j-1] = temp[j]
                self.move_left()
                for i in range(0,self.size):
                        temp = list()
                        for j in range(0,self.size):
                               temp.append(self[i][j])
                        for j in range(0,self.size):
                               self[i][self.size-j-1] = temp[j]
                
############################################## Game Solver ######################################
#Count empty blocks
def solver_countempty(GAME):
        a = 0
        for i in range(0,GAME.size):
                for j in range(0,GAME.size):
                        if GAME[i][j] == 0:
                                a = a + 1
        return a
#Find greatest corner
def solver_greatestcorner(GAME):
        a = [-1,-1]
        if GAME[0][0] == GAME.find_max()['value']:
                a[0] = 0
                a[1] = 0
        elif GAME[0][GAME.size-1] == GAME.find_max()['value']:
                a[0] = 0
                a[1] = GAME.size - 1
        elif GAME[GAME.size-1][0] == GAME.find_max()['value']:
                a[0] = GAME.size - 1
                a[1] = 0
        elif GAME[GAME.size-1][GAME.size-1] == GAME.find_max()['value']:
                a[0] = GAME.size - 1
                a[1] = GAME.size - 1
        return a
#Check possiblity of moving greatest entry to the corner
def solver_moveanalysis(GAME):
        cornerscore = {'w':0,'a':0,'s':0,'d':0}
        spacescore = {'w':0,'a':0,'s':0,'d':0}
        game_temp = Game(GAME.size,GAME.ratio)
        for i in range(0,GAME.size):
                for j in range(0,GAME.size):
                        game_temp[i][j] = GAME[i][j]
        game_temp.move_up()
        if game_temp[:] != GAME[:]:
                if solver_greatestcorner(game_temp)[1] != -1: cornerscore['w'] = 1
                spacescore['w'] = solver_countempty(game_temp)
        else: spacescore['w'] = 0

        game_temp = Game(GAME.size,GAME.ratio)
        for i in range(0,GAME.size):
                for j in range(0,GAME.size):
                        game_temp[i][j] = GAME[i][j]
        game_temp.move_left()
        if game_temp[:] != GAME[:]:
                if solver_greatestcorner(game_temp)[1] != -1: cornerscore['a'] = 1
                spacescore['a'] = solver_countempty(game_temp)
        else: spacescore['a'] = 0
       
        game_temp = Game(GAME.size,GAME.ratio)
        for i in range(0,GAME.size):
                for j in range(0,GAME.size):
                        game_temp[i][j] = GAME[i][j]
        game_temp.move_down()
        if game_temp[:] != GAME[:]:
                if solver_greatestcorner(game_temp)[1] != -1: cornerscore['s'] = 1
                spacescore['s'] = solver_countempty(game_temp)
        else: spacescore['s'] = 0

        game_temp = Game(GAME.size,GAME.ratio)
        for i in range(0,GAME.size):
                for j in range(0,GAME.size):
                        game_temp[i][j] = GAME[i][j]
        game_temp.move_right()
        if game_temp[:] != GAME[:]:
                if solver_greatestcorner(game_temp)[1] != -1: cornerscore['d'] = 1
                spacescore['d'] = solver_countempty(game_temp)
        else: spacescore['d'] = 0

        a = -1
        move = ' '
        for i in ['w','a','s','d']:
                b = a
                a = max(a,cornerscore[i]*GAME.size*GAME.size+spacescore[i])
                if b != a: move = i
#variable control
        print cornerscore
        print spacescore
        return move
#Main solver
def solver_main(GAME):
        return solver_moveanalysis(GAME)[0]
############################################## Main #############################################                      
def main(machine):
        print "Let's Play \"2048\"!\nSelect the size: 2, 3, 4, 5"
        a = input()
        print "Then input the ratio: 2, 5, 10"
        b = input()
        game = Game(a,b)
        print "Now, start. \nUse 'a' for left move,\n'd' for right,\n's' for downward,\n'w' for upward."
        game.next()
        game.next()
        tag = 0
        while tag == 0:
                temp = list()
                for i in range(0,game.size):
                        temp.append(game[i])
                if 2048 in temp:
                        print "You get 2048!"
                game.printout()
                print "next step:"
                prestep = list()
                for i in range(0,game.size):
                        prestep.append(game[i][:])
                data_input = np.array([sum(game,[])])
                data_input[data_input==0] = 1
                data_input = np.log(data_input)/np.log(2)
                solution_set = machine.test(data_input)
                while prestep[:] == game[:]:
                        #a = raw_input()
                        #a = solver_main(game)
                        proba_init = 0
                        for key in solution_set:
                            proba = solution_set[key]
                            if proba > proba_init:
                                a = key
                            proba_init = proba
                        print a,proba
                       # while a not in ['a','s','d','w']:
                       #         print "Invalid input, next step"
                                #a = raw_input()
#toggle control
                       #         break
                       #         a = solver_main(game)
                        if a == 'a':
                                game.move_left()
                        elif a == 's':
                                game.move_down()
                        elif a == 'd':
                                game.move_right()
                        else: game.move_up()
                        if prestep[:] == game[:]:
                                local_flag = 0
                                print "Invalid move, next step"
                                soltemp = solution_set.pop(a)
                game.next()
                if game.zero_entry == []:
                        tag = 1
                        for j in range(0,game.size):
                                for i in range(0,game.size-1):
                                        if game[i][j] == game[i+1][j]:
                                                tag = 0
                                                break
                                if tag == 0:
                                        break
                        if tag == 1:
                                for i in range(0,game.size):
                                        for j in range(0,game.size-1):
                                                if game[i][j] == game[i][j+1]:
                                                        tag = 0
                                                        break
                                        if tag == 0:
                                                break
#toggle control
#                if raw_input("Do you want next round?(y/n)") == 'n': break
        game.printout()
        print "Game over."
        
#The following is the main program.

if __name__ == '__main__':
        ans = 'y'
        while ans == 'y':
                mgr = solver.Solver()
                mgr.load("SVM","train1")
                main(mgr.machines[0]["machine"])
                ans = raw_input("Do you want another one?(y/n)")
