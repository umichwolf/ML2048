#This is the game "2048".

import copy
import random
import Msupport
import tensorflow as tf
import Msolver

#define the board
class Board(list):
#Initialization
    def __init__(self, parameter):
        super().__init__(self)
        self._para = parameter # for example {"size": 4, "odd_2": 0.5}
        self._zero_entries_list = list()
        self.new_board()

    @property
    def para(self):
        return self._para
    @property
    def zero_entries_list(self):
        return self._zero_entries_list
# return a clean board
    def new_board(self):
        for i in range(0,self._para["size"]):
            row = list()
            for j in range(0,self._para["size"]):
                row.append(0)
            self.append(row)
        self.next()
        self.next()

# update zero entries list
    def _update_zero_entries_list(self):
        self._zero_entries_list = list()
        for i in range(0,self._para["size"]):
            for j in range(0,self._para["size"]):
                if self[i][j] == 0:
                    row = [i,j]
                    self._zero_entries_list.append(row)

# check the game ends or not
    def gameend(self):
        tag = 0
        if self._zero_entries_list == []:
            tag = 1
            for j in range(0,self._para["size"]):
                for i in range(0,self._para["size"]-1):
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

# generate next number
    def next(self):
        self._update_zero_entries_list()
        temp = random.choice(self.zero_entry)
        rand = random.uniform(0,1)
        if rand <= self._para["odd_2"]:
            self[temp[0]][temp[1]] = 2
        else:
            self[temp[0]][temp[1]] = 4
        self._update_zero_entries_list()

# find the max entry
    def find_max(self):
        max_entry = {'row':-1,'col':-1,'value':-1}
        for i in range(0,self.para["size"]):
            for j in range(0,self.para["size"]):
                if max_entry['value'] <= self[i][j]:
                    max_entry['row'] = i
                    max_entry['col'] = j
                    max_entry['value'] = self[i][j]
        return max_entry

# print the game board
    def print_board(self):
        l = len(str((self.find_max())['value']))
        for i in range(0,self._para["size"]):
            print('|',end="")
            for j in range(0,self._para["size"]):
                fill = l - len(str(self[i][j]))
                space = ' '* fill
                print(space + str(self[i][j]) + '|',end="")
            print('\n')

# basic operations
    def _move_up(self):
        for j in range(0,self._para["size"]):
            temp = list()
            for i in range(0,self._para["size"]):
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
            for i in range(0,self._para["size"]):
                if i < len(temp1):
                    self[i][j] = temp1[i]
                else: self[i][j] = 0

    def _move_down(self):
        for j in range(0,self._para["size"]):
            temp = list()
            for i in range(0,self._para["size"]):
                   temp.append(self[i][j])
            for i in range(0,self._para["size"]):
                   self[self._para["size"]-i-1][j] = temp[i]
        self._move_up()
        for j in range(0,self._para["size"]):
            temp = list()
            for i in range(0,self._para["size"]):
                   temp.append(self[i][j])
            for i in range(0,self._para["size"]):
                   self[self._para["size"]-i-1][j] = temp[i]

    def _move_left(self):
        for i in range(0,self._para["size"]):
            temp = list()
            for j in range(0,self._para["size"]):
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
            for j in range(0,self._para["size"]):
                if j < len(temp1):
                    self[i][j] = temp1[j]
                else: self[i][j] = 0

    def _move_right(self):
        for i in range(0,self._para["size"]):
            temp = list()
            for j in range(0,self._para["size"]):
                   temp.append(self[i][j])
            for j in range(0,self._para["size"]):
                   self[i][self._para["size"]-j-1] = temp[j]
        self._move_left()
        for i in range(0,self._para["size"]):
            temp = list()
            for j in range(0,self._para["size"]):
                   temp.append(self[i][j])
            for j in range(0,self._para["size"]):
                   self[i][self._para["size"]-j-1] = temp[j]
    def move(self, action):
        temp = list()
        temp[:] = self[:]
        if action == 'w':
            self._move_up()
        if action == 's':
            self._move_down()
        if action == 'a':
            self._move_left()
        if action == 'd':
            self._move_right()
        if temp[:] == self[:]:
            print('illegal move!')
            return 0
        return 1

class Game:
    """
    this class provides interfaces of playing games by
    human players or ai players, recording games, saving
    games and replaying games.
    """
    def __init__(self):
        self._game = list()
        self.pow = './game_archives/'
        self.idle()

    def idle(self):
        order = input("""Choose one from the list:
            1: new game\n2: new game by ai\n3: load\n4: replay\n5: exit""")
        if order == '1':
            parameter = {'size': input('size: '),
                'odd_2': input('odd of 2(between 0 and 1): ')}
            self.new_game(parameter)
        if order == '2':
            ai_player = input('Name of the ai player: ')
            self.new_game_by_ai(ai_player)
        if order == '3':
            self.load(self.pow + input('Name of the game: '))
        if order == '4':
            self.replay(self.pow + input('Name of the game: '))
        if order == '5':
            pass

    def new_game(self,parameter):
        self._board = Board(parameter)
        self._play()

    def _play(self):
        endgame_flag = self._board.gameend()
        while endgame == 0:
            order = input('Your Move(wsad,s:stop): ')
            if order in ['w','s','a','d']:
                self.push(self._board,order)
                if self._board.move(order):
                    self._board.next()
                else
            elif order == 's':
                break
            endgame_flag = self._board.gameend()

        if endgame_flag == 1:
            print('Game over!')
        if input('Do you want to save the game?(y/n) ') == 'y':
            self.save(input('Name of the game: '))
        self.idle()

    def save(self,filename):
        with open(self.pow+filename,'w') as f:
            for line in self._game:
                for char in line:
                f.write(str(char)+' ')
                f.write('\n')

    def finaldata_foutput(self,filename):
        try:
            fo = open(filename,'w')
            #print self.data
            for onegame in self.finaldata:
                for i in onegame:
                    s = str(i,' ')
                    fo.write(s)
                fo.write('\n')
            fo.close()
        except:
            print("Output to ",filename," failed.")

#
    # #Find greatest corner
    # def greatestcorner(self):
        # a = [-1,-1]
        # if self[0][0] == self.find_max()['value']:
            # a[0] = 0
            # a[1] = 0
        # elif self[0][self.para["size"]-1] == self.find_max()['value']:
            # a[0] = 0
            # a[1] = self.para["size"] - 1
        # elif self[self.para["size"]-1][0] == self.find_max()['value']:
            # a[0] = self.para["size"] - 1
            # a[1] = 0
        # elif self[self.para["size"]-1][self.para["size"]-1] == self.find_max()['value']:
            # a[0] = self.para["size"] - 1
            # a[1] = self.para["size"] - 1
        # return a
    # #Check possiblity of moving greatest entry to the corner
    # def _2048_naivesolver(self):
        # cornerscore = {'w':0,'a':0,'s':0,'d':0}
        # spacescore = {'w':0,'a':0,'s':0,'d':0}
#
        # tempgame = list()
#
        # #print "got herer!!!!!!!!!!!!!!"
        # for i in range(0,self.para["size"]):
            # tempgame.append(self[i][:])
        # #print tempgame
        # self.move_up()
        # #print "get here naivesolver!!!1"
        # if tempgame[:] != self[:]:
            # if self.greatestcorner()[1] != -1: cornerscore['w'] = 1
            # spacescore['w'] = self.countempty()
        # else: spacescore['w'] = 0
        # for i in range(0,self.para["size"]):
            # for j in range(0,self.para["size"]):
                 # self[i][j] = tempgame[i][j]
#
        # self.move_left()
        # if tempgame[:] != self[:]:
            # if self.greatestcorner()[1] != -1: cornerscore['a'] = 1
            # spacescore['a'] = self.countempty()
        # else: spacescore['a'] = 0
        # for i in range(0,self.para["size"]):
            # for j in range(0,self.para["size"]):
                 # self[i][j] = tempgame[i][j]
#
#
        # self.move_down()
        # if tempgame[:] != self[:]:
            # if self.greatestcorner()[1] != -1: cornerscore['s'] = 1
            # spacescore['s'] = self.countempty()
        # else: spacescore['s'] = 0
        # for i in range(0,self.para["size"]):
            # for j in range(0,self.para["size"]):
                 # self[i][j] = tempgame[i][j]
#
#
        # self.move_right()
        # if tempgame[:] != self[:]:
            # if self.greatestcorner()[1] != -1: cornerscore['d'] = 1
            # spacescore['d'] = self.countempty()
        # else: spacescore['d'] = 0
        # for i in range(0,self.para["size"]):
            # for j in range(0,self.para["size"]):
                 # self[i][j] = tempgame[i][j]
        # #print tempgame
        # #print "= = = = ="
        # #self.printout()
#
        # movelist = list()
        # a = -1
        # for i in ['w','a','s','d']:
            # b = cornerscore[i]*self.para["size"]*self.para["size"]+spacescore[i]
            # if b > a:
                # movelist = [i]
                # a = b
            # elif b==a:
                # movelist.append(i)
        # if len(movelist) > 1:
            # move = random.choice(movelist)
        # else:
            # move = movelist[0]
    # #variable control
        # #print cornerscore
        # #print spacescore
        # return move

    #########plaey a whole game and record the process ######

    def gameplay(self,choicefunc):
        #print "here i am "
        onegame = list()
        game = Game()
        game.allzero()
        game.next()
        game.next()
        tag = 0
        while tag == 0:
            onestep = game[0][:]
            for i in range(1,self.para["size"]):
                onestep = onestep + self[i][:]
            move = choicefunc()
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

    def _2048_cnnsolver(self):
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
            print("Output to ",para_dict["data_foutputfile"]," failed.")

    #play multiple games but only collect the final step data
    def play(self,para_dict):
        for i in range(int(para_dict["numofplay"])):
            onegame, gamefinal = self.gameplay(self.method_dict[para_dict["method"]])
            self.finaldata.append(gamefinal)
        try:
            self.finaldata_foutput(para_dict["outputfile"])
        except:
            print("Output to ",para_dict["outputfile"]," failed.")

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
            print("Output to ",para_dict["outputfile"]," failed.")

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
            print("illegal input!")
        if tempgame[:] != self[:]:
            for i in range(0,self.para["size"]):
                for j in range(0,self.para["size"]):
                     self[i][j] = tempgame[i][j]
            return True
        else: return False

    def push(self):
        pass

    def pop(self):
        pass

    def new_game_by_ai(self,ai_player):
        pass

    def save(self,game_name):
        pass

    def load(self):
        # self._board = ...
        self._play()

    def replay(self,game_name):
        pass

#The following is the main program.

if __name__ == '__main__':
    game_play = Game_play(1)
    #game_play.naiveplay()
    #game_play.data_foutput("output.txt")
