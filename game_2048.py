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

    def load_board(self,board):
        self[:] = board

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
        while endgame_flag == 0:
            self._board.print_board()
            order = input('Your Move(wsad,e:exit): ')
            while order != 'e' and self._board.move(order) == 0:
                order = input('Your Move(wsad,e:exit): ')
                if order == 'e':
                    break
            if order == 'e':
                break
            self.push(order)
            endgame_flag = self._board.gameend()

        if endgame_flag == 1:
            self._board.print_board()
            print('Game over!')
        if input('Do you want to save the game?(y/n) ') == 'y':
            self.save(input('Name of the game: '))
        self.idle()

    def save(self,filename):
        with open(self.pow+filename,'w') as f:
            for line in self._game:
                f.write(str(line))
                f.write('\n')
        with open(self.pow+filename+'.para','w') as f:
            f.write(str(self._board.para))

    def _load_game(self,filename):
        with open(self.pow+filename,'r') as f:
            for line in f:
                self._game.append(eval(line))
        with open(self.pow+filename+'.para','r') as f:
            self._board = Board(eval(f.read()))

    def load(self,filename):
        self._load_game(filename)
        move = self.pop()
        self._board.move(temp_move)
        self._play()

    def push(self,move):
        for row in self._board:
            self._game.append(row)
        self._game.append(move)

    def pop(self,step=-1):
        temp_board = list()
        for idx in range(self._board.para['size']):
            temp_board.append(self._game[step][-2-idx*n:-2-(idx+1)*n])
        self._board.load_board(temp_board)
        return = self._game[step][-1]

    def new_game_by_ai(self,ai_player):
        self._board = Board(parameter)
        endgame_flag = self._board.gameend()
        while endgame_flag == 0:
            self._board.print_board()
            if input('next?(y/n) ') == 'y':
                order = ai_player.move(self._board)
                if self._board.move(order) == 0:
                    print('AI error.')
                    break
                self.push(order)
                self._board.next()
            else:
                break
            endgame_flag = self._board.gameend()
        if endgame_flag == 1:
            self._board.print_board()
            print('Game over!')
        if input('Do you want to save the game?(y/n) ') == 'y':
            self.save(input('Name of the game: '))
        self.idle()


    def replay(self,filename):
        self._load_game(filename)
        for idx in range(len(self._game)):
            move = self.pop(idx)
            self._board.print_board()
            if input('next?(y/n) ') == 'y':
                self._board.move(move)
            else:
                break
        if idx == len(self._game)-1:
            print('Game over!')
        else:
            pass
        self.idle()

#The following is the main program.

if __name__ == '__main__':
    game = Game()
