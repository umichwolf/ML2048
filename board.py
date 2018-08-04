import random
import time

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
        for i in range(self._para['size']):
            for j in range(self._para['size']):
                self[i][j] = board[i][j]

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
        random.seed(time.time())
        temp = random.choice(self._zero_entries_list)
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

    def move(self, action, quiet=0):
        temp = [0] * self._para['size']
        for i in range(self._para['size']):
            temp[i] = self[i].copy()
        if action == 'w':
            self._move_up()
        if action == 's':
            self._move_down()
        if action == 'a':
            self._move_left()
        if action == 'd':
            self._move_right()
        if temp[:] == self[:]:
            if quiet == 0:
                print('illegal move!')
            return 0
        return 1
