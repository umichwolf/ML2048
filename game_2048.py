#This is the game "2048".

import copy
from board import Board
from ai_player import Ai

class Game:
    """
    this class provides interfaces of playing games by
    human players or ai players, recording games, saving
    games and replaying games.
    """
    def __init__(self):
        self.pow = './game_archives/'
        self.idle()

    def idle(self):
        self._game = list()
        order = input(
"""Choose one from the list:
    1: new game
    2: new game by ai
    3: load
    4: replay
    5: exit\n""")
        if order == '1':
            parameter = {'size': eval(input('size: ')),
                'odd_2': eval(input('odd of 2(between 0 and 1): '))}
            self.new_game(parameter)
        if order == '2':
            parameter = {'size': eval(input('size: ')),
                'odd_2': eval(input('odd of 2(between 0 and 1): '))}
            name = input('Name of the ai player: ')
            self.new_game_by_ai(name,parameter)
        if order == '3':
            self.load(input('Name of the game: '))
        if order == '4':
            self.replay(input('Name of the game: '))
        if order == '5':
            pass

    def new_game(self,parameter):
        self._board = Board(parameter)
        self._play()

    def _play(self):
        endgame_flag = self._board.gameend()
        while endgame_flag == 0:
            self._board.print_board()
            self.push()
            order = input('Your Move(wsad,e:exit): ')
            while order != 'e' and self._board.move(order) == 0:
                order = input('Your Move(wsad,e:exit): ')
                if order == 'e':
                    break
            if order == 'e':
                break
            self.push(order)
            self._board.next()
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
        self.pop()
        self._play()

    def push(self,move=None):
        temp = []
        if move==None:
            for row in self._board:
                temp.extend(row)
            self._game.append(temp)
        else:
            self._game[-1].append(move)

    def pop(self):
        temp_board = list()
        n = self._board.para['size']
        temp_row = self._game.pop()
        for idx in range(n):
            temp_board.append(temp_row[idx*n:(idx+1)*n])
        self._board.load_board(temp_board)

    def read(self,idx):
        temp_board = list()
        n = self._board.para['size']
        temp_row = self._game[idx]
        for jdx in range(n):
            temp_board.append(temp_row[jdx*n:(jdx+1)*n])
        self._board.load_board(temp_board)
        if len(temp_row) == n*n+1:
            return self._game[idx][-1]
        else:
            return None

    def new_game_by_ai(self,name,parameter):
        self._board = Board(parameter)
        ai_player = Ai(name,parameter)
        endgame_flag = self._board.gameend()
        while endgame_flag == 0:
            self._board.print_board()
            self.push()
            if input('next?(y/n) ') == 'y':
                board = copy.deepcopy(self._board)
                order = ai_player.move(board)
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
            move = self.read(idx)
            self._board.print_board()
            if move == None:
                print('Game over!')
                break
            print(move)
            if input('next?(y/n) ') == 'y':
                self._board.move(move)
            else:
                break
        self.idle()

#The following is the main program.

if __name__ == '__main__':
    game = Game()
