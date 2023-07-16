# MILESTONE PROJECT 1 - TIC TAC TOE ########################################################

class TicTacToe:

    def __init__(self):
        self.positions = {''.join(pos): ' ' for pos in zip('123', 'abc')}
        self.players = []
        self.count = 0
        self.victor = ''

    def play_game(self):

        print('\n----------- Tic Tac Toe -------------\n')

        one_pick = str(input('Player 1, Noughts or Crosses?'))
        lower_one_pick = one_pick.lower()
        if lower_one_pick in {'0', 'noughts'}:
            self.players = [(0, '0'), (1, 'X')]
        elif lower_one_pick in {'x', 'crosses'}:
            self.players = [(0, 'X'), (1, '0')]

        while not self.victory() and self.count < 9:
            self.rounds()

        if self.victor:
            print(f'Congratulations Player {self.find_victor()}, you have won!')
        else:
            print('Well done guys, what a boring game...')

    def rounds(self):
        while True:
            p = input(f'\nPlayer {self.players[0][0] + 1}, enter your desired position:')
            if p == 'q':
                quit()
            if p not in self.positions.keys():
                print('\nInvalid position, try again dood')
            elif self.positions[p] in {'0', 'X'}:
                print('\nPosition already filled, try again')
            else:
                break
        self.positions[p] = self.players[0][1]

        self.display_board()

        self.players.reverse()
        self.count += 1

    def victory(self):
        if self.count < 5:
            return False
        spaces_list = list(self.positions.values())
        for i, space in enumerate(spaces_list):
            if space and space != ' ':
                if i == 0:
                    if space == spaces_list[i + 1] == spaces_list[i + 2]:
                        self.victor = space
                        return True
                    elif space == spaces_list[i + 3] == spaces_list[i + 6]:
                        self.victor = space
                        return True
                    elif space == spaces_list[i + 4] == spaces_list[i + 8]:
                        self.victor = space
                        return True

                elif i == 1:
                    if space == spaces_list[i + 3] == spaces_list[i + 6]:
                        self.victor = space
                        return True

                elif i == 2:
                    if space == spaces_list[i + 2] == spaces_list[i + 4]:
                        self.victor = space
                        return True
                    elif space == spaces_list[i + 3] == spaces_list[i + 6]:
                        self.victor = space
                        return True          

                elif i == 3:
                    if space == spaces_list[i + 1] == spaces_list[i + 2]:
                        self.victor = space
                        return True          

                elif i == 6:
                    if space == spaces_list[i + 1] == spaces_list[i + 2]:
                        self.victor = space
                        return True
        return False

    def find_victor(self):
        for player in self.players:
            if self.victor == player[1]:
                return player[0] + 1

    def display_board(self):
        pos = self.positions
        print(f"\n\n        3     {pos['a3']}    |    {pos['b3']}    |    {pos['c3']}\
                  \n          -----------------------------\
                  \n        2     {pos['a2']}    |    {pos['b2']}    |    {pos['c2']}\
                  \n          -----------------------------\
                  \n        1     {pos['a1']}    |    {pos['b1']}    |    {pos['c1']}\
                \n\n              a         b         c          \n")


tictac = TicTacToe()
tictac.play_game()