from __future__ import print_function
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import sys
import re  # 正規表現
import random
import copy


PLAYER1 = 1.0
PLAYER2 = 2.0
NONE = 0.0
SIZE = 7   # ボードサイズ SIZE*SIZE
REWARD_WIN = 1  # 勝ったときの報酬
REWARD_LOSE = -1  # 負けたときの報酬
# 2次元のボード上での隣接8方向の定義（左から，上，右上，右，右下，下，左下，左，左上）
DIR = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))

'''
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_nodes):
        w = chainer.initializers.HeNormal(scale=1.0)  # 重みの初期化
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(49, n_nodes, initialW=w)
            self.l2 = L.Linear(n_nodes, n_nodes, initialW=w)
            self.l3 = L.Linear(n_nodes, n_nodes, initialW=w)
            self.l4 = L.Linear(n_nodes, n_actions, initialW=w)

    # フォワード処理
    def __call__(self, x):
        #print('DEBUG: forward {}'.format(x))
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        return chainerrl.action_value.DiscreteActionValue(self.l4(h))
'''

class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_nodes):
        w = chainer.initializers.HeNormal(scale=1.0) # 重みの初期化
        super(QFunction, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(1, 4, 2, 1, 0)
            self.c2 = L.Convolution2D(4, 8, 2, 1, 0)
            self.c3 = L.Convolution2D(8, 16, 2, 1, 0)
            self.l4 = L.Linear(256, n_nodes, initialW=w)
            self.l5 = L.Linear(n_nodes, n_actions, initialW=w)

    # フォワード処理
    def __call__(self, x):
        #print('DEBUG: forward {}'.format(x))
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        h = F.relu(self.l4(h))
        return chainerrl.action_value.DiscreteActionValue(self.l5(h))


class Board():
    def __init__(self):
        self.board = np.zeros((SIZE, SIZE), dtype=np.float32)
        self.winner = NONE  # 勝者
        self.turn = PLAYER1
        self.sub_turn = PLAYER2
        self.game_end = False  # ゲーム終了チェックフラグ
        self.available_pos = self.get_possible_moves()  # self.turnの石が置ける場所のリスト

    def board_reset(self):
        self.board = np.zeros((SIZE, SIZE), dtype=np.float32)
        self.winner = NONE  # 勝者
        self.turn = PLAYER1
        self.sub_turn = PLAYER2
        self.game_end = False  # ゲーム終了チェックフラグ
        self.available_pos = self.get_possible_moves()  # self.turnの石が置ける場所のリスト

    def random_action(self):
        if len(self.available_pos) > 0:
            pos = random.choice(self.available_pos)  # 置く場所をランダムに決める
            return pos
        return False  # 置く場所なし

    def put_stone(self, x, player):
        count = 0
        while(self.get_board(x, count) > 0):
            count += 1
            if(count >= SIZE):
                return
        self.board[x, count] = float(player)

    def remove_stone(self, x):
        count = 0
        while(self.get_board(x, SIZE - 1 - count) == NONE):
            count += 1
            if(count >= SIZE):
                return
        self.board[x, SIZE - 1 - count] = 0.0

    def agent_action(self, x):
        lethal = self.lethal_move(self.turn)
        if(lethal >= 0):
            self.put_stone(lethal, self.turn)
            return
        self.put_stone(x, self.turn)

    def get_possible_moves(self):
        moves = []
        for i in range(0, SIZE):
            if(self.board[i, SIZE - 1] == 0):
                moves.append(i)
        return moves

    def is_possible_move(self, x):
        if(self.get_board(x, SIZE - 1) == PLAYER1):
            return False
        if(self.get_board(x, SIZE - 1) == PLAYER2):
            return False
        else:
            return True
    
    def is_end(self):
        for i in range(SIZE):
            for j in range(SIZE):
                for dir in DIR:
                    if(self.count_seq_stones(i, j, dir, PLAYER1) >= 4):
                        return PLAYER1
                    if(self.count_seq_stones(i, j, dir, PLAYER2) >= 4):
                        return PLAYER2
        return NONE

    def lethal_move(self, player):
        moves = self.get_possible_moves()
        for i in moves:
            self.put_stone(i, self.sub_turn)
            if(self.is_end() == self.sub_turn):
                self.remove_stone(i)
                return i
            self.remove_stone(i)
        return -1

    def count_seq_stones(self, x, y, dir, player):
        count = 0
        while(self.get_board(x + count * dir[0], y + count * dir[1]) == player):
            count += 1
        return count

    def change_turn(self):
        self.turn = PLAYER2 if self.turn == PLAYER1 else PLAYER1
        self.sub_turn = PLAYER2 if self.turn == PLAYER1 else PLAYER1
        self.available_pos = self.get_possible_moves()

    def draw_board(self):
        for y in range(SIZE):
            for x in range(SIZE):
                print(int(self.board[x, SIZE - 1 - y]), end = " ")
            print('') 
        print('\r\n')

    def get_board(self, x, y):
        if(0 <= x < SIZE and 0 <= y < SIZE):
            return self.board[x, y]
        else:
            return -1
        
def main():
    board = Board()


    # モデルの読み込み
    obs_size = SIZE * SIZE  # ボードサイズ（=NN入力次元数）
    n_actions = 7
    n_nodes = 226  # 中間層のノード数
    q_func = QFunction(obs_size, n_actions, n_nodes)

    # optimizerの設定
    optimizer = chainer.optimizers.Adam(eps=1e-3)
    optimizer.setup(q_func)
    # 減衰率
    gamma = 0.995
    # ε-greedy法
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.1, decay_steps=100000, random_action_func=board.random_action)
    # Expericence Replay用のバッファ（十分大きく，エージェント毎に用意）
    replay_buffer_b = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    agent_black = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer_b, gamma, explorer,
        replay_start_size=1000, minibatch_size=128, update_interval=1, target_update_interval=1000)
    agent_white = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer_b, gamma, explorer,
        replay_start_size=1000, minibatch_size=128, update_interval=1, target_update_interval=1000)
    print('level:')
    level = input()
    print('first: 0, second: 1, AIvsAI: 2')
    turn = int(input())
    if(turn == 1):
        agent_black.load('agent_black_' + level)
        while(board.game_end == False):
            print(board.turn)
            boardcopy = np.reshape(board.board.copy(), (1,SIZE,SIZE))
            pos = agent_black.act(boardcopy)
            if(board.is_possible_move(pos)):
                board.agent_action(pos)
            else:
                board.agent_action(board.random_action())             
            board.draw_board()
            board.winner = board.is_end()
            if(board.winner != NONE):
                break
            if(board.winner == NONE):
                lethal = board.lethal_move(board.turn)
                if(lethal != -1):
                    board.winner = board.sub_turn
                    break
            board.change_turn()
            print(board.turn)
            board.agent_action(int(input()))
            board.draw_board()
            board.winner = board.is_end()
            if(board.winner != NONE):
                break
            if(board.winner == NONE):
                lethal = board.lethal_move(board.turn)
                if(lethal != -1):
                    board.winner = board.sub_turn
                    break
            board.change_turn()
    elif(turn == 0):
        agent_white.load('agent_white_' + level)
        board.draw_board()
        while(board.game_end == False):
            print(board.turn)
            board.agent_action(int(input()))
            board.draw_board()
            board.winner = board.is_end()
            if(board.winner != NONE):
                break
            if(board.winner == NONE):
                lethal = board.lethal_move(board.turn)
                if(lethal != -1):
                    board.winner = board.sub_turn
                    break
            board.change_turn()
            print(board.turn)
            boardcopy = np.reshape(board.board.copy(), (1,SIZE,SIZE))
            pos = agent_white.act(boardcopy)
            if(board.is_possible_move(pos)):
                board.agent_action(pos)
            else:
                board.agent_action(board.random_action())                              
            board.draw_board()
            board.winner = board.is_end()
            if(board.winner != NONE):
                break
            if(board.winner == NONE):
                lethal = board.lethal_move(board.turn)
                if(lethal != -1):
                    board.winner = board.sub_turn
                    break
            board.change_turn()
    else:
        agent_black.load('agent_black_' + level)
        agent_white.load('agent_white_' + level)
        while(board.game_end == False):
            print(board.turn)
            boardcopy = np.reshape(board.board.copy(), (1,SIZE,SIZE))
            pos = agent_white.act(boardcopy)
            if(board.is_possible_move(pos)):
                board.agent_action(pos)
            else:
                board.agent_action(board.random_action())                              
            board.draw_board()
            board.winner = board.is_end()
            if(board.winner != NONE):
                break
            if(board.winner == NONE):
                lethal = board.lethal_move(board.turn)
                if(lethal != -1):
                    board.winner = board.sub_turn
                    break
            board.change_turn()
    print('winner:' + str(board.winner))
    input()

if __name__ == "__main__":
    main()