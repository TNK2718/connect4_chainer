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


PLAYER1 = 1
PLAYER2 = 2
NONE = 0    
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


# 学習用のBoard
class Board():
    def __init__(self):
        self.board = np.zeros((SIZE, SIZE), dtype=np.float32)
        self.winner = NONE  # 勝者
        self.turn = PLAYER1
        self.sub_turn = PLAYER2
        self.game_end = False  # ゲーム終了チェックフラグ
        self.available_pos = self.get_possible_moves()

    def board_reset(self):
        self.board = np.zeros((SIZE, SIZE), dtype=np.float32)
        self.winner = NONE  # 勝者
        self.turn = PLAYER1
        self.sub_turn = PLAYER2
        self.game_end = False  # ゲーム終了チェックフラグ
        self.available_pos = self.get_possible_moves()  # self.turnの石が置ける場所のリスト

    def random_action(self):
        moves = self.get_possible_moves()
        if len(moves) > 0:
            pos = random.choice(moves)  # 置く場所をランダムに決める
            return pos
        return False  # 置く場所なし

    def put_stone(self, x, player):
        count = 0
        while(self.get_board(x, count) == PLAYER1 or self.get_board(x, count) == PLAYER2):
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
            if(self.get_board(i, SIZE - 1) == 0):
                moves.append(i)
        return moves

    def get_available_moves(self):
        moves = []
        for i in range(0, SIZE):
            if(self.is_available_move(i)):
                moves.append(i)
        return moves

    def is_available_move(self, x):
        if(self.get_board(x, SIZE - 1) != NONE):
            return False
        self.put_stone(x, self.turn)
        if(self.lethal_move(self.turn) != -1):
            self.remove_stone(x)
            return False
        self.remove_stone(x)
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

    # そこに置かなければ負ける手
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
                print(int(self.board[x, SIZE - 1 - y]), end = "")
            print('') 
        print('\r\n')

    def get_board(self, x, y):
        if(0 <= x < SIZE and 0 <= y < SIZE):
            return self.board[x, y]
        else:
            return -1
        

def main():
    #chainer.serializers.save_npz('result/out.model', model)

    board = Board()  # ボード初期化

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
        start_epsilon=1.0, end_epsilon=0.15, decay_steps=100000, random_action_func=board.random_action)
    # 経験再生
    replay_buffer_b = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    replay_buffer_w = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    # 白黒２種のエージェント
    agent_black = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer_b, gamma, explorer,
        replay_start_size=1000, minibatch_size=128, update_interval=1, target_update_interval=1000)
    agent_white = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer_w, gamma, explorer,
        replay_start_size=1000, minibatch_size=128, update_interval=1, target_update_interval=1000)
    agents = ['', agent_black, agent_white]

    print('episodes: ')
    n_episodes = int(input())  # 最大のゲーム回数
    win = 0  # 黒の勝利数
    lose = 0  # 黒の敗北数
    draw = 0  # 引き分け

    for i in range(1, n_episodes + 1):
        board.board_reset()
        #print('DEBUG epi {} {}'.format(i+1, len(replay_buffer_b.memory)))
        rewards = [0, 0, 0]  # 報酬リセット

        while not board.game_end:  # ゲームが終わるまで繰り返す
            board.available_pos = board.get_available_moves()
            #print(board.available_pos)
            #board.draw_board()
            if(board.get_possible_moves().__len__() == 0):
                board.game_end = True
            elif(board.available_pos.__len__() == 0):
                board.game_end = True
                board.winner = board.sub_turn
            else:
                boardcopy = np.reshape(board.board.copy(), (1,SIZE,SIZE))
                while True: # 置ける場所が見つかるまで繰り返す
                    pos = agents[board.turn].act_and_train(boardcopy, rewards[board.turn])
                    if pos in board.available_pos:                        
                        break
                    else:
                        rewards[board.turn] = REWARD_LOSE # 石が置けない場所であれば負の報酬                                        
                # 石を配置
                board.agent_action(pos)
                # 勝敗判定
                board.winner = board.is_end()
                if(board.winner == NONE):
                    lethal = board.lethal_move(board.turn)
                    if(lethal != -1):
                        boardcopy = np.reshape(board.board.copy(), (1,SIZE,SIZE))
                        board.winner = board.sub_turn
                        count = 0
                        while True:
                            if(pos == lethal):
                                board.change_turn()
                                rewards[board.sub_turn] = REWARD_WIN
                                for i in range(0, count):
                                    agents[board.sub_turn].act_and_train(boardcopy, rewards[board.sub_turn])
                                    board.remove_stone(pos)
                                board.agent_action(pos)
                                break
                            else:
                                board.put_stone(pos, board.sub_turn)
                                boardcopy = np.reshape(board.board.copy(), (1,SIZE,SIZE))
                                rewards[board.sub_turn] = REWARD_LOSE # 勝てない場所であれば負の報酬
                                pos = agents[board.sub_turn].act_and_train(boardcopy, rewards[board.sub_turn])
                                board.remove_stone(pos)
                                count += 1

                if(board.winner != NONE):
                    board.game_end = True
                    
            # ゲーム時の処理
            if board.game_end:
                if board.winner == PLAYER1:
                    rewards[PLAYER1] = REWARD_WIN  # 黒の勝ち報酬
                    rewards[PLAYER2] = REWARD_LOSE  # 白の負け報酬
                    win += 1
                elif board.winner == 0:
                    draw += 1
                else:
                    rewards[PLAYER1] = REWARD_LOSE
                    rewards[PLAYER2] = REWARD_WIN
                    lose += 1
                # エピソードを終了して学習
                boardcopy = np.reshape(board.board.copy(), (1, SIZE, SIZE))
                # 勝者のエージェントの学習
                agents[board.turn].stop_episode_and_train(
                    boardcopy, rewards[board.turn], True)
                board.change_turn()
                # 敗者のエージェントの学習
                agents[board.turn].stop_episode_and_train(
                    boardcopy, rewards[board.turn], True)
            else:
                board.change_turn()

        # 学習の進捗表示
        if i % 100 == 0:
            print('==== Episode {} : black win {}, black lose {}, draw {} ===='.format(
                i, win, lose, draw))  # 勝敗数は黒石基準
            print('<BLACK> statistics: {}, epsilon {}'.format(
                agent_black.get_statistics(), agent_black.explorer.epsilon))
            print('<WHITE> statistics: {}, epsilon {}'.format(
                agent_white.get_statistics(), agent_white.explorer.epsilon))
            # カウンタ変数の初期化
            win = 0
            lose = 0
            draw = 0

        if i % 1000 == 0:  # 1000エピソードごとにモデルを保存する
            agent_black.save("agent_black_" + str(i))
            agent_white.save("agent_white_" + str(i))


if __name__ == '__main__':
    main()
