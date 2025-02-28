import random
import numpy as np
import psutil
import time
import os
import pynvml
import tensorflow as tf
from tensorflow import keras

###############################################################################
# 1) GPU割り当て設定 (省略可)
###############################################################################
def select_and_configure_gpu(gpu_index=0):
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("GPUが検出されませんでした。CPUのみで実行します。")
        return
    if gpu_index < 0 or gpu_index >= len(gpus):
        print(f"指定したgpu_index={gpu_index}は無効です。CPUのみで実行します。")
        return

    try:
        tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        print(f"GPUデバイス index={gpu_index} のみが使用されます。")
    except RuntimeError as e:
        print(e)

def print_gpu_memory_usage(gpu_index=0):
    """GPUメモリ使用状況をpynvmlで表示"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_mb = mem_info.total / 1024**2
    used_mb  = mem_info.used  / 1024**2
    free_mb  = mem_info.free  / 1024**2
    print(f"[GPU index={gpu_index} Memory]:")
    print(f"  Total: {total_mb:.2f} MB")
    print(f"  Used : {used_mb:.2f} MB")
    print(f"  Free : {free_mb:.2f} MB\n")
    pynvml.nvmlShutdown()

# 必要に応じて
select_and_configure_gpu(gpu_index=0)

###############################################################################
# 2) グローバル変数 (推論時間 & メモリ使用量)
###############################################################################
model_inference_times = []
model_memory_usages   = []

mc_inference_times    = []
mc_memory_usages      = []

process = psutil.Process(os.getpid())

###############################################################################
# 3) オセロ盤処理
###############################################################################
def init_board():
    bd = np.zeros((8,8),dtype=int)
    # 黒=+1, 白=-1
    bd[3,3] = bd[4,4] = -1  # 白
    bd[3,4] = bd[4,3] =  1  # 黒
    return bd

def print_board(board):
    symbols = {1:"●", -1:"○", 0:"."}
    for row in board:
        print(" ".join(symbols[v] for v in row))
    print()

def can_put(board, row, col, is_black_turn):
    if board[row,col]!=0:
        return False
    current_player= 1 if is_black_turn else -1
    opponent= -current_player
    directions= [
        (-1,0),(1,0),(0,-1),(0,1),
        (-1,-1),(1,-1),(-1,1),(1,1)
    ]
    for dx,dy in directions:
        x,y= row+dx, col+dy
        found_opp= False
        while 0<=x<8 and 0<=y<8:
            if board[x,y]==opponent:
                found_opp= True
            elif board[x,y]== current_player:
                if found_opp:
                    return True
                break
            else:
                break
            x+= dx
            y+= dy
    return False

def put(board, row, col, is_black_turn):
    player= 1 if is_black_turn else -1
    board[row,col]= player
    directions= [
        (-1,0),(1,0),(0,-1),(0,1),
        (-1,-1),(1,-1),(-1,1),(1,1)
    ]
    for dx,dy in directions:
        x,y= row+dx,col+dy
        stones_to_flip=[]
        while 0<=x<8 and 0<=y<8 and board[x,y]== -player:
            stones_to_flip.append((x,y))
            x+= dx
            y+= dy
        if 0<=x<8 and 0<=y<8 and board[x,y]== player:
            for (fx,fy) in stones_to_flip:
                board[fx,fy]= player

def count_stones(board):
    black= np.sum(board==1)
    white= np.sum(board== -1)
    return black,white

###############################################################################
# 4) モデルAI (Keras)
###############################################################################
class ModelAI:
    def __init__(self, model, label="unknown", my_color=1):
        """
        model: kerasモデル
        label: (small/medium/large等)
        my_color: +1(黒) or -1(白)
        """
        self.model= model
        self.label= label
        self.my_color= my_color

    def choose_move(self, board, is_black_turn):
        # 自分の番かどうか
        current_player= 1 if is_black_turn else -1
        if current_player!= self.my_color:
            return None

        valid_moves= [
            (r,c) for r in range(8) for c in range(8)
            if can_put(board, r,c, is_black_turn)
        ]
        if not valid_moves:
            return None

        board_input= np.zeros((1,2,8,8), dtype='int8')
        if self.my_color== 1:
            # 黒番
            board_input[0,0]= (board==1).astype('int8')
            board_input[0,1]= (board==-1).astype('int8')
        else:
            # 白番
            board_input[0,0]= (board==-1).astype('int8')
            board_input[0,1]= (board==1).astype('int8')

        before_mem= process.memory_info().rss
        start_time= time.time()

        preds= self.model.predict(board_input, verbose=0)[0]  # shape=(64,)

        end_time= time.time()
        after_mem= process.memory_info().rss
        elapsed= end_time- start_time
        mem_used= after_mem- before_mem

        global model_inference_times, model_memory_usages
        model_inference_times.append(elapsed)
        model_memory_usages.append(mem_used)

        color_name= "黒" if self.my_color==1 else "白"
        print(f"[Model({self.label})AI({color_name})] "
              f"MemUsed={mem_used}, Time={elapsed:.6f}s")

        move_probs= [ preds[r*8+c] for (r,c) in valid_moves ]
        sorted_probs= sorted(list(set(move_probs)), reverse=True)
        if len(sorted_probs)>=2:
            best= sorted_probs[0]
            secd= sorted_probs[1]
        else:
            best= sorted_probs[0]
            secd= None

        top1= [(r,c) for (r,c) in valid_moves
               if np.isclose(preds[r*8+c], best)]
        top2= []
        if secd is not None:
            top2= [(r,c) for (r,c) in valid_moves
                   if np.isclose(preds[r*8+c], secd)]

        p= random.random()
        if top2 and p<0.1:
            return random.choice(top2)
        else:
            return random.choice(top1)

###############################################################################
# 5) MinimaxAI
###############################################################################
class MinimaxAI:
    def __init__(self, depth=5, my_color=1):
        self.depth= depth
        self.my_color= my_color

        self.weights= np.array([
            [100, -20,  10,   5,   5,  10, -20, 100],
            [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
            [ 10,  -2,  -1,  -1,  -1,  -1,  -2,  10],
            [  5,  -2,  -1,  -1,  -1,  -1,  -2,   5],
            [  5,  -2,  -1,  -1,  -1,  -1,  -2,   5],
            [ 10,  -2,  -1,  -1,  -1,  -1,  -2,  10],
            [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
            [100, -20,  10,   5,   5,  10, -20, 100]
        ],dtype=float)

    @property
    def label(self):
        return f"Minimax(depth={self.depth}, color={self.my_color})"

    def choose_move(self, board, is_black_turn):
        current_player= 1 if is_black_turn else -1
        if current_player!= self.my_color:
            return None

        valid_moves= [
            (r,c) for r in range(8) for c in range(8)
            if can_put(board, r,c, is_black_turn)
        ]
        if not valid_moves:
            return None

        best_val= float('-inf')
        best_move= None
        for mv in valid_moves:
            new_bd= board.copy()
            put(new_bd, mv[0], mv[1], is_black_turn)
            val= self.minimax(new_bd, self.depth-1, not is_black_turn)
            if val> best_val:
                best_val= val
                best_move= mv

        print(f"[MinimaxAI(depth={self.depth}, color={self.my_color})] best_eval={best_val:.2f}")
        return best_move

    def minimax(self, board, depth, is_black_turn):
        if depth==0 or np.all(board!=0):
            return self.evaluate_position(board)

        current_player= 1 if is_black_turn else -1
        valid_moves= [
            (r,c) for r in range(8) for c in range(8)
            if can_put(board, r,c, is_black_turn)
        ]
        if not valid_moves:
            opp= not is_black_turn
            opp_moves= [
                (r,c) for r in range(8) for c in range(8)
                if can_put(board, r,c, opp)
            ]
            if not opp_moves:
                return self.evaluate_position(board)
            else:
                return self.minimax(board, depth, opp)

        if current_player== self.my_color:
            best= float('-inf')
            for mv in valid_moves:
                bd2= board.copy()
                put(bd2, mv[0], mv[1], is_black_turn)
                val= self.minimax(bd2, depth-1, not is_black_turn)
                best= max(best,val)
            return best
        else:
            worst= float('inf')
            for mv in valid_moves:
                bd2= board.copy()
                put(bd2, mv[0], mv[1], is_black_turn)
                val= self.minimax(bd2, depth-1, not is_black_turn)
                worst= min(worst,val)
            return worst

    def evaluate_position(self, board):
        me= self.my_color
        opp= -me
        score_me= np.sum((board==me)* self.weights)
        score_op= np.sum((board==opp)* self.weights)
        return score_me- score_op

###############################################################################
# 6) HybridWhiteAI (白)
###############################################################################
class HybridWhiteAI:
    def __init__(self, model_ai, minimax_ai, threshold=15):
        """
        model_ai: ModelAIなど (my_color=-1)
        minimax_ai: MinimaxAI (my_color=-1)
        threshold: 残り空きマスがこの数以下でMinimaxに切替
        """
        self.model_ai  = model_ai
        self.minimax_ai= minimax_ai
        self.threshold = threshold
        self.my_color  = -1

    def choose_move(self, board, is_black_turn):
        # 白番でなければパス
        current_player= -1
        if is_black_turn:
            return None

        empty_count= np.count_nonzero(board==0)
        if empty_count<= self.threshold:
            print(f"[HybridWhiteAI] 空き={empty_count} <= {self.threshold} => Minimax")
            return self.minimax_ai.choose_move(board, False)
        else:
            print(f"[HybridWhiteAI] 空き={empty_count} => モデルAI(白)")
            return self.model_ai.choose_move(board, False)

###############################################################################
# 7) MonteCarloAI (黒)
###############################################################################
class MonteCarloAIBlack:
    """
    黒番用のモンテカルロAI: (黒目線で result=1(黒勝))
    """
    def __init__(self, num_simulations=50):
        self.num_simulations= num_simulations
        self.my_color= 1

    def simulate_game(self, board, is_black_turn):
        """1回のランダムプレイアウト (黒>白=1, 白>黒=-1, 同数=0)"""
        sim_bd= board.copy()
        turn= is_black_turn
        while True:
            valid_moves= [
                (r,c) for r in range(8) for c in range(8)
                if can_put(sim_bd, r,c, turn)
            ]
            if valid_moves:
                mv= random.choice(valid_moves)
                put(sim_bd, mv[0], mv[1], turn)
            else:
                # パス後 相手も打てなければ終了
                if not any(can_put(sim_bd, rr,cc, not turn)
                           for rr in range(8) for cc in range(8)):
                    break
            turn= not turn

        black,white= count_stones(sim_bd)
        if black>white:
            return 1
        elif white>black:
            return -1
        else:
            return 0

    def choose_move(self, board, is_black_turn):
        # 自分の番か?
        if not is_black_turn:
            return None

        valid_moves= [
            (r,c) for r in range(8) for c in range(8)
            if can_put(board, r,c, True)
        ]
        if not valid_moves:
            return None

        before_mem= process.memory_info().rss
        start_time= time.time()

        move_scores= {}
        for mv in valid_moves:
            # mvを打った局面から self.num_simulations 回ランダムプレイアウト
            sim_bd= board.copy()
            put(sim_bd, mv[0], mv[1], True)
            score= 0
            for _ in range(self.num_simulations):
                bd_copy= sim_bd.copy()
                result= self.simulate_game(bd_copy, False)  # 次は白
                # 黒視点 result=1(黒勝)
                score+= result
            move_scores[mv]= score

        end_time= time.time()
        after_mem= process.memory_info().rss
        elapsed= end_time- start_time
        mem_used= after_mem- before_mem

        global mc_inference_times, mc_memory_usages
        mc_inference_times.append(elapsed)
        mc_memory_usages.append(mem_used)

        print(f"[MonteCarloAI(黒)] MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

        # score最大の手
        best_move= max(move_scores, key=move_scores.get)
        return best_move

###############################################################################
# 8) 1ゲーム実行: 黒(MonteCarloAI), 白(HybridWhiteAI)
###############################################################################
def play_one_game(black_ai, white_ai):
    board= init_board()
    is_black_turn= True  # 先手=黒(MonteCarlo)

    while True:
        print_board(board)
        if is_black_turn:
            print("【黒番: MonteCarloAI】")
            move= black_ai.choose_move(board, True)
        else:
            print("【白番: HybridWhiteAI】")
            move= white_ai.choose_move(board, False)

        if move:
            print(f"→ 着手 {move}\n")
            put(board, move[0], move[1], is_black_turn)
        else:
            print("→ パス\n")
            # パス後 相手も打てないなら終了
            if not any(can_put(board, rr, cc, not is_black_turn) for rr in range(8) for cc in range(8)):
                break
        is_black_turn= not is_black_turn

    # 終了
    print("===== 対局終了 =====")
    print_board(board)
    black,white= count_stones(board)
    print(f"黒(MonteCarloAI)={black}, 白(HybridWhiteAI)={white}")
    if black> white:
        print("→ 黒(MonteCarloAI)の勝利!\n")
        return 1
    elif white> black:
        print("→ 白(HybridWhiteAI)の勝利!\n")
        return -1
    else:
        print("→ 引き分け\n")
        return 0

###############################################################################
# 9) 複数回対戦
###############################################################################
def simulate_games(black_ai, white_ai, num_games=10):
    global model_inference_times, model_memory_usages
    global mc_inference_times,   mc_memory_usages

    model_inference_times=[]
    model_memory_usages=[]
    mc_inference_times=[]
    mc_memory_usages=[]

    black_wins=0
    white_wins=0
    draws=0

    for i in range(num_games):
        print(f"\n=== Game {i+1}/{num_games} ===")
        result= play_one_game(black_ai, white_ai)
        if   result== 1: black_wins+=1
        elif result==-1: white_wins+=1
        else:            draws+=1

    # 結果表示
    print(f"\n===== 対戦結果({num_games}回) =====")
    print(f"黒(MonteCarloAI) 勝利: {black_wins}")
    print(f"白(HybridWhiteAI) 勝利: {white_wins}")
    print(f"引き分け               : {draws}")

    # MonteCarloAI(黒) のシミュレーション統計
    if mc_inference_times:
        avg_time_mc= np.mean(mc_inference_times)
        max_time_mc= np.max(mc_inference_times)
        avg_mem_mc= np.mean(mc_memory_usages)
        max_mem_mc= np.max(mc_memory_usages)
        print("\n--- MonteCarloAI(黒)のシミュレーション統計 ---")
        print(f"シミュレーション時間(秒): 平均={avg_time_mc:.6f}, 最大={max_time_mc:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem_mc:.0f}, 最大={max_mem_mc:.0f}")
    else:
        print("\nMonteCarloAI(黒)でのシミュレーションは行われませんでした。")

    # HybridWhiteAI(白) の モデル推論統計
    if model_inference_times:
        avg_time= np.mean(model_inference_times)
        max_time= np.max(model_inference_times)
        avg_mem= np.mean(model_memory_usages)
        max_mem= np.max(model_memory_usages)
        print("\n--- HybridWhiteAI(白)内 (モデル推論) 統計 ---")
        print(f"推論時間(秒): 平均={avg_time:.6f}, 最大={max_time:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem:.0f}, 最大={max_mem:.0f}")
    else:
        print("\nモデル推論は一度も行われませんでした。")


###############################################################################
# 10) メイン
###############################################################################
if __name__=="__main__":
    print_gpu_memory_usage(gpu_index=0)

    # 1) MonteCarloAIBlack(黒番)
    black_ai= MonteCarloAIBlack(num_simulations=100)

    # 2) HybridWhiteAI(白番)
    #    - モデルAI(白)
    keras_model= keras.models.load_model("data/model_small")
    model_white= ModelAI(keras_model, label="small", my_color=-1)
    #    - Minimax(白)
    minimax_white= MinimaxAI(depth=5, my_color=-1)
    #    - HybridWhiteAI
    white_ai= HybridWhiteAI(model_white, minimax_white, threshold=15)

    # 3) 複数回対戦
    simulate_games(black_ai, white_ai, num_games=500)
