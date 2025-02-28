import random
import numpy as np
import psutil
import time
import os
import pynvml
import tensorflow as tf
from tensorflow import keras

###############################################################################
# 1) GPU割り当ての設定 (必要に応じて使用)
###############################################################################
def select_and_configure_gpu(gpu_index=0):
    """
    指定したgpu_indexのGPUのみを可視化し、memory growthを有効化。
    GPUが無い場合やgpu_indexが範囲外の場合はCPUのみで実行。
    """
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

###############################################################################
# 2) GPUメモリ状況を表示する関数 (pynvml使用)
###############################################################################
def print_gpu_memory_usage(gpu_index=0):
    """GPUメモリの使用状況を表示"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_mb = mem_info.total / 1024**2
    used_mb  = mem_info.used  / 1024**2
    free_mb  = mem_info.free  / 1024**2
    print(f"[GPU Memory Usage index={gpu_index}]:")
    print(f"  Total: {total_mb:.2f} MB")
    print(f"  Used : {used_mb:.2f} MB")
    print(f"  Free : {free_mb:.2f} MB\n")
    pynvml.nvmlShutdown()

###############################################################################
# 3) グローバル変数（推論時間 & メモリ使用量記録）
###############################################################################
model_inference_times = []
model_memory_usages   = []

# MonteCarloAI部分（白）用の時間・メモリ統計
mc_inference_times = []
mc_memory_usages   = []

# psutil で現在のプロセスを取得
process = psutil.Process(os.getpid())

###############################################################################
# 4) オセロ関連の関数
###############################################################################
def init_board():
    """
    オセロ初期配置:
      黒=+1, 白=-1, 空=0
    """
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = board[4, 4] = -1  # 白
    board[3, 4] = board[4, 3] =  1  # 黒
    return board

def print_board(board):
    symbols = {1:"●", -1:"○", 0:"."}
    for row in board:
        print(" ".join(symbols[v] for v in row))
    print()

def can_put(board, row, col, is_black_turn):
    if board[row,col]!=0:
        return False
    current_player = 1 if is_black_turn else -1
    opponent = -current_player

    directions = [
        (-1,0),(1,0),(0,-1),(0,1),
        (-1,-1),(-1,1),(1,-1),(1,1)
    ]
    for dx,dy in directions:
        x,y = row+dx,col+dy
        found_opp=False
        while 0<=x<8 and 0<=y<8:
            if board[x,y]==opponent:
                found_opp=True
            elif board[x,y]==current_player:
                if found_opp:
                    return True
                break
            else:
                break
            x+=dx
            y+=dy
    return False

def put(board, row, col, is_black_turn):
    player= 1 if is_black_turn else -1
    board[row,col]= player
    directions= [
        (-1,0),(1,0),(0,-1),(0,1),
        (-1,-1),(-1,1),(1,-1),(1,1)
    ]
    for dx,dy in directions:
        x,y= row+dx, col+dy
        stones_to_flip= []
        while 0<=x<8 and 0<=y<8 and board[x,y]== -player:
            stones_to_flip.append((x,y))
            x+=dx
            y+=dy
        # 挟めたなら反転
        if 0<=x<8 and 0<=y<8 and board[x,y]== player:
            for fx,fy in stones_to_flip:
                board[fx,fy]= player

def count_stones(board):
    black= np.sum(board==1)
    white= np.sum(board== -1)
    return black,white


###############################################################################
# 5) モデルAI (Keras) -- 1位を90%、2位を10%で選択
###############################################################################
class ModelAI:
    def __init__(self, model, model_label="unknown"):
        self.model= model
        self.model_label= model_label

    def choose_move(self, board, is_black_turn):
        valid_moves= [
            (r,c) for r in range(8) for c in range(8)
            if can_put(board, r,c, is_black_turn)
        ]
        if not valid_moves:
            return None

        board_input= np.zeros((1,2,8,8), dtype='int8')
        if is_black_turn:
            board_input[0,0]= (board==1).astype('int8')
            board_input[0,1]= (board==-1).astype('int8')
        else:
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

        turn_label= "黒" if is_black_turn else "白"
        print(f"[Model({self.model_label})AI - {turn_label}] "
              f"MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

        move_probs= [ preds[r*8+c] for (r,c) in valid_moves ]
        sorted_unique_probs= sorted(list(set(move_probs)), reverse=True)
        if len(sorted_unique_probs)>=2:
            max_prob= sorted_unique_probs[0]
            second_prob= sorted_unique_probs[1]
        else:
            max_prob= sorted_unique_probs[0]
            second_prob= None

        top1_candidates= [
            (r,c) for (r,c) in valid_moves
            if np.isclose(preds[r*8+c], max_prob)
        ]
        if second_prob is not None:
            top2_candidates= [
                (r,c) for (r,c) in valid_moves
                if np.isclose(preds[r*8+c], second_prob)
            ]
        else:
            top2_candidates= []

        p= random.random()
        if top2_candidates and p<0.1:
            return random.choice(top2_candidates)
        else:
            return random.choice(top1_candidates)


###############################################################################
# 6) MinimaxAI
###############################################################################
class MinimaxAI:
    def __init__(self, depth=5, my_color=1):
        """
        depth: 探索深さ
        my_color: +1(黒) or -1(白)
        """
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
        # 自分の番かを確認
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

        print(f"[MinimaxAI(depth={self.depth})] best_eval= {best_val:.2f}")
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
            # 両者パス?
            opp_moves= [
                (r,c) for r in range(8) for c in range(8)
                if can_put(board, r,c, not is_black_turn)
            ]
            if not opp_moves:
                return self.evaluate_position(board)
            else:
                return self.minimax(board, depth, not is_black_turn)

        if current_player== self.my_color:
            best= float('-inf')
            for mv in valid_moves:
                new_bd= board.copy()
                put(new_bd, mv[0], mv[1], is_black_turn)
                val= self.minimax(new_bd, depth-1, not is_black_turn)
                best= max(best,val)
            return best
        else:
            worst= float('inf')
            for mv in valid_moves:
                new_bd= board.copy()
                put(new_bd, mv[0], mv[1], is_black_turn)
                val= self.minimax(new_bd, depth-1, not is_black_turn)
                worst= min(worst,val)
            return worst

    def evaluate_position(self, board):
        me= self.my_color
        opp= -me

        score_me= np.sum((board== me)* self.weights)
        score_op= np.sum((board== opp)* self.weights)
        return score_me- score_op


###############################################################################
# 7) ハイブリッドAI (黒): 序盤はモデル, 終盤(threshold以下)でMinimax
###############################################################################
class HybridBlackAI:
    def __init__(self, model_ai, minimax_ai, threshold=15):
        """
        model_ai : ModelAI (Keras) など
        minimax_ai: MinimaxAI
        threshold: 残り空きマスがコレ以下でMinimaxに切替
        """
        self.model_ai   = model_ai
        self.minimax_ai = minimax_ai
        self.threshold  = threshold

    def choose_move(self, board, is_black_turn):
        if not is_black_turn:
            return None  # 黒番でなければパス
        empty_count= np.count_nonzero(board==0)
        if empty_count<= self.threshold:
            print(f"[HybridBlackAI] 空き={empty_count} <= {self.threshold} => Minimax")
            return self.minimax_ai.choose_move(board, True)
        else:
            print(f"[HybridBlackAI] 空き={empty_count} => モデルAI")
            return self.model_ai.choose_move(board, True)


###############################################################################
# 8) モンテカルロAI (白)
###############################################################################
class MonteCarloAI:
    def __init__(self, num_simulations=50):
        self.num_simulations= num_simulations

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
        if black> white:
            return 1
        elif white> black:
            return -1
        else:
            return 0

    def choose_move(self, board, is_black_turn):
        valid_moves= [
            (r,c) for r in range(8) for c in range(8)
            if can_put(board, r,c, is_black_turn)
        ]
        if not valid_moves:
            return None

        before_mem= process.memory_info().rss
        start_time= time.time()

        move_scores= {}
        for mv in valid_moves:
            # mvを打った局面から self.num_simulations 回ランダムプレイアウト
            sim_bd= board.copy()
            put(sim_bd, mv[0], mv[1], is_black_turn)
            score= 0
            for _ in range(self.num_simulations):
                bd_copy= sim_bd.copy()
                result= self.simulate_game(bd_copy, not is_black_turn)
                # 黒視点の result=1(黒勝)
                # MonteCarloAIは白番なので、scoreは -result
                score+= -result
            move_scores[mv]= score

        end_time= time.time()
        after_mem= process.memory_info().rss
        elapsed= end_time- start_time
        mem_used= after_mem- before_mem

        global mc_inference_times, mc_memory_usages
        mc_inference_times.append(elapsed)
        mc_memory_usages.append(mem_used)

        print(f"[MonteCarloAI(白)] MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

        # score最大の手を返す
        best_move= max(move_scores, key=move_scores.get)
        return best_move


###############################################################################
# 9) 対戦 (黒=HybridBlackAI, 白=MonteCarloAI)
###############################################################################
def play_one_game(black_ai, white_ai):
    board= init_board()
    is_black_turn= True

    while True:
        print_board(board)
        if is_black_turn:
            print("【黒番: HybridBlackAI】")
            move= black_ai.choose_move(board, True)
        else:
            print("【白番: MonteCarloAI】")
            move= white_ai.choose_move(board, False)

        if move:
            print(f"→ 着手 {move}\n")
            put(board, move[0], move[1], is_black_turn)
        else:
            print("→ パス\n")
            if not any(can_put(board, rr,cc, not is_black_turn) for rr in range(8) for cc in range(8)):
                break
        is_black_turn= not is_black_turn

    # 終了
    print("===== 対局終了 =====")
    print_board(board)
    black,white= count_stones(board)
    print(f"黒(HybridBlackAI)={black}, 白(MonteCarloAI)={white}")
    if black> white:
        print("→ 黒(HybridBlackAI)の勝利!\n")
        return 1
    elif white> black:
        print("→ 白(MonteCarloAI)の勝利!\n")
        return -1
    else:
        print("→ 引き分け\n")
        return 0

###############################################################################
# 10) 複数回対戦
###############################################################################
def simulate_games(black_ai, white_ai, num_games=10):
    global model_inference_times, model_memory_usages
    global mc_inference_times, mc_memory_usages

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
    print(f"\n===== 対戦結果 ({num_games}回) =====")
    print(f"黒(HybridBlackAI) 勝利: {black_wins}")
    print(f"白(MonteCarloAI) 勝利: {white_wins}")
    print(f"引き分け            : {draws}")

    # ハイブリッドAIのモデル部(CNN)推論統計
    if model_inference_times:
        avg_time_m= np.mean(model_inference_times)
        max_time_m= np.max(model_inference_times)
        avg_mem_m = np.mean(model_memory_usages)
        max_mem_m = np.max(model_memory_usages)
        print("\n--- HybridBlackAI内 (モデル推論) 統計 ---")
        print(f"推論時間(秒): 平均={avg_time_m:.6f}, 最大={max_time_m:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem_m:.0f}, 最大={max_mem_m:.0f}")
    else:
        print("\nモデル推論は行われませんでした。")

    # MonteCarloAI(白) のシミュレーション統計
    if mc_inference_times:
        avg_time_mc= np.mean(mc_inference_times)
        max_time_mc= np.max(mc_inference_times)
        avg_mem_mc = np.mean(mc_memory_usages)
        max_mem_mc = np.max(mc_memory_usages)
        print("\n--- MonteCarloAI(白) のシミュレーション統計 ---")
        print(f"シミュレーション時間(秒): 平均={avg_time_mc:.6f}, 最大={max_time_mc:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem_mc:.0f}, 最大={max_mem_mc:.0f}")
    else:
        print("\nMonteCarloAI(白)でのシミュレーションは行われませんでした。")


###############################################################################
# メイン
###############################################################################
if __name__=="__main__":
    # 必要ならGPU設定
    select_and_configure_gpu(gpu_index=0)
    print_gpu_memory_usage(gpu_index=0)

    # 1) CNNモデルをロード (Keras)
    keras_model= keras.models.load_model("data/model_small")
    model_ai= ModelAI(keras_model, model_label="small")

    # 2) MinimaxAI
    minimax_ai= MinimaxAI(depth=5, my_color=1)  # 黒

    # 3) HybridBlackAI(黒) (序盤～中盤はmodel_ai, 終盤はminimax_ai)
    black_ai= HybridBlackAI(model_ai, minimax_ai, threshold=15)

    # 4) 白番 = MonteCarloAI
    mc_ai= MonteCarloAI(num_simulations=100)

    # 5) 複数回対戦
    simulate_games(black_ai, mc_ai, num_games=500)
