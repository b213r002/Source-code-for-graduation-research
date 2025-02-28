import random
import numpy as np
import psutil
import time
import os
import pynvml
import tensorflow as tf
from tensorflow import keras

###############################################################################
# 1) GPUの設定 (省略可)
###############################################################################
def select_and_configure_gpu(gpu_index=0):
    """
    指定したgpu_index の GPU のみ可視化し、memory growthを有効化。
    GPUが無い場合や不正なindexの場合は CPU のみを使用。
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
        print(f"GPUデバイス index={gpu_index} のみ使用します。")
    except RuntimeError as e:
        print(e)

def print_gpu_memory_usage(gpu_index=0):
    """
    pynvml を用いて、指定GPUデバイスのメモリ使用状況を表示する。
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_mb = mem_info.total / 1024**2
    used_mb  = mem_info.used  / 1024**2
    free_mb  = mem_info.free  / 1024**2
    print(f"[Before Game] GPU Memory (index={gpu_index}):")
    print(f"  Total: {total_mb:.2f} MB")
    print(f"  Used : {used_mb:.2f} MB")
    print(f"  Free : {free_mb:.2f} MB\n")
    pynvml.nvmlShutdown()

model_inference_times = []
model_memory_usages   = []
minimax_inference_times = []
minimax_memory_usages   = []

process = psutil.Process(os.getpid())
select_and_configure_gpu(gpu_index=0)

def init_board():
    bd = np.zeros((8,8),dtype=int)
    bd[3,3] = bd[4,4] = -1  # 白
    bd[3,4] = bd[4,3] =  1  # 黒
    return bd

def print_board(board):
    symbols = {1:"●", -1:"○", 0:"."}
    for row in board:
        print(" ".join(symbols[v] for v in row))
    print()

def can_put(board, row, col, player):
    if board[row,col]!=0:
        return False
    opp = -player
    directions = [
        (-1,0),(1,0),(0,-1),(0,1),
        (-1,-1),(-1,1),(1,-1),(1,1)
    ]
    for dx,dy in directions:
        x,y = row+dx,col+dy
        found_opp=False
        while 0<=x<8 and 0<=y<8:
            if board[x,y]==opp:
                found_opp=True
            elif board[x,y]==player:
                if found_opp:
                    return True
                break
            else:
                break
            x+=dx
            y+=dy
    return False

def put(board, row, col, player):
    board[row,col] = player
    opp = -player
    directions = [
        (-1,0),(1,0),(0,-1),(0,1),
        (-1,-1),(-1,1),(1,-1),(1,1)
    ]
    for dx,dy in directions:
        x,y = row+dx,col+dy
        flip_stones=[]
        while 0<=x<8 and 0<=y<8 and board[x,y]==opp:
            flip_stones.append((x,y))
            x+=dx
            y+=dy
        if 0<=x<8 and 0<=y<8 and board[x,y]==player:
            for fx,fy in flip_stones:
                board[fx,fy]=player

def count_stones(board):
    black = np.sum(board==1)
    white = np.sum(board== -1)
    return black,white

class ModelAI:
    def __init__(self, model, label="unknown", my_color=1):
        self.model = model
        self.model_label = label
        self.my_color = my_color

    def choose_move(self, board, current_player):
        if current_player!=self.my_color:
            return None
        valid_moves = [
            (r,c) for r in range(8) for c in range(8)
            if can_put(board, r, c, self.my_color)
        ]
        if not valid_moves:
            return None

        board_input = np.zeros((1,2,8,8), dtype='int8')
        me  = (board == self.my_color).astype('int8')
        opp = (board == -self.my_color).astype('int8')
        board_input[0,0] = me
        board_input[0,1] = opp

        before_mem = process.memory_info().rss
        start_time = time.time()

        preds = self.model.predict(board_input, verbose=0)[0]

        end_time = time.time()
        after_mem = process.memory_info().rss
        elapsed = end_time - start_time
        mem_used = after_mem - before_mem

        model_inference_times.append(elapsed)
        model_memory_usages.append(mem_used)

        turn_label = "(ModelAI) me=+1(黒)" if self.my_color==1 else "(ModelAI) me=-1(白)"
        print(f"[{turn_label}] MemUsed={mem_used}, Time={elapsed:.6f}s")

        move_vals = [ preds[r*8 + c] for (r,c) in valid_moves ]
        sort_vals = sorted(list(set(move_vals)), reverse=True)
        if len(sort_vals)>=2:
            best=  sort_vals[0]
            secd=  sort_vals[1]
        else:
            best= sort_vals[0]
            secd= None

        top1= [(r,c) for (r,c),v in zip(valid_moves, move_vals) if np.isclose(v,best)]
        if secd is not None:
            top2= [(r,c) for (r,c),v in zip(valid_moves, move_vals) if np.isclose(v,secd)]
        else:
            top2= []

        p= random.random()
        if top2 and p<0.1:
            return random.choice(top2)
        else:
            return random.choice(top1)

class MinimaxAI:
    def __init__(self, depth=2, my_color=-1):
        self.depth = depth
        self.my_color = my_color

        self.weights = np.array([
            [100, -20,  10,   5,   5,  10, -20, 100],
            [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
            [ 10,  -2,  -1,  -1,  -1,  -1,  -2,  10],
            [  5,  -2,  -1,  -1,  -1,  -1,  -2,   5],
            [  5,  -2,  -1,  -1,  -1,  -1,  -2,   5],
            [ 10,  -2,  -1,  -1,  -1,  -1,  -2,  10],
            [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
            [100, -20,  10,   5,   5,  10, -20, 100]
        ], dtype=float)

    @property
    def label(self):
        return f"Minimax(depth={self.depth})"

    def choose_move(self, board, current_player):
        if current_player!=self.my_color:
            return None

        before_mem = process.memory_info().rss
        start_time = time.time()

        val, mv = self.minimax(board, self.depth, True)

        end_time = time.time()
        after_mem = process.memory_info().rss
        elapsed = end_time - start_time
        mem_used = after_mem - before_mem

        minimax_inference_times.append(elapsed)
        minimax_memory_usages.append(mem_used)

        if self.my_color==1:
            turn_label="(MinimaxAI) me=+1(黒)"
        else:
            turn_label="(MinimaxAI) me=-1(白)"

        print(f"[{turn_label} {self.label}] Time={elapsed:.3f}s (eval={val:.1f}) MemUsed={mem_used}")
        return mv

    def evaluate_position(self, board):
        me= self.my_color
        opp=-me

        score_me = np.sum((board==me)* self.weights)
        score_op = np.sum((board==opp)*self.weights)
        pos_eval = score_me - score_op

        mob_me = sum(can_put(board, r, c, me)  for r in range(8) for c in range(8))
        mob_op = sum(can_put(board, r, c, opp) for r in range(8) for c in range(8))
        mob_eval = mob_me - mob_op

        stable_me  = self.count_stable(board, me)
        stable_opp = self.count_stable(board, opp)
        stable_eval= stable_me - stable_opp

        black, white = count_stones(board)
        me_count = black if me==1 else white
        op_count = black if opp==1 else white
        disc_eval= me_count - op_count

        total_stones= me_count + op_count
        if total_stones<20:
            score= 0.3*pos_eval + 3.0*mob_eval + 1.0*stable_eval
        elif total_stones<50:
            score= 0.7*pos_eval + 2.0*mob_eval + 2.0*stable_eval
        else:
            score= 1.0*pos_eval + 1.0*mob_eval + 4.0*stable_eval + 2.0*disc_eval

        return score

    def count_stable(self, board, color):
        val=0
        corners=[(0,0),(0,7),(7,0),(7,7)]
        for (rr,cc) in corners:
            if board[rr,cc]==color:
                val+=2
        for c in range(8):
            if board[0,c]==color: val+=1
            if board[7,c]==color: val+=1
        for r in range(8):
            if board[r,0]==color: val+=1
            if board[r,7]==color: val+=1
        return val

    def minimax(self, board, depth, is_my_turn, alpha=float('-inf'), beta=float('inf')):
        if depth==0 or np.all(board!=0):
            return self.evaluate_position(board),None

        me= self.my_color
        opp=-me
        player= me if is_my_turn else opp

        valid_moves= [
            (r,c) for r in range(8) for c in range(8)
            if can_put(board,r,c,player)
        ]
        if not valid_moves:
            other_can= any(can_put(board, rr, cc, opp if player==me else me)
                           for rr in range(8) for cc in range(8))
            if not other_can:
                return self.evaluate_position(board),None
            val,_= self.minimax(board, depth, not is_my_turn, alpha,beta)
            return val,None

        move_evals=[]
        for mv in valid_moves:
            new_bd= board.copy()
            put(new_bd, mv[0], mv[1], player)
            sc,_= self.minimax(new_bd, depth-1, not is_my_turn, alpha,beta)
            move_evals.append( (sc,mv) )

            if is_my_turn:
                alpha= max(alpha, sc)
                if beta<=alpha:
                    break
            else:
                beta= min(beta, sc)
                if beta<=alpha:
                    break

        if is_my_turn:
            return max(move_evals, key=lambda x:x[0])
        else:
            return min(move_evals, key=lambda x:x[0])

def play_one_game(aiA, aiB):
    board = init_board()
    is_A_turn=True

    # Aの名前
    if isinstance(aiA, ModelAI):
        aiA_name= f"Model({aiA.model_label})"
    else:
        aiA_name= f"{aiA.label}"
    # Bの名前
    if isinstance(aiB, ModelAI):
        aiB_name= f"Model({aiB.model_label})"
    else:
        aiB_name= f"{aiB.label}"

    print(f"先手(A)={aiA_name}, 後手(B)={aiB_name}\n")

    while True:
        print_board(board)

        current_player = aiA.my_color if is_A_turn else aiB.my_color
        if is_A_turn:
            move = aiA.choose_move(board, current_player)
        else:
            move = aiB.choose_move(board, current_player)

        if move:
            print(f"→ 着手 {move} (Color={'+1(黒)' if current_player==1 else '-1(白)'})\n")
            put(board, move[0], move[1], current_player)
        else:
            print("→ パス\n")
            other_player= aiB.my_color if is_A_turn else aiA.my_color
            if not any(can_put(board, rr, cc, other_player) for rr in range(8) for cc in range(8)):
                break
        is_A_turn= not is_A_turn

    print("===== 対局終了 =====")
    print_board(board)
    black, white= count_stones(board)
    print(f"最終: 黒={black}, 白={white}")

    if black>white:
        # 黒勝利 => どちらが黒か?
        if aiA.my_color==1:
            winner= aiA_name
        else:
            winner= aiB_name
        print(f"→ 勝者: {winner}\n")
        return winner
    elif white>black:
        # 白勝利 => どちらが白か?
        if aiA.my_color==-1:
            winner= aiA_name
        else:
            winner= aiB_name
        print(f"→ 勝者: {winner}\n")
        return winner
    else:
        print("→ 引き分け\n")
        return "draw"

def simulate_games(num_games=5, depthA=3, depthB=3, first_is_minimax=True):
    global model_inference_times
    global model_memory_usages
    global minimax_inference_times
    global minimax_memory_usages

    model_inference_times=[]
    model_memory_usages=[]
    minimax_inference_times=[]
    minimax_memory_usages=[]

    # ★ ここでロードするファイルを指定
    model_path= "data/model_small"
    # model_path= "data/model_medium"
    # model_path= "data/model_large"

    model_keras= keras.models.load_model(model_path)

    # ファイルパスからラベルを推定しておく (自動判定の例)
    if "small" in model_path:
        loaded_label= "small"
    elif "medium" in model_path:
        loaded_label= "medium"
    elif "large" in model_path:
        loaded_label= "large"
    else:
        loaded_label= "unknown"

    # 先手(A)、後手(B) の設定
    if first_is_minimax:
        # A=MinimaxAI(黒=+1)
        aiA= MinimaxAI(depth=depthA, my_color=+1)
        # B=ModelAI(白=-1), label=ファイル名から推定
        aiB= ModelAI(model_keras, label=loaded_label, my_color=-1)
    else:
        # A=ModelAI(黒=+1), label=ファイル名から推定
        aiA= ModelAI(model_keras, label=loaded_label, my_color=+1)
        # B=MinimaxAI(白=-1)
        aiB= MinimaxAI(depth=depthB, my_color=-1)

    # 勝敗カウント
    A_wins=0
    B_wins=0
    draws=0

    for i in range(num_games):
        print(f"\n=== Game {i+1}/{num_games} ===")
        winner= play_one_game(aiA, aiB)
        if winner=="draw":
            draws+=1
        elif winner==(aiA.label if isinstance(aiA,MinimaxAI) else f"Model({aiA.model_label})"):
            A_wins+=1
        elif winner==(aiB.label if isinstance(aiB,MinimaxAI) else f"Model({aiB.model_label})"):
            B_wins+=1

    print("===== 最終結果 =====")
    if isinstance(aiA, MinimaxAI):
        print(f"先手(A)={aiA.label}, 後手(B)=Model({aiB.model_label})")
    else:
        print(f"先手(A)=Model({aiA.model_label}), 後手(B)={aiB.label}")
    print(f"{num_games}回の対戦で、勝敗: (先手(A)={A_wins}, 後手(B)={B_wins}, 引き分け={draws})")

    # 勝率 (引き分けを分母に含めるかどうかは自由)
    win_rate_A= A_wins/num_games*100
    win_rate_B= B_wins/num_games*100
    print(f"→ 先手(A)の勝率= {win_rate_A:.1f}%, 後手(B)の勝率= {win_rate_B:.1f}% (引き分けは除外)")

    # ModelAI(白/黒) の推論統計
    if model_inference_times:
        avg_time_model= np.mean(model_inference_times)
        max_time_model= np.max(model_inference_times)
        avg_mem_model=  np.mean(model_memory_usages)
        max_mem_model=  np.max(model_memory_usages)
        print("\n--- ModelAIの推論統計 ---")
        print(f"推論時間(秒): 平均={avg_time_model:.6f}, 最大={max_time_model:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem_model:.0f}, 最大={max_mem_model:.0f}")
    else:
        print("\nModelAIは1回も推論を行いませんでした。")

    # MinimaxAI(黒/白) の探索統計
    if minimax_inference_times:
        avg_time_mini= np.mean(minimax_inference_times)
        max_time_mini= np.max(minimax_inference_times)
        avg_mem_mini=  np.mean(minimax_memory_usages)
        max_mem_mini=  np.max(minimax_memory_usages)
        print("\n--- MinimaxAIの探索統計 ---")
        print(f"探索時間(秒): 平均={avg_time_mini:.6f}, 最大={max_time_mini:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem_mini:.0f}, 最大={max_mem_mini:.0f}")
    else:
        print("\nMinimaxAIは1回も探索を行いませんでした。")

if __name__=="__main__":
    print_gpu_memory_usage(gpu_index=0)

    # 先手=Minimax(黒), 後手=Model(白) で 15回対戦
    simulate_games(num_games=500, depthA=5, depthB=5, first_is_minimax=True)

    # 先手=Model(黒), 後手=Minimax(白) で 15回対戦
    # simulate_games(num_games=500, depthA=5, depthB=5, first_is_minimax=False)








