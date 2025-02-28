import random
import numpy as np
import psutil
import time
import os
import pynvml
import tensorflow as tf
from tensorflow import keras

###############################################################################
# 1) GPU割り当ての設定
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
        # 指定したGPUデバイスのみ可視化
        tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
        # メモリ成長を有効化
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        print(f"GPUデバイス index={gpu_index} のみが使用されます。")
    except RuntimeError as e:
        # set_visible_devicesは初期化前に呼び出す必要があるため、
        # 既に初期化済みだとRuntimeErrorになる場合がある
        print(e)

###############################################################################
# 2) GPUメモリ状況を表示する関数 (pynvml使用)
###############################################################################
def print_gpu_memory_usage(gpu_index=0):
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

###############################################################################
# グローバル変数（推論時間 & メモリ使用量記録）
###############################################################################
model_inference_times = []
model_memory_usages   = []

mc_inference_times = []
mc_memory_usages   = []

# psutil で現在のプロセスを取得
process = psutil.Process(os.getpid())

###############################################################################
# 3) GPUを選択・設定
###############################################################################
select_and_configure_gpu(gpu_index=0)  # 例: 0番目のGPUを使用

###############################################################################
# モデルの読み込み & モデルサイズの設定
###############################################################################
model = keras.models.load_model("data/model_large")
model_size = "large"  # "small", "medium", "large" など

###############################################################################
# オセロ盤の初期化
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

###############################################################################
# 石を置けるか判定
###############################################################################
def can_put(board, row, col, is_black_turn):
    if board[row, col] != 0:
        return False

    current_player = 1 if is_black_turn else -1
    opponent = -current_player

    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    for dx, dy in directions:
        x, y = row + dx, col + dy
        found_opponent = False
        while 0 <= x < 8 and 0 <= y < 8:
            if board[x, y] == opponent:
                found_opponent = True
            elif board[x, y] == current_player:
                if found_opponent:
                    return True
                break
            else:
                break
            x += dx
            y += dy
    return False

###############################################################################
# 石を置いて相手の石を反転
###############################################################################
def put(board, row, col, is_black_turn):
    player = 1 if is_black_turn else -1
    board[row, col] = player

    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    for dx, dy in directions:
        x, y = row + dx, col + dy
        stones_to_flip = []
        while 0 <= x < 8 and 0 <= y < 8 and board[x, y] == -player:
            stones_to_flip.append((x, y))
            x += dx
            y += dy
        # 挟めた場合のみ反転
        if 0 <= x < 8 and 0 <= y < 8 and board[x, y] == player:
            for fx, fy in stones_to_flip:
                board[fx, fy] = player

###############################################################################
# 盤面表示
###############################################################################
def print_board(board):
    symbols = {1: "●", -1: "○", 0: "."}
    for row in board:
        print(" ".join(symbols[v] for v in row))
    print()

###############################################################################
# 石数を数えて勝敗を判定
###############################################################################
def count_stones(board):
    black = np.sum(board == 1)
    white = np.sum(board == -1)
    return black, white

###############################################################################
# 4) モデルAIクラス (後手=白) 
#    "1位の手を90%、2位の手を10%" のロジックを組み込み
###############################################################################
class ModelAI:
    def __init__(self, model, model_label="unknown"):
        self.model = model
        self.model_label = model_label  # 例: "small", "medium", etc.

    def choose_move(self, board, is_black_turn):
        valid_moves = [(r, c) for r in range(8) for c in range(8)
                       if can_put(board, r, c, is_black_turn)]
        if not valid_moves:
            return None

        # (1,2,8,8) の入力データを作成
        board_input = np.zeros((1, 2, 8, 8), dtype='int8')
        if is_black_turn:
            # 黒番 → channel0=黒, channel1=白
            board_input[0, 0] = (board == 1).astype('int8')
            board_input[0, 1] = (board == -1).astype('int8')
        else:
            # 白番 → channel0=白, channel1=黒
            board_input[0, 0] = (board == -1).astype('int8')
            board_input[0, 1] = (board ==  1).astype('int8')

        # CPUメモリ&時間を計測
        before_mem = process.memory_info().rss
        start_time = time.time()

        predictions = self.model.predict(board_input, verbose=0)[0]  # shape=(64,)

        end_time = time.time()
        after_mem = process.memory_info().rss
        elapsed  = end_time - start_time
        mem_used = after_mem - before_mem

        # グローバルリストへ記録
        model_inference_times.append(elapsed)
        model_memory_usages.append(mem_used)

        # ログ出力(モデルサイズ)
        turn_label = "白" if not is_black_turn else "黒"
        print(f"[Model({self.model_label})AI - {turn_label}番] MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

        # ========== ここで "1位を90%、2位を10%" のロジックを実装 ==========
        # valid_movesのpredictions値を取得
        move_probs = [predictions[r*8 + c] for (r, c) in valid_moves]
        # distinctな確率値をソート (上位2つを取得)
        sorted_unique_probs = sorted(list(set(move_probs)), reverse=True)

        if len(sorted_unique_probs) >= 2:
            max_prob = sorted_unique_probs[0]
            second_prob = sorted_unique_probs[1]
        else:
            # 2位が無い場合
            max_prob = sorted_unique_probs[0]
            second_prob = None

        # 1位の候補手, 2位の候補手を抽出
        top1_candidates = [
            (r, c) for (r, c) in valid_moves
            if np.isclose(predictions[r*8 + c], max_prob)
        ]
        if second_prob is not None:
            top2_candidates = [
                (r, c) for (r, c) in valid_moves
                if np.isclose(predictions[r*8 + c], second_prob)
            ]
        else:
            top2_candidates = []

        # 90%で1位、10%で2位を選ぶ (2位が無い場合は必ず1位)
        p = random.random()
        if top2_candidates and p < 0.1:
            # 10%で2位の候補からランダム
            return random.choice(top2_candidates)
        else:
            # 90% or 2位手なし → 1位の候補からランダム
            return random.choice(top1_candidates)

###############################################################################
# モンテカルロ法AIクラス (先手=黒)
###############################################################################
class MonteCarloAI:
    def __init__(self, num_simulations=50):
        self.num_simulations = num_simulations

    def simulate_game(self, board, is_black_turn):
        """1回のランダムプレイアウト: 黒>白=1, 白>黒=-1, 同数=0"""
        simulated_board = board.copy()
        turn = is_black_turn
        while True:
            valid_moves = [(r, c) for r in range(8) for c in range(8)
                           if can_put(simulated_board, r, c, turn)]
            if valid_moves:
                move = random.choice(valid_moves)
                put(simulated_board, move[0], move[1], turn)
            else:
                # パス後、相手も置けなければ終了
                if not any(can_put(simulated_board, rr, cc, not turn) for rr in range(8) for cc in range(8)):
                    break
            turn = not turn

        black, white = count_stones(simulated_board)
        if black > white:
            return 1
        elif white > black:
            return -1
        else:
            return 0

    def choose_move(self, board, is_black_turn):
        # 修正箇所: row/col → r/c と統一
        valid_moves = [(r, c) for r in range(8) for c in range(8)
                       if can_put(board, r, c, is_black_turn)]
        if not valid_moves:
            return None

        before_mem = process.memory_info().rss
        start_time = time.time()

        move_scores = {}
        for move in valid_moves:
            score = 0
            # moveを実際に打ってからランダムプレイアウト
            for _ in range(self.num_simulations):
                sim_board = board.copy()
                put(sim_board, move[0], move[1], is_black_turn)
                result = self.simulate_game(sim_board, not is_black_turn)
                # 黒視点: result=1(黒勝), -1(白勝)
                # is_black_turn=Trueなら score+=result, Falseなら -result
                score += result if is_black_turn else -result
            move_scores[move] = score

        end_time = time.time()
        after_mem = process.memory_info().rss
        elapsed  = end_time - start_time
        mem_used = after_mem - before_mem

        mc_inference_times.append(elapsed)
        mc_memory_usages.append(mem_used)

        print(f"[MonteCarloAI] MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

        # スコアが最大の手を選択
        best_move = max(move_scores, key=move_scores.get)
        return best_move

###############################################################################
# 1ゲーム対戦 (黒=MonteCarloAI, 白=Model(small)AI)
###############################################################################
def play_one_game(mc_ai, model_ai):
    board = init_board()
    is_black_turn = True  # 先手=黒(MonteCarloAI)

    while True:
        print_board(board)

        if is_black_turn:
            print("【黒番: MonteCarloAI】")
            move = mc_ai.choose_move(board, True)
        else:
            print(f"【白番: Model({model_ai.model_label})AI】")
            move = model_ai.choose_move(board, False)

        if move:
            print(f"→ 着手 {move}\n")
            put(board, move[0], move[1], is_black_turn)
        else:
            print("→ パス\n")
            # パス後、相手も置けなければ終了
            if not any(can_put(board, rr, cc, not is_black_turn) for rr in range(8) for cc in range(8)):
                break

        is_black_turn = not is_black_turn

    print("===== 対局終了 =====")
    print_board(board)
    black, white = count_stones(board)
    print(f"黒(MonteCarloAI) = {black}, 白(Model({model_ai.model_label})AI) = {white}")

    if black > white:
        print("→ 黒(MonteCarloAI)の勝利!\n")
        return 1
    elif white > black:
        print(f"→ 白(Model({model_ai.model_label})AI)の勝利!\n")
        return -1
    else:
        print("→ 引き分け\n")
        return 0

###############################################################################
# 複数回の対戦で勝敗と統計を表示
###############################################################################
def simulate_games(num_games=10, num_simulations_mc=50):
    global model_inference_times
    global model_memory_usages
    global mc_inference_times
    global mc_memory_usages

    model_inference_times = []
    model_memory_usages   = []
    mc_inference_times    = []
    mc_memory_usages      = []

    mc_ai    = MonteCarloAI(num_simulations=num_simulations_mc)
    model_ai = ModelAI(model, model_label=model_size)

    mc_wins    = 0
    model_wins = 0
    draws      = 0

    for i in range(num_games):
        print(f"=== Game {i+1}/{num_games} ===")
        result = play_one_game(mc_ai, model_ai)
        if result == 1:
            mc_wins += 1
        elif result == -1:
            model_wins += 1
        else:
            draws += 1

    # 最終結果
    print(f"\n===== 対戦結果({num_games}回) =====")
    print(f"MonteCarloAI(黒) 勝利: {mc_wins}")
    print(f"Model({model_size})AI(白) 勝利: {model_wins}")
    print(f"引き分け              : {draws}")

    # ModelAI(白) の推論統計
    if model_inference_times:
        print(f"\n--- Model({model_size})AI(白)の推論統計 ---")
        avg_time_model = np.mean(model_inference_times)
        max_time_model = np.max(model_inference_times)
        avg_mem_model  = np.mean(model_memory_usages)
        max_mem_model  = np.max(model_memory_usages)
        print(f"推論時間(秒): 平均={avg_time_model:.6f}, 最大={max_time_model:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem_model:.0f}, 最大={max_mem_model:.0f}")
    else:
        print(f"\nModel({model_size})AI(白)は1回も推論しませんでした。")

    # MonteCarloAI(黒) のシミュレーション統計
    if mc_inference_times:
        print("\n--- MonteCarloAI(黒)のシミュレーション統計 ---")
        avg_time_mc = np.mean(mc_inference_times)
        max_time_mc = np.max(mc_inference_times)
        avg_mem_mc  = np.mean(mc_memory_usages)
        max_mem_mc  = np.max(mc_memory_usages)
        print(f"シミュレーション時間(秒): 平均={avg_time_mc:.6f}, 最大={max_time_mc:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem_mc:.0f}, 最大={max_mem_mc:.0f}")
    else:
        print("\nMonteCarloAI(黒)は1回もシミュレーションしませんでした。")

###############################################################################
# メイン (例: GPUメモリ状況を表示してから500回対戦実行)
###############################################################################
if __name__ == "__main__":
    print_gpu_memory_usage(gpu_index=0)  # 0番目GPUのメモリ状況を表示
    simulate_games(num_games=500, num_simulations_mc=100)

