import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import psutil
import statistics
import time
import os
import pynvml

###############################################################################
# 1) GPUの割り当てを制御する関数 (必要に応じて使用)
###############################################################################
def select_and_configure_gpu(gpu_index=0):
    """
    指定した gpu_index の GPU のみを可視化し、memory growth を有効化。
    GPU が無い場合やインデックスが範囲外の場合は CPU 実行。
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("GPU が検出されませんでした。CPU のみで実行します。")
        return
    if gpu_index < 0 or gpu_index >= len(gpus):
        print(f"指定した gpu_index={gpu_index} は範囲外です。CPU のみで実行します。")
        return

    try:
        tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        print(f"GPU デバイス (index={gpu_index}) のみを使用し、メモリ成長を有効化しました。")
    except RuntimeError as e:
        print(e)

###############################################################################
# 2) GPU メモリ状況を表示する関数 (pynvml 使用) -- 任意
###############################################################################
def print_gpu_memory_usage(gpu_index=0):
    """
    pynvml を用いて、指定した GPU のメモリ状況を表示する。
    """
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
# グローバル変数: 推論時間 & メモリ使用量をモデルAIが使う場合のみ記録
###############################################################################
inference_times = []
memory_usages   = []

# psutil で現在のプロセスを取得 (メモリ使用量測定に使う)
process = psutil.Process(os.getpid())

###############################################################################
# 3) オセロ関連の関数
###############################################################################
def init_board():
    """
    オセロの初期配置:
      黒=+1, 白=-1, 空=0
    """
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = board[4, 4] = -1  # 白
    board[3, 4] = board[4, 3] =  1  # 黒
    return board

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

def print_board(board):
    symbols = {1: "●", -1: "○", 0: "."}
    for row in board:
        print(" ".join(symbols[v] for v in row))
    print()

def count_stones(board):
    black = np.sum(board == 1)
    white = np.sum(board == -1)
    return black, white


###############################################################################
# 4) 黒番(モデルAI)向けクラス: Kerasモデル
###############################################################################
class KerasModelAI:
    def __init__(self, model, label="keras_unknown"):
        """
        Kerasモデルを使うAI
        model : 事前学習済みのKerasモデル
        label : 識別用ラベル (例: "small", "medium", "large"など)
        """
        self.model = model
        self.label = label

    def choose_move(self, board, is_black_turn):
        """
        1) 合法手を列挙
        2) Kerasモデルで推論し、上位2候補を 90%/10% で選択
        """
        valid_moves = [(r, c) for r in range(8) for c in range(8)
                       if can_put(board, r, c, is_black_turn)]
        if not valid_moves:
            return None  # パス

        # (1,2,8,8) 形式で入力
        board_input = np.zeros((1, 2, 8, 8), dtype='int8')
        if is_black_turn:
            board_input[0, 0] = (board ==  1).astype('int8')  # 黒
            board_input[0, 1] = (board == -1).astype('int8')  # 白
        else:
            board_input[0, 0] = (board == -1).astype('int8')  # 白
            board_input[0, 1] = (board ==  1).astype('int8')  # 黒

        # 推論前後のメモリ＆時間を計測
        before_mem = process.memory_info().rss
        start_time = time.time()

        predictions = self.model.predict(board_input, verbose=0)[0]  # shape=(64,)

        end_time = time.time()
        after_mem = process.memory_info().rss
        elapsed  = end_time - start_time
        mem_used = after_mem - before_mem

        inference_times.append(elapsed)
        memory_usages.append(mem_used)

        who = "黒" if is_black_turn else "白"
        print(f"[KerasModelAI({self.label}) - {who}] MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

        # 確率上位2手を 90%:10% で選ぶ
        move_probs = [predictions[r*8 + c] for (r,c) in valid_moves]
        sorted_unique_probs = sorted(list(set(move_probs)), reverse=True)

        if len(sorted_unique_probs) >= 2:
            max_prob = sorted_unique_probs[0]
            second_prob = sorted_unique_probs[1]
        else:
            max_prob = sorted_unique_probs[0]
            second_prob = None

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

        p = random.random()
        if top2_candidates and p < 0.1:
            return random.choice(top2_candidates)
        else:
            return random.choice(top1_candidates)

###############################################################################
# 5) 黒番(モデルAI)向けクラス: 量子化TFLiteモデル
###############################################################################
class TFLiteModelAI:
    def __init__(self, tflite_path, label="tflite_unknown"):
        """
        TFLite (量子化済み)モデルを使うAI
        tflite_path: .tflite ファイル
        """
        self.label = label
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def choose_move(self, board, is_black_turn):
        valid_moves = [(r, c) for r in range(8) for c in range(8)
                       if can_put(board, r, c, is_black_turn)]
        if not valid_moves:
            return None  # パス

        # ダイナミックレンジ量子化なら float32 入力が多い
        board_input = np.zeros((1, 2, 8, 8), dtype=np.float32)
        if is_black_turn:
            board_input[0, 0] = (board ==  1).astype(np.float32)  # 黒
            board_input[0, 1] = (board == -1).astype(np.float32)  # 白
        else:
            board_input[0, 0] = (board == -1).astype(np.float32)  # 白
            board_input[0, 1] = (board ==  1).astype(np.float32)  # 黒

        before_mem = process.memory_info().rss
        start_time = time.time()

        # TFLite推論
        self.interpreter.set_tensor(self.input_details[0]['index'], board_input)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # shape=(64,)

        end_time = time.time()
        after_mem = process.memory_info().rss
        elapsed  = end_time - start_time
        mem_used = after_mem - before_mem

        inference_times.append(elapsed)
        memory_usages.append(mem_used)

        who = "黒" if is_black_turn else "白"
        print(f"[TFLiteModelAI({self.label}) - {who}] MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

        # 確率上位2手を 90%:10% で選ぶ (Keras版と同じロジック)
        move_probs = [predictions[r*8 + c] for (r,c) in valid_moves]
        sorted_unique_probs = sorted(list(set(move_probs)), reverse=True)

        if len(sorted_unique_probs) >= 2:
            max_prob = sorted_unique_probs[0]
            second_prob = sorted_unique_probs[1]
        else:
            max_prob = sorted_unique_probs[0]
            second_prob = None

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

        p = random.random()
        if top2_candidates and p < 0.1:
            return random.choice(top2_candidates)
        else:
            return random.choice(top1_candidates)

###############################################################################
# 6) 白番(ランダムAI)用クラス
###############################################################################
class RandomAI:
    def choose_move(self, board, is_black_turn):
        valid_moves = [(r, c) for r in range(8) for c in range(8)
                       if can_put(board, r, c, is_black_turn)]
        if not valid_moves:
            return None
        return random.choice(valid_moves)

###############################################################################
# 7) 1ゲーム実行: 黒(モデルAI) vs. 白(ランダムAI)
###############################################################################
def play_one_game(black_ai, white_ai):
    board = init_board()
    is_black_turn = True  # 黒(モデルAI)が先手

    while True:
        print_board(board)

        if is_black_turn:
            print("【黒番: ModelAI】")
            move = black_ai.choose_move(board, True)
        else:
            print("【白番: RandomAI】")
            move = white_ai.choose_move(board, False)

        if move:
            print(f"  → 着手: {move}\n")
            put(board, move[0], move[1], is_black_turn)
        else:
            print("  → パス\n")
            # パス後、相手もパスなら終了
            if not any(can_put(board, r, c, not is_black_turn) for r in range(8) for c in range(8)):
                break

        is_black_turn = not is_black_turn

    # 対局終了
    print("===== 対局終了 =====")
    print_board(board)

    black, white = count_stones(board)
    print(f"黒(モデルAI) = {black}, 白(ランダムAI) = {white}")

    if black > white:
        print("→ 黒(モデルAI)の勝利!\n")
        return 1
    elif white > black:
        print("→ 白(ランダムAI)の勝利!\n")
        return -1
    else:
        print("→ 引き分け\n")
        return 0

###############################################################################
# 8) 複数回対戦し、結果 & 推論統計を表示
###############################################################################
def simulate_games(black_ai, white_ai, num_games=10):
    global inference_times
    global memory_usages
    inference_times = []
    memory_usages   = []

    black_wins = 0
    white_wins = 0
    draws      = 0

    for i in range(num_games):
        print(f"=== Game {i+1}/{num_games} ===")
        result = play_one_game(black_ai, white_ai)
        if result == 1:
            black_wins += 1
        elif result == -1:
            white_wins += 1
        else:
            draws += 1

    # 結果表示
    print(f"\n===== 対戦結果({num_games}回) =====")
    print(f"黒(モデルAI) 勝利: {black_wins}")
    print(f"白(ランダムAI) 勝利: {white_wins}")
    print(f"引き分け             : {draws}")

    # 推論統計
    if inference_times:
        avg_time = np.mean(inference_times)
        max_time = np.max(inference_times)
        avg_mem  = np.mean(memory_usages)
        max_mem  = np.max(memory_usages)
        print("\n--- モデルAIの推論統計 ---")
        print(f"推論時間(秒): 平均={avg_time:.6f}, 最大={max_time:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem:.0f}, 最大={max_mem:.0f}")
    else:
        print("\nモデルAIの推論が行われませんでした。")

###############################################################################
# 9) メイン
###############################################################################
if __name__ == "__main__":
    # GPUを使いたいなら (任意)
    select_and_configure_gpu(gpu_index=0)

    # GPUのメモリ状況を確認したければ (任意)
    #print_gpu_memory_usage(gpu_index=0)

    # ---- 例1: Kerasモデルを黒番に使用する場合 ----
    # keras_model = keras.models.load_model("data/model_large")
    # black_ai = KerasModelAI(keras_model, label="large")

    # ---- 例2: 量子化TFLiteモデルを黒番に使用する場合 ----
    # tflite_path = "model_small_dynamic_range.tflite"
    # black_ai = TFLiteModelAI(tflite_path, label="small_quant")

    # tflite_path = "model_medium_dynamic_range.tflite"
    # black_ai = TFLiteModelAI(tflite_path, label="medium_quant")

    tflite_path = "model_large_dynamic_range.tflite"
    black_ai = TFLiteModelAI(tflite_path, label="large_quant")

    # 白番(ランダムAI)
    white_ai = RandomAI()

    # 複数回対戦 (例: 10回)
    simulate_games(black_ai, white_ai, num_games=500)




