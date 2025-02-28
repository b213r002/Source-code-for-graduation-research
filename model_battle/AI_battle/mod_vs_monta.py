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
        # 指定したGPUデバイスのみ可視化
        tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
        # メモリ成長を有効化
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        print(f"GPUデバイス index={gpu_index} のみが使用されます。")
    except RuntimeError as e:
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
# 5) モデルAI (Keras) -- 1位を90%、2位を10%で選択
###############################################################################
class ModelAI:
    def __init__(self, model, model_label="unknown"):
        """
        Kerasモデルを使った推論AI
        """
        self.model = model
        self.model_label = model_label  # small,medium,largeなど識別用

    def choose_move(self, board, is_black_turn):
        valid_moves = [(r, c) for r in range(8) for c in range(8)
                       if can_put(board, r, c, is_black_turn)]
        if not valid_moves:
            return None

        # 入力データ作成
        board_input = np.zeros((1, 2, 8, 8), dtype='int8')
        if is_black_turn:
            board_input[0, 0] = (board == 1).astype('int8')
            board_input[0, 1] = (board == -1).astype('int8')
        else:
            board_input[0, 0] = (board == -1).astype('int8')
            board_input[0, 1] = (board ==  1).astype('int8')

        # 推論前後でメモリ・時間を計測
        before_mem = process.memory_info().rss
        start_time = time.time()

        predictions = self.model.predict(board_input, verbose=0)[0]  # shape=(64,)

        end_time = time.time()
        after_mem = process.memory_info().rss
        elapsed  = end_time - start_time
        mem_used = after_mem - before_mem

        # グローバルリストへ記録
        global model_inference_times, model_memory_usages
        model_inference_times.append(elapsed)
        model_memory_usages.append(mem_used)

        turn_label = "黒" if is_black_turn else "白"
        print(f"[Model({self.model_label})AI - {turn_label}番] "
              f"MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

        # 上位手を確率的に選択 (1位=90%, 2位=10%)
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
# 6) モデルAI (TFLite) -- 1位を90%、2位を10%で選択
###############################################################################
class TFLiteModelAI:
    def __init__(self, tflite_path, model_label="unknown"):
        """
        TFLiteモデルを使った推論AI
        tflite_path: .tfliteファイルへのパス
        """
        self.model_label = model_label

        # TFLiteInterpreterをロードして初期化
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        # 入出力テンソルの情報を取得
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def choose_move(self, board, is_black_turn):
        valid_moves = [(r, c) for r in range(8) for c in range(8)
                       if can_put(board, r, c, is_black_turn)]
        if not valid_moves:
            return None

        # 入力データ作成 (float32でもOK: ダイナミックレンジ量子化なら)
        board_input = np.zeros((1, 2, 8, 8), dtype=np.float32)
        if is_black_turn:
            board_input[0, 0] = (board == 1).astype(np.float32)
            board_input[0, 1] = (board == -1).astype(np.float32)
        else:
            board_input[0, 0] = (board == -1).astype(np.float32)
            board_input[0, 1] = (board ==  1).astype(np.float32)

        # 推論前後でメモリ・時間を計測
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

        # グローバルリストへ記録
        global model_inference_times, model_memory_usages
        model_inference_times.append(elapsed)
        model_memory_usages.append(mem_used)

        turn_label = "黒" if is_black_turn else "白"
        print(f"[TFLiteModel({self.model_label})AI - {turn_label}番] "
              f"MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

        # 上位手を確率的に選択 (1位=90%, 2位=10%)
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
# 7) モンテカルロAI (白番)
###############################################################################
class MonteCarloAI:
    def __init__(self, num_simulations=50):
        self.num_simulations = num_simulations

    def simulate_game(self, board, is_black_turn):
        """1回のランダムプレイアウト: 黒>白=1, 白>黒=-1, 同数=0"""
        simulated_board = board.copy()
        turn = is_black_turn
        while True:
            valid_moves = [(r,c) for r in range(8) for c in range(8)
                           if can_put(simulated_board, r, c, turn)]
            if valid_moves:
                move = random.choice(valid_moves)
                put(simulated_board, move[0], move[1], turn)
            else:
                # パス後、相手も置けなければ終了
                if not any(can_put(simulated_board, rr, cc, not turn)
                           for rr in range(8) for cc in range(8)):
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
        valid_moves = [(r,c) for r in range(8) for c in range(8)
                       if can_put(board, r, c, is_black_turn)]
        if not valid_moves:
            return None

        before_mem = process.memory_info().rss
        start_time = time.time()

        move_scores = {}
        for move in valid_moves:
            score = 0
            # moveを実際に打ってからランダムプレイアウト
            sim_board = board.copy()
            put(sim_board, move[0], move[1], is_black_turn)

            # その後の局面を self.num_simulations 回 ランダムプレイアウト
            for _ in range(self.num_simulations):
                board_copy = sim_board.copy()
                result = self.simulate_game(board_copy, not is_black_turn)
                # 黒視点: result=1(黒勝), -1(白勝)
                # is_black_turn=False (白番)ならスコアを逆転
                score += result if is_black_turn else -result
            move_scores[move] = score

        end_time = time.time()
        after_mem = process.memory_info().rss
        elapsed  = end_time - start_time
        mem_used = after_mem - before_mem

        global mc_inference_times, mc_memory_usages
        mc_inference_times.append(elapsed)
        mc_memory_usages.append(mem_used)

        print(f"[MonteCarloAI(白)] MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

        best_move = max(move_scores, key=move_scores.get)
        return best_move


###############################################################################
# 8) 対戦 (黒番 = 任意モデルAI, 白番=MonteCarloAI)
###############################################################################
def play_one_game(model_ai, mc_ai):
    board = init_board()
    is_black_turn = True  # 先手=黒(model_ai)

    while True:
        print_board(board)
        if is_black_turn:
            print("【黒番: ModelAI】")
            move = model_ai.choose_move(board, True)
        else:
            print("【白番: MonteCarloAI】")
            move = mc_ai.choose_move(board, False)

        if move:
            print(f"→ 着手 {move}\n")
            put(board, move[0], move[1], is_black_turn)
        else:
            print("→ パス\n")
            # パスしたあと相手も打てなければ終了
            if not any(can_put(board, rr, cc, not is_black_turn) 
                       for rr in range(8) for cc in range(8)):
                break

        is_black_turn = not is_black_turn

    print("===== 対局終了 =====")
    print_board(board)
    black, white = count_stones(board)
    print(f"黒(ModelAI) = {black}, 白(MonteCarloAI) = {white}")

    if black > white:
        print("→ 黒(ModelAI)の勝利!\n")
        return 1
    elif white > black:
        print("→ 白(MonteCarloAI)の勝利!\n")
        return -1
    else:
        print("→ 引き分け\n")
        return 0

###############################################################################
# 9) 複数回の対戦を行い結果表示
###############################################################################
def simulate_games(model_ai, mc_ai, num_games=10):
    global model_inference_times
    global model_memory_usages
    global mc_inference_times
    global mc_memory_usages

    # グローバルリスト初期化
    model_inference_times = []
    model_memory_usages   = []
    mc_inference_times    = []
    mc_memory_usages      = []

    model_wins = 0
    mc_wins    = 0
    draws      = 0

    for i in range(num_games):
        print(f"=== Game {i+1}/{num_games} ===")
        result = play_one_game(model_ai, mc_ai)
        if result == 1:
            model_wins += 1
        elif result == -1:
            mc_wins += 1
        else:
            draws += 1

    # 最終結果表示
    print(f"\n===== 対戦結果({num_games}回) =====")
    print(f"ModelAI(黒) 勝利: {model_wins}")
    print(f"MonteCarloAI(白) 勝利: {mc_wins}")
    print(f"引き分け            : {draws}")

    # モデルAI(黒)の推論統計
    if model_inference_times:
        avg_time_model = np.mean(model_inference_times)
        max_time_model = np.max(model_inference_times)
        avg_mem_model  = np.mean(model_memory_usages)
        max_mem_model  = np.max(model_memory_usages)
        print(f"\n--- ModelAI(黒)の推論統計 ---")
        print(f"推論時間(秒): 平均={avg_time_model:.6f}, 最大={max_time_model:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem_model:.0f}, 最大={max_mem_model:.0f}")
    else:
        print("\nModelAI(黒)は1回も推論しませんでした。")

    # モンテカルロAI(白)の統計
    if mc_inference_times:
        avg_time_mc = np.mean(mc_inference_times)
        max_time_mc = np.max(mc_inference_times)
        avg_mem_mc  = np.mean(mc_memory_usages)
        max_mem_mc  = np.max(mc_memory_usages)
        print(f"\n--- MonteCarloAI(白)のシミュレーション統計 ---")
        print(f"シミュレーション時間(秒): 平均={avg_time_mc:.6f}, 最大={max_time_mc:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem_mc:.0f}, 最大={max_mem_mc:.0f}")
    else:
        print("\nMonteCarloAI(白)は1回もシミュレーションしませんでした。")


###############################################################################
# 10) メイン
###############################################################################
if __name__ == "__main__":
    # 必要に応じてGPU設定 (任意)
    select_and_configure_gpu(gpu_index=0)
    print_gpu_memory_usage(gpu_index=0)

    # 例1: 通常のKerasモデルを使う場合
    # model_keras = keras.models.load_model("data/model_large")
    # model_ai = ModelAI(model_keras, model_label="large")

    # 例2: 量子化TFLiteモデルを使う場合
    tflite_path = "model_large_dynamic_range.tflite"
    model_ai = TFLiteModelAI(tflite_path, model_label="large_quant")

    # MonteCarloAI
    mc_ai = MonteCarloAI(num_simulations=100)

    # 複数回対戦
    simulate_games(model_ai, mc_ai, num_games=500)




# import random
# import numpy as np
# import psutil
# import time
# import os
# import pynvml
# import tensorflow as tf
# from tensorflow import keras

# ###############################################################################
# # 1) GPU割り当ての設定
# ###############################################################################
# def select_and_configure_gpu(gpu_index=0):
#     """
#     指定したgpu_indexのGPUのみを可視化し、memory growthを有効化。
#     GPUが無い場合やgpu_indexが範囲外の場合はCPUのみで実行。
#     """
#     gpus = tf.config.list_physical_devices('GPU')
#     if not gpus:
#         print("GPUが検出されませんでした。CPUのみで実行します。")
#         return

#     if gpu_index < 0 or gpu_index >= len(gpus):
#         print(f"指定したgpu_index={gpu_index}は無効です。CPUのみで実行します。")
#         return

#     try:
#         # 指定したGPUデバイスのみ可視化
#         tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
#         # メモリ成長を有効化
#         tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
#         print(f"GPUデバイス index={gpu_index} のみが使用されます。")
#     except RuntimeError as e:
#         # set_visible_devicesは初期化前に呼び出す必要があるため、
#         # 既に初期化済みだとRuntimeErrorになる場合がある
#         print(e)

# ###############################################################################
# # 2) GPUメモリ状況を表示する関数 (pynvml使用)
# ###############################################################################
# def print_gpu_memory_usage(gpu_index=0):
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
#     mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     total_mb = mem_info.total / 1024**2
#     used_mb  = mem_info.used  / 1024**2
#     free_mb  = mem_info.free  / 1024**2
#     print(f"[Before Game] GPU Memory (index={gpu_index}):")
#     print(f"  Total: {total_mb:.2f} MB")
#     print(f"  Used : {used_mb:.2f} MB")
#     print(f"  Free : {free_mb:.2f} MB\n")
#     pynvml.nvmlShutdown()

# ###############################################################################
# # グローバル変数（推論時間 & メモリ使用量記録）
# ###############################################################################
# model_inference_times = []
# model_memory_usages   = []

# mc_inference_times = []
# mc_memory_usages   = []

# # psutil で現在のプロセスを取得
# process = psutil.Process(os.getpid())

# ###############################################################################
# # 3) GPUを選択・設定
# ###############################################################################
# select_and_configure_gpu(gpu_index=0)  # 例: 0番目のGPUを使用

# ###############################################################################
# # モデルの読み込み & モデルサイズの設定
# ###############################################################################
# model = keras.models.load_model("data/model_large")
# model_size = "large"  # "small", "medium", "large" など

# ###############################################################################
# # オセロ盤の初期化
# ###############################################################################
# def init_board():
#     """
#     オセロ初期配置:
#       黒=+1, 白=-1, 空=0
#     """
#     board = np.zeros((8, 8), dtype=int)
#     board[3, 3] = board[4, 4] = -1  # 白
#     board[3, 4] = board[4, 3] =  1  # 黒
#     return board

# ###############################################################################
# # 石を置けるか判定
# ###############################################################################
# def can_put(board, row, col, is_black_turn):
#     if board[row, col] != 0:
#         return False

#     current_player = 1 if is_black_turn else -1
#     opponent = -current_player

#     directions = [
#         (-1, 0), (1, 0), (0, -1), (0, 1),
#         (-1, -1), (-1, 1), (1, -1), (1, 1)
#     ]
#     for dx, dy in directions:
#         x, y = row + dx, col + dy
#         found_opponent = False
#         while 0 <= x < 8 and 0 <= y < 8:
#             if board[x, y] == opponent:
#                 found_opponent = True
#             elif board[x, y] == current_player:
#                 if found_opponent:
#                     return True
#                 break
#             else:
#                 break
#             x += dx
#             y += dy
#     return False

# ###############################################################################
# # 石を置いて相手の石を反転
# ###############################################################################
# def put(board, row, col, is_black_turn):
#     player = 1 if is_black_turn else -1
#     board[row, col] = player

#     directions = [
#         (-1, 0), (1, 0), (0, -1), (0, 1),
#         (-1, -1), (-1, 1), (1, -1), (1, 1)
#     ]
#     for dx, dy in directions:
#         x, y = row + dx, col + dy
#         stones_to_flip = []
#         while 0 <= x < 8 and 0 <= y < 8 and board[x, y] == -player:
#             stones_to_flip.append((x, y))
#             x += dx
#             y += dy
#         # 挟めた場合のみ反転
#         if 0 <= x < 8 and 0 <= y < 8 and board[x, y] == player:
#             for fx, fy in stones_to_flip:
#                 board[fx, fy] = player

# ###############################################################################
# # 盤面表示
# ###############################################################################
# def print_board(board):
#     symbols = {1: "●", -1: "○", 0: "."}
#     for row in board:
#         print(" ".join(symbols[v] for v in row))
#     print()

# ###############################################################################
# # 石数を数えて勝敗を判定
# ###############################################################################
# def count_stones(board):
#     black = np.sum(board == 1)
#     white = np.sum(board == -1)
#     return black, white

# ###############################################################################
# # モデルAI (黒番) -- 1位を90%, 2位を10%で選択
# ###############################################################################
# class ModelAI:
#     def __init__(self, model, model_label="unknown"):
#         self.model = model
#         self.model_label = model_label  # "small","medium","large"など

#     def choose_move(self, board, is_black_turn):
#         valid_moves = [(r, c) for r in range(8) for c in range(8)
#                        if can_put(board, r, c, is_black_turn)]
#         if not valid_moves:
#             return None

#         board_input = np.zeros((1, 2, 8, 8), dtype='int8')
#         if is_black_turn:
#             # 黒番 → channel0=黒, channel1=白
#             board_input[0, 0] = (board == 1).astype('int8')
#             board_input[0, 1] = (board == -1).astype('int8')
#         else:
#             # 白番 → channel0=白, channel1=黒
#             board_input[0, 0] = (board == -1).astype('int8')
#             board_input[0, 1] = (board ==  1).astype('int8')

#         before_mem = process.memory_info().rss
#         start_time = time.time()

#         predictions = self.model.predict(board_input, verbose=0)[0]  # shape=(64,)

#         end_time = time.time()
#         after_mem = process.memory_info().rss
#         elapsed  = end_time - start_time
#         mem_used = after_mem - before_mem

#         global model_inference_times, model_memory_usages
#         model_inference_times.append(elapsed)
#         model_memory_usages.append(mem_used)

#         turn_label = "黒" if is_black_turn else "白"
#         print(f"[Model({self.model_label})AI - {turn_label}番] "
#               f"MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

#         move_probs = [predictions[r*8 + c] for (r,c) in valid_moves]
#         sorted_unique_probs = sorted(list(set(move_probs)), reverse=True)

#         if len(sorted_unique_probs) >= 2:
#             max_prob = sorted_unique_probs[0]
#             second_prob = sorted_unique_probs[1]
#         else:
#             max_prob = sorted_unique_probs[0]
#             second_prob = None

#         top1_candidates = [
#             (r, c) for (r, c) in valid_moves
#             if np.isclose(predictions[r*8 + c], max_prob)
#         ]
#         if second_prob is not None:
#             top2_candidates = [
#                 (r, c) for (r, c) in valid_moves
#                 if np.isclose(predictions[r*8 + c], second_prob)
#             ]
#         else:
#             top2_candidates = []

#         p = random.random()
#         if top2_candidates and p < 0.1:
#             return random.choice(top2_candidates)
#         else:
#             return random.choice(top1_candidates)


# ###############################################################################
# # モンテカルロAI (白番)
# ###############################################################################
# class MonteCarloAI:
#     def __init__(self, num_simulations=50):
#         self.num_simulations = num_simulations

#     def simulate_game(self, board, is_black_turn):
#         """1回のランダムプレイアウト: 黒>白=1, 白>黒=-1, 同数=0"""
#         simulated_board = board.copy()
#         turn = is_black_turn
#         while True:
#             valid_moves = [(r,c) for r in range(8) for c in range(8)
#                            if can_put(simulated_board, r, c, turn)]
#             if valid_moves:
#                 move = random.choice(valid_moves)
#                 put(simulated_board, move[0], move[1], turn)
#             else:
#                 # パス後、相手も置けなければ終了
#                 if not any(can_put(simulated_board, rr, cc, not turn)
#                            for rr in range(8) for cc in range(8)):
#                     break
#             turn = not turn

#         black, white = count_stones(simulated_board)
#         if black > white:
#             return 1
#         elif white > black:
#             return -1
#         else:
#             return 0

#     def choose_move(self, board, is_black_turn):
#         valid_moves = [(r,c) for r in range(8) for c in range(8)
#                        if can_put(board, r, c, is_black_turn)]
#         if not valid_moves:
#             return None

#         before_mem = process.memory_info().rss
#         start_time = time.time()

#         move_scores = {}
#         for move in valid_moves:
#             score = 0
#             # 仮にこのmoveを打った後でランダムプレイアウト
#             sim_board = board.copy()
#             put(sim_board, move[0], move[1], is_black_turn)

#             # その後の局面を self.num_simulations 回 ランダムプレイアウト
#             for _ in range(self.num_simulations):
#                 board_copy = sim_board.copy()
#                 result = self.simulate_game(board_copy, not is_black_turn)
#                 # 黒視点: result=1(黒勝), -1(白勝)
#                 # is_black_turn=False (白番)ならスコアを逆転
#                 score += result if is_black_turn else -result
#             move_scores[move] = score

#         end_time = time.time()
#         after_mem = process.memory_info().rss
#         elapsed  = end_time - start_time
#         mem_used = after_mem - before_mem

#         global mc_inference_times, mc_memory_usages
#         mc_inference_times.append(elapsed)
#         mc_memory_usages.append(mem_used)

#         print(f"[MonteCarloAI(白)] MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

#         best_move = max(move_scores, key=move_scores.get)
#         return best_move

# ###############################################################################
# # 対戦 (黒=ModelAI, 白=MonteCarloAI)
# ###############################################################################
# def play_one_game(model_ai, mc_ai):
#     board = init_board()
#     is_black_turn = True  # 先手=黒(ModelAI)

#     while True:
#         print_board(board)
#         if is_black_turn:
#             print("【黒番: ModelAI】")
#             move = model_ai.choose_move(board, True)
#         else:
#             print("【白番: MonteCarloAI】")
#             move = mc_ai.choose_move(board, False)

#         if move:
#             print(f"→ 着手 {move}\n")
#             put(board, move[0], move[1], is_black_turn)
#         else:
#             print("→ パス\n")
#             # パス後、相手も置けなければ終了
#             if not any(can_put(board, rr, cc, not is_black_turn) 
#                        for rr in range(8) for cc in range(8)):
#                 break

#         is_black_turn = not is_black_turn

#     print("===== 対局終了 =====")
#     print_board(board)
#     black, white = count_stones(board)
#     print(f"黒(ModelAI) = {black}, 白(MonteCarloAI) = {white}")

#     if black > white:
#         print("→ 黒(ModelAI)の勝利!\n")
#         return 1
#     elif white > black:
#         print("→ 白(MonteCarloAI)の勝利!\n")
#         return -1
#     else:
#         print("→ 引き分け\n")
#         return 0

# ###############################################################################
# # 複数回の対戦を行い結果表示
# ###############################################################################
# def simulate_games(num_games=10, num_simulations_mc=50):
#     global model_inference_times
#     global model_memory_usages
#     global mc_inference_times
#     global mc_memory_usages

#     # グローバルリスト初期化
#     model_inference_times = []
#     model_memory_usages   = []
#     mc_inference_times    = []
#     mc_memory_usages      = []

#     model_ai = ModelAI(model, model_label=model_size)
#     mc_ai    = MonteCarloAI(num_simulations=num_simulations_mc)

#     model_wins = 0
#     mc_wins    = 0
#     draws      = 0

#     for i in range(num_games):
#         print(f"=== Game {i+1}/{num_games} ===")
#         result = play_one_game(model_ai, mc_ai)
#         if result == 1:
#             model_wins += 1
#         elif result == -1:
#             mc_wins += 1
#         else:
#             draws += 1

#     # 最終結果表示
#     print(f"\n===== 対戦結果({num_games}回) =====")
#     print(f"ModelAI(黒) 勝利: {model_wins}")
#     print(f"MonteCarloAI(白) 勝利: {mc_wins}")
#     print(f"引き分け            : {draws}")

#     # ModelAI(黒)の推論統計
#     if model_inference_times:
#         avg_time_model = np.mean(model_inference_times)
#         max_time_model = np.max(model_inference_times)
#         avg_mem_model  = np.mean(model_memory_usages)
#         max_mem_model  = np.max(model_memory_usages)
#         print(f"\n--- Model({model_size})AI(黒)の推論統計 ---")
#         print(f"推論時間(秒): 平均={avg_time_model:.6f}, 最大={max_time_model:.6f}")
#         print(f"メモリ使用量(bytes): 平均={avg_mem_model:.0f}, 最大={max_mem_model:.0f}")
#     else:
#         print(f"\nModel({model_size})AI(黒)は1回も推論しませんでした。")

#     # MonteCarloAI(白)の統計
#     if mc_inference_times:
#         avg_time_mc = np.mean(mc_inference_times)
#         max_time_mc = np.max(mc_inference_times)
#         avg_mem_mc  = np.mean(mc_memory_usages)
#         max_mem_mc  = np.max(mc_memory_usages)
#         print(f"\n--- MonteCarloAI(白)のシミュレーション統計 ---")
#         print(f"シミュレーション時間(秒): 平均={avg_time_mc:.6f}, 最大={max_time_mc:.6f}")
#         print(f"メモリ使用量(bytes): 平均={avg_mem_mc:.0f}, 最大={max_mem_mc:.0f}")
#     else:
#         print("\nMonteCarloAI(白)は1回もシミュレーションしませんでした。")

# ###############################################################################
# # メイン: 例として 500回対戦 & MonteCarloAIのシミュ数=100
# ###############################################################################
# if __name__ == "__main__":
#     print_gpu_memory_usage(gpu_index=0)  # GPUのメモリ状況を表示
#     simulate_games(num_games=500, num_simulations_mc=100)


