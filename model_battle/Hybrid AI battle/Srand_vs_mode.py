import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import psutil
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
# グローバル変数: 推論時間 & メモリ使用量を記録 (モデルAIが使う場合のみ)
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
# 4) モデルAI用クラス群 (Keras / TFLite)
###############################################################################
class KerasModelAI:
    def __init__(self, model, label="keras_unknown"):
        """
        Kerasモデルを使うAI (黒番/白番いずれにも使える)
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
            # 黒番
            board_input[0, 0] = (board ==  1).astype('int8')  # 黒
            board_input[0, 1] = (board == -1).astype('int8')  # 白
        else:
            # 白番
            board_input[0, 0] = (board == -1).astype('int8')  # 白
            board_input[0, 1] = (board ==  1).astype('int8')  # 黒

        # 推論前後のメモリ＆時間を計測
        before_mem = process.memory_info().rss
        start_time = time.time()

        # モデル推論
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

class TFLiteModelAI:
    def __init__(self, tflite_path, label="tflite_unknown"):
        """
        TFLite (量子化済み)モデルを使うAI
        tflite_path: .tflite ファイルへのパス
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

        # ダイナミックレンジ量子化想定: float32 入力が多い
        board_input = np.zeros((1, 2, 8, 8), dtype=np.float32)
        if is_black_turn:
            # 黒番
            board_input[0, 0] = (board ==  1).astype(np.float32)
            board_input[0, 1] = (board == -1).astype(np.float32)
        else:
            # 白番
            board_input[0, 0] = (board == -1).astype(np.float32)
            board_input[0, 1] = (board ==  1).astype(np.float32)

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

        # 確率上位2手を 90%:10% で選択
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
# 5) 黒番(ランダムAI)用クラス
###############################################################################
class RandomAI:
    def choose_move(self, board, is_black_turn):
        valid_moves = [(r, c) for r in range(8) for c in range(8)
                       if can_put(board, r, c, is_black_turn)]
        if not valid_moves:
            return None
        return random.choice(valid_moves)

###############################################################################
# 6) MinimaxAI (白用) : my_color = -1
###############################################################################
class MinimaxAIWhite:
    def __init__(self, depth=3):
        """
        depth : 探索の深さ
        白番 (my_color=-1) として動作
        """
        self.depth = depth
        self.my_color = -1

        # コーナーや辺を優遇する簡易ウェイト例
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
        return f"MinimaxWhite(depth={self.depth})"

    def choose_move(self, board, is_black_turn):
        # 白番でなければパス
        if is_black_turn:
            return None

        valid_moves = [(r,c) for r in range(8) for c in range(8)
                       if can_put(board, r, c, is_black_turn)]
        if not valid_moves:
            return None

        best_val = float('-inf')
        best_move = None
        for mv in valid_moves:
            new_bd = board.copy()
            put(new_bd, mv[0], mv[1], is_black_turn)  # 白番で着手
            val = self.minimax(new_bd, self.depth-1, not is_black_turn)
            if val > best_val:
                best_val = val
                best_move = mv

        print(f"[MinimaxAIWhite] best_eval={best_val:.2f}")
        return best_move

    def minimax(self, board, depth, is_black_turn):
        # 終了 or depth=0
        if depth == 0 or np.all(board != 0):
            return self.evaluate_position(board)

        current_player = 1 if is_black_turn else -1
        valid_moves = [
            (r, c) for r in range(8) for c in range(8)
            if can_put(board, r, c, is_black_turn)
        ]

        # =============== 修正: パス時の両者パス判定を追加 ===============
        if not valid_moves:
            # 相手の合法手もあるか確認
            opponent_valid = [
                (r, c) for r in range(8) for c in range(8)
                if can_put(board, r, c, not is_black_turn)
            ]
            if not opponent_valid:
                # 両者連続パス => 終局評価
                return self.evaluate_position(board)
            else:
                # 自分だけパス (depth据え置きで相手へ)
                return self.minimax(board, depth, not is_black_turn)
        # =============================================================

        if current_player == self.my_color:
            # 自分(白)が最大化
            value = float('-inf')
            for mv in valid_moves:
                new_bd = board.copy()
                put(new_bd, mv[0], mv[1], is_black_turn)
                value = max(value, self.minimax(new_bd, depth-1, not is_black_turn))
            return value
        else:
            # 相手(黒)が最小化
            value = float('inf')
            for mv in valid_moves:
                new_bd = board.copy()
                put(new_bd, mv[0], mv[1], is_black_turn)
                value = min(value, self.minimax(new_bd, depth-1, not is_black_turn))
            return value

    def evaluate_position(self, board):
        me   = self.my_color     # -1(白)
        opp  = -me               # +1(黒)
        score_me = np.sum((board == me)  * self.weights)
        score_op = np.sum((board == opp) * self.weights)
        return score_me - score_op

###############################################################################
# 7) HybridWhiteAI (白番): 残り○マス以下で MinimaxAIWhite を使用
###############################################################################
class HybridWhiteAI:
    def __init__(self, model_ai, minimax_ai, threshold=15):
        """
        model_ai  : KerasModelAI または TFLiteModelAI
        minimax_ai: MinimaxAIWhite インスタンス
        threshold : 残り空きマスがこの数以下になったらminimaxへ切り替え
        """
        self.model_ai  = model_ai
        self.minimax_ai= minimax_ai
        self.threshold = threshold

    def choose_move(self, board, is_black_turn):
        # 白番の手番のみ動作 (is_black_turn=False)
        if is_black_turn:
            return None

        empty_count = np.count_nonzero(board == 0)
        if empty_count <= self.threshold:
            # 終盤 → Minimaxに切り替え
            print(f"[HybridWhiteAI] 残り空きマス={empty_count} <= {self.threshold} ⇒ MinimaxAIWhiteに切り替え")
            return self.minimax_ai.choose_move(board, is_black_turn)
        else:
            # それ以外はモデルAIを使用
            return self.model_ai.choose_move(board, is_black_turn)

###############################################################################
# 8) 1ゲーム実行: 黒(RandomAI) vs. 白(HybridWhiteAI)
###############################################################################
def play_one_game(black_ai, white_ai):
    board = init_board()
    is_black_turn = True  # 先手=黒(RandomAI)

    while True:
        print_board(board)
        if is_black_turn:
            print("【黒番: RandomAI】")
            move = black_ai.choose_move(board, True)
        else:
            print("【白番: HybridWhiteAI】")
            move = white_ai.choose_move(board, False)

        if move:
            print(f"  → 着手: {move}\n")
            put(board, move[0], move[1], is_black_turn)
        else:
            print("  → パス\n")
            # パス後、相手もパスなら終了
            if not any(can_put(board, rr, cc, not is_black_turn) for rr in range(8) for cc in range(8)):
                break
        is_black_turn = not is_black_turn

    # 対局終了
    print("===== 対局終了 =====")
    print_board(board)

    black, white = count_stones(board)
    print(f"黒(RandomAI) = {black}, 白(HybridWhiteAI) = {white}")

    if black > white:
        print("→ 黒(RandomAI)の勝利!\n")
        return -1
    elif white > black:
        print("→ 白(HybridWhiteAI)の勝利!\n")
        return 1
    else:
        print("→ 引き分け\n")
        return 0

###############################################################################
# 9) 複数回対戦し、結果 & 推論統計を表示
###############################################################################
def simulate_games(black_ai, white_ai, num_games=10):
    global inference_times
    global memory_usages
    inference_times = []
    memory_usages   = []

    white_wins = 0
    black_wins = 0
    draws      = 0

    for i in range(num_games):
        print(f"\n=== Game {i+1}/{num_games} ===")
        result = play_one_game(black_ai, white_ai)
        if result == 1:
            white_wins += 1
        elif result == -1:
            black_wins += 1
        else:
            draws += 1

    # 結果表示
    print(f"\n===== 対戦結果({num_games}回) =====")
    print(f"黒(RandomAI) 勝利: {black_wins}")
    print(f"白(HybridWhiteAI) 勝利: {white_wins}")
    print(f"引き分け             : {draws}")

    # 推論統計 (モデル部分のみ計測)
    if inference_times:
        avg_time = np.mean(inference_times)
        max_time = np.max(inference_times)
        avg_mem  = np.mean(memory_usages)
        max_mem  = np.max(memory_usages)
        print("\n--- (HybridWhiteAI内) モデル推論の統計 ---")
        print(f"推論時間(秒): 平均={avg_time:.6f}, 最大={max_time:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem:.0f}, 最大={max_mem:.0f}")
    else:
        print("\n(HybridWhiteAI)モデルAIの推論が行われませんでした。")

###############################################################################
# 10) メイン
###############################################################################
if __name__ == "__main__":
    # (任意) GPU 設定
    select_and_configure_gpu(gpu_index=0)
    print_gpu_memory_usage(gpu_index=0)

    # 1) モデルを用意 (Keras か TFLite)
    keras_model = keras.models.load_model("data/model_small")
    model_ai    = KerasModelAI(keras_model, label="small")

    # tflite_path = "model_small_dynamic_range.tflite"
    # model_ai    = TFLiteModelAI(tflite_path, label="small_quant")

    # 2) MinimaxAI (白用) を用意
    minimax_white = MinimaxAIWhite(depth=5)

    # 3) HybridWhiteAI: 残り15マス以下 → MinimaxAIWhite、それ以外はモデル推論
    white_ai = HybridWhiteAI(model_ai, minimax_white, threshold=15)

    # 4) 黒番はランダムAI
    black_ai = RandomAI()

    # 5) 複数回対戦
    simulate_games(black_ai, white_ai, num_games=500)


