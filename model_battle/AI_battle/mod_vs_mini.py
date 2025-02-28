import random
import numpy as np
import psutil
import time
import os
import pynvml
import tensorflow as tf
from tensorflow import keras

###############################################################################
# 1) GPU割り当ての設定 (省略可)
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
# 2) GPUメモリ状況を表示する関数 (pynvml使用) -- デバッグ用
###############################################################################
def print_gpu_memory_usage(gpu_index=0):
    """GPUメモリの使用状況を表示"""
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
model_inference_times   = []
model_memory_usages     = []
minimax_inference_times = []
minimax_memory_usages   = []

# psutil で現在のプロセスを取得
process = psutil.Process(os.getpid())

###############################################################################
# (必要なら) GPUを選択・設定
###############################################################################
select_and_configure_gpu(gpu_index=0)

###############################################################################
# オセロ盤関連の基本関数
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

def print_board(board):
    symbols = {1: "●", -1: "○", 0: "."}
    for row in board:
        print(" ".join(symbols[v] for v in row))
    print()

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
    """盤面に石を置き、反転処理も行う"""
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

def count_stones(board):
    black = np.sum(board == 1)
    white = np.sum(board == -1)
    return black, white

###############################################################################
# 1) ModelAI (Keras) - 予測上位2手から 90% / 10% で選択
###############################################################################
class ModelAI:
    def __init__(self, model, model_label="unknown"):
        self.model = model
        self.model_label = model_label

    def choose_move(self, board, is_black_turn):
        valid_moves = [
            (r, c) for r in range(8) for c in range(8)
            if can_put(board, r, c, is_black_turn)
        ]
        if not valid_moves:
            return None

        # NN入力データ shape=(1,2,8,8)  (int8想定)
        board_input = np.zeros((1, 2, 8, 8), dtype='int8')
        if is_black_turn:
            board_input[0, 0] = (board ==  1).astype('int8')  # 黒
            board_input[0, 1] = (board == -1).astype('int8')  # 白
        else:
            board_input[0, 0] = (board == -1).astype('int8')  # 白
            board_input[0, 1] = (board ==  1).astype('int8')  # 黒

        # CPUメモリ & 時間 計測
        before_mem = process.memory_info().rss
        start_time = time.time()

        predictions = self.model.predict(board_input, verbose=0)[0]  # shape=(64,)

        end_time = time.time()
        after_mem = process.memory_info().rss
        elapsed  = end_time - start_time
        mem_used = after_mem - before_mem

        model_inference_times.append(elapsed)
        model_memory_usages.append(mem_used)

        turn_label = "黒" if is_black_turn else "白"
        print(f"[Model({self.model_label})AI - {turn_label}番] "
              f"MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

        # 上位2候補から 90% / 10% で着手
        move_probs = [predictions[r*8 + c] for (r, c) in valid_moves]
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
# 2) TFLiteModelAI (量子化後モデル) - 同様に上位2手から 90% / 10% で選択
###############################################################################
class TFLiteModelAI:
    def __init__(self, tflite_path, model_label="unknown"):
        self.model_label = model_label

        # TFLite Interpreter をロード
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def choose_move(self, board, is_black_turn):
        valid_moves = [
            (r, c) for r in range(8) for c in range(8)
            if can_put(board, r, c, is_black_turn)
        ]
        if not valid_moves:
            return None

        # ダイナミックレンジ量子化モデルなら float32 入力の場合が多い
        board_input = np.zeros((1, 2, 8, 8), dtype=np.float32)
        if is_black_turn:
            board_input[0, 0] = (board ==  1).astype(np.float32)
            board_input[0, 1] = (board == -1).astype(np.float32)
        else:
            board_input[0, 0] = (board == -1).astype(np.float32)
            board_input[0, 1] = (board ==  1).astype(np.float32)

        before_mem = process.memory_info().rss
        start_time = time.time()

        # TFLite推論
        self.interpreter.set_tensor(self.input_details[0]['index'], board_input)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0]  # shape=(64,)

        end_time = time.time()
        after_mem = process.memory_info().rss
        elapsed = end_time - start_time
        mem_used = after_mem - before_mem

        model_inference_times.append(elapsed)
        model_memory_usages.append(mem_used)

        turn_label = "黒" if is_black_turn else "白"
        print(f"[TFLiteModel({self.model_label})AI - {turn_label}番] "
              f"MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

        # 上位2候補から 90% / 10% で着手 (ロジックはKeras版と同じ)
        move_probs = [predictions[r*8 + c] for (r, c) in valid_moves]
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
# 3) 複合評価 + MoveOrdering + 簡易オープニングブックを導入した MinimaxAI (白番)
###############################################################################
class MinimaxAI:
    def __init__(self, depth=2):
        self.depth = depth

        # 位置評価用のテーブル
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

        # 簡易オープニングブック
        self.opening_book = {
            # 例: まだ1手も打たれていない段階で検索
            (): (2, 3),
            ((2,3),): (2,2),
        }
        self.move_history = []

    def get_phase(self, board):
        """盤面の埋まり具合で序盤/中盤/終盤をざっくり判定"""
        num_stones = np.count_nonzero(board)
        if num_stones < 20:
            return "early"
        elif num_stones < 50:
            return "mid"
        else:
            return "late"

    def count_mobility(self, board, is_black):
        moves = 0
        for r in range(8):
            for c in range(8):
                if board[r,c] == 0:
                    if can_put(board, r, c, is_black):
                        moves += 1
        return moves

    def count_stable(self, board, is_black):
        """簡易的に角・辺を安定石とみなす"""
        val = 1 if is_black else -1
        stable = 0
        corners = [(0,0),(0,7),(7,0),(7,7)]
        for (rr,cc) in corners:
            if board[rr,cc] == val:
                stable += 2
        for c in range(8):
            if board[0,c] == val:
                stable += 1
            if board[7,c] == val:
                stable += 1
        for r in range(8):
            if board[r,0] == val:
                stable += 1
            if board[r,7] == val:
                stable += 1
        return stable

    def evaluate_complex(self, board, is_black_turn):
        """複合評価(黒視点)"""
        black_score = np.sum((board == 1 ) * self.weights)
        white_score = np.sum((board == -1) * self.weights)
        pos_eval = black_score - white_score

        black_mob = self.count_mobility(board, True)
        white_mob = self.count_mobility(board, False)
        mob_eval = black_mob - white_mob

        black_stable = self.count_stable(board, True)
        white_stable = self.count_stable(board, False)
        stable_eval = black_stable - white_stable

        black_count, white_count = count_stones(board)
        disc_eval = black_count - white_count

        phase = self.get_phase(board)
        if phase == "early":
            score = 0.3*pos_eval + 3.0*mob_eval + 1.0*stable_eval
        elif phase == "mid":
            score = 0.7*pos_eval + 2.0*mob_eval + 2.0*stable_eval
        else:
            score = 1.0*pos_eval + 1.0*mob_eval + 4.0*stable_eval + 2.0*disc_eval

        # 黒視点 → 白番なら符号反転
        return score if is_black_turn else -score

    def minimax(self, board, depth, is_black_turn, alpha, beta):
        if depth == 0 or np.all(board != 0):
            return self.evaluate_complex(board, is_black_turn), None

        valid_moves = [
            (r, c) for r in range(8) for c in range(8)
            if can_put(board, r, c, is_black_turn)
        ]
        if not valid_moves:
            # 相手も置けなければ終局
            if not any(can_put(board, rr, cc, not is_black_turn) for rr in range(8) for cc in range(8)):
                return self.evaluate_complex(board, is_black_turn), None
            # パス
            score_pass, _ = self.minimax(board, depth, not is_black_turn, alpha, beta)
            return score_pass, None

        # MoveOrdering用の簡易評価
        move_evals = []
        for mv in valid_moves:
            new_bd = board.copy()
            put(new_bd, mv[0], mv[1], is_black_turn)
            sc = self.evaluate_complex(new_bd, True)  # 黒視点
            move_evals.append((sc, mv))

        # ソート(黒=降順, 白=昇順)
        if is_black_turn:
            move_evals.sort(key=lambda x: x[0], reverse=True)
        else:
            move_evals.sort(key=lambda x: x[0])

        if is_black_turn:  # Max層
            max_eval = float('-inf')
            best_move = None
            for (mv_sc, mv) in move_evals:
                new_bd = board.copy()
                put(new_bd, mv[0], mv[1], True)
                eval_val, _ = self.minimax(new_bd, depth-1, False, alpha, beta)
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = mv
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:  # Min層
            min_eval = float('inf')
            best_move = None
            for (mv_sc, mv) in move_evals:
                new_bd = board.copy()
                put(new_bd, mv[0], mv[1], False)
                eval_val, _ = self.minimax(new_bd, depth-1, True, alpha, beta)
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = mv
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def choose_move(self, board, is_black_turn):
        # 開始数手はオープニングブック参照
        if len(self.move_history) < 4:
            key = tuple(self.move_history)
            if key in self.opening_book:
                next_move = self.opening_book[key]
                print(f"[OpeningBook] => {next_move}")
                return next_move

        before_mem = process.memory_info().rss
        start_time = time.time()

        eval_val, move = self.minimax(board, self.depth, is_black_turn, float('-inf'), float('inf'))

        end_time = time.time()
        after_mem = process.memory_info().rss
        elapsed = end_time - start_time
        mem_used = after_mem - before_mem

        minimax_inference_times.append(elapsed)
        minimax_memory_usages.append(mem_used)

        who = "黒" if is_black_turn else "白"
        print(f"[MinimaxAI+Enhanced(depth={self.depth}) - {who}番] "
              f"MemUsed={mem_used} bytes, Time={elapsed:.6f} s (eval={eval_val:.1f})")
        return move

###############################################################################
# 対戦 (黒=任意のモデルAI, 白=MinimaxAI)
###############################################################################
def play_one_game(model_ai, minimax_ai):
    board = init_board()
    is_black_turn = True  # 先手=黒(ModelAI or TFLiteModelAI)
    minimax_ai.move_history = []  # オープニングブック用に履歴をクリア

    while True:
        print_board(board)
        if is_black_turn:
            print(f"【黒番: Model({model_ai.model_label})AI】")
            move = model_ai.choose_move(board, True)
            if move:
                minimax_ai.move_history.append(move)  # 黒が打った手を記録
        else:
            print(f"【白番: MinimaxAI+Enhanced(depth={minimax_ai.depth})】")
            move = minimax_ai.choose_move(board, False)
            if move:
                minimax_ai.move_history.append(move)

        if move:
            print(f"→ 着手 {move}\n")
            put(board, move[0], move[1], is_black_turn)
        else:
            print("→ パス\n")
            # パス後、相手も打てないなら終了
            if not any(can_put(board, rr, cc, not is_black_turn) for rr in range(8) for cc in range(8)):
                break

        is_black_turn = not is_black_turn

    # 対局終了
    print("===== 対局終了 =====")
    print_board(board)
    black, white = count_stones(board)
    print(f"黒(Model({model_ai.model_label})) = {black}, 白(MinimaxAI+Enhanced) = {white}")

    if black > white:
        print("→ 黒(Model)の勝利!\n")
        return 1
    elif white > black:
        print("→ 白(MinimaxAI+Enhanced)の勝利!\n")
        return -1
    else:
        print("→ 引き分け\n")
        return 0

###############################################################################
# 複数回対戦 & 統計表示
###############################################################################
def simulate_games(model_ai, minimax_ai, num_games=10):
    global model_inference_times
    global model_memory_usages
    global minimax_inference_times
    global minimax_memory_usages

    model_inference_times   = []
    model_memory_usages     = []
    minimax_inference_times = []
    minimax_memory_usages   = []

    model_wins   = 0
    minimax_wins = 0
    draws        = 0

    for i in range(num_games):
        print(f"=== Game {i+1}/{num_games} ===")
        result = play_one_game(model_ai, minimax_ai)
        if result == 1:
            model_wins += 1
        elif result == -1:
            minimax_wins += 1
        else:
            draws += 1

    print(f"\n===== 対戦結果({num_games}回, depth={minimax_ai.depth}) =====")
    print(f"黒(ModelAI) 勝利: {model_wins}")
    print(f"白(MinimaxAI+Enhanced) 勝利: {minimax_wins}")
    print(f"引き分け               : {draws}")

    # ModelAI(黒) の推論統計
    if model_inference_times:
        avg_time_model = np.mean(model_inference_times)
        max_time_model = np.max(model_inference_times)
        avg_mem_model  = np.mean(model_memory_usages)
        max_mem_model  = np.max(model_memory_usages)
        print(f"\n--- Model({model_ai.model_label})AI(黒)の推論統計 ---")
        print(f"推論時間(秒): 平均={avg_time_model:.6f}, 最大={max_time_model:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem_model:.0f}, 最大={max_mem_model:.0f}")
    else:
        print(f"\nModel({model_ai.model_label})AI(黒)は1回も推論しませんでした。")

    # MinimaxAI(白) の探索統計
    if minimax_inference_times:
        avg_time_mini = np.mean(minimax_inference_times)
        max_time_mini = np.max(minimax_inference_times)
        avg_mem_mini  = np.mean(minimax_memory_usages)
        max_mem_mini  = np.max(minimax_memory_usages)
        print(f"\n--- MinimaxAI+Enhanced(白)の探索統計 ---")
        print(f"探索時間(秒): 平均={avg_time_mini:.6f}, 最大={max_time_mini:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem_mini:.0f}, 最大={max_mem_mini:.0f}")
    else:
        print("\nMinimaxAI(白)は1回も探索を実行しませんでした。")

###############################################################################
# メイン
###############################################################################
if __name__ == "__main__":
    print_gpu_memory_usage(gpu_index=0)

    # --- 1) Kerasモデルを使う場合 ---
    # model_keras = keras.models.load_model("data/model_large")  # 例
    # black_ai = ModelAI(model_keras, model_label="large")

    # --- 2) TFLite量子化モデルを使う場合 ---
    #   (例) ダイナミックレンジ量子化のファイル
    # black_ai = TFLiteModelAI("model_small_dynamic_range.tflite", model_label="small_quant")
    # black_ai = TFLiteModelAI("model_medium_dynamic_range.tflite", model_label="medium_quant")
    # black_ai = TFLiteModelAI("model_large_dynamic_range.tflite", model_label="large_quant")

    # ※どちらかを有効にして使ってください
    #  ここではKerasモデルを例示
    # model_keras = keras.models.load_model("data/model_small")
    # black_ai = ModelAI(model_keras, model_label=" small")
    # model_keras = keras.models.load_model("data/model_medium")
    # black_ai = ModelAI(model_keras, model_label=" medium")
    model_keras = keras.models.load_model("data/model_large")
    black_ai = ModelAI(model_keras, model_label=" large")

    # MinimaxAI (白)
    white_ai = MinimaxAI(depth=5)

    # 対戦開始 (例: 10回対戦)
    simulate_games(black_ai, white_ai, num_games=500)



#  import random
# import numpy as np
# import psutil
# import time
# import os
# import pynvml
# import tensorflow as tf
# from tensorflow import keras

# ###############################################################################
# # 1) GPU割り当ての設定 (省略可)
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
#         tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
#         tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
#         print(f"GPUデバイス index={gpu_index} のみが使用されます。")
#     except RuntimeError as e:
#         print(e)

# ###############################################################################
# # 2) GPUメモリ状況を表示する関数 (pynvml使用) -- デバッグ用
# ###############################################################################
# def print_gpu_memory_usage(gpu_index=0):
#     """GPUメモリの使用状況を表示"""
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
# model_inference_times   = []
# model_memory_usages     = []
# minimax_inference_times = []
# minimax_memory_usages   = []

# # psutil で現在のプロセスを取得
# process = psutil.Process(os.getpid())

# ###############################################################################
# # 3) GPUを選択・設定
# ###############################################################################
# select_and_configure_gpu(gpu_index=0)

# ###############################################################################
# # 学習済みモデルの読み込み & モデルサイズ
# ###############################################################################
# model = keras.models.load_model("data/model_large")  # 事前学習済みモデルファイル
# model_size = "large"  # "small", "medium", "large" etc.

# ###############################################################################
# # オセロ盤関連の基本関数
# ###############################################################################
# def init_board():
#     """
#     オセロの初期配置:
#       黒=+1, 白=-1, 空=0
#     """
#     board = np.zeros((8, 8), dtype=int)
#     board[3, 3] = board[4, 4] = -1  # 白
#     board[3, 4] = board[4, 3] =  1  # 黒
#     return board

# def print_board(board):
#     symbols = {1: "●", -1: "○", 0: "."}
#     for row in board:
#         print(" ".join(symbols[v] for v in row))
#     print()

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

# def put(board, row, col, is_black_turn):
#     """盤面に石を置き、反転処理も行う"""
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

# def count_stones(board):
#     black = np.sum(board == 1)
#     white = np.sum(board == -1)
#     return black, white

# ###############################################################################
# # ModelAI (黒番) : 予測で上位2手から90%/10%で着手
# ###############################################################################
# class ModelAI:
#     def __init__(self, model, model_label="unknown"):
#         self.model = model
#         self.model_label = model_label

#     def choose_move(self, board, is_black_turn):
#         valid_moves = [
#             (r, c) for r in range(8) for c in range(8)
#             if can_put(board, r, c, is_black_turn)
#         ]
#         if not valid_moves:
#             return None

#         # NN入力データ shape=(1,2,8,8)
#         board_input = np.zeros((1, 2, 8, 8), dtype='int8')
#         if is_black_turn:
#             board_input[0, 0] = (board ==  1).astype('int8')  # 黒
#             board_input[0, 1] = (board == -1).astype('int8')  # 白
#         else:
#             board_input[0, 0] = (board == -1).astype('int8')  # 白
#             board_input[0, 1] = (board ==  1).astype('int8')  # 黒

#         # CPUメモリ & 時間 計測
#         before_mem = process.memory_info().rss
#         start_time = time.time()

#         predictions = self.model.predict(board_input, verbose=0)[0]  # shape=(64,)

#         end_time = time.time()
#         after_mem = process.memory_info().rss
#         elapsed  = end_time - start_time
#         mem_used = after_mem - before_mem

#         model_inference_times.append(elapsed)
#         model_memory_usages.append(mem_used)

#         turn_label = "黒" if is_black_turn else "白"
#         print(f"[Model({self.model_label})AI - {turn_label}番] MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

#         # 上位2候補から 90% / 10% で着手
#         move_probs = [predictions[r*8 + c] for (r, c) in valid_moves]
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
# # 複合評価 + MoveOrdering + 簡易オープニングブックを導入した MinimaxAI
# ###############################################################################
# class MinimaxAI:
#     def __init__(self, depth=2):
#         self.depth = depth

#         # 位置評価用のテーブル
#         self.weights = np.array([
#             [100, -20,  10,   5,   5,  10, -20, 100],
#             [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
#             [ 10,  -2,  -1,  -1,  -1,  -1,  -2,  10],
#             [  5,  -2,  -1,  -1,  -1,  -1,  -2,   5],
#             [  5,  -2,  -1,  -1,  -1,  -1,  -2,   5],
#             [ 10,  -2,  -1,  -1,  -1,  -1,  -2,  10],
#             [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
#             [100, -20,  10,   5,   5,  10, -20, 100]
#         ], dtype=float)

#         # 簡易オープニングブック (サンプル)
#         # 序盤数手分だけ定石を入れてみる（実際には大規模に増やす）
#         self.opening_book = {
#             # 例: まだ1手も打たれていない段階で検索
#             (): (2, 3),  # 「何も打たれてない」状態なら (2,3)へ打つ
#             # 例: 1手目 (2,3) が打たれた後は (2,2)へ打つ
#             ((2,3),): (2,2),
#         }
#         # 盤面の手履歴を追跡用
#         self.move_history = []

#     def get_phase(self, board):
#         """盤面の埋まり具合で序盤/中盤/終盤をざっくり判定"""
#         num_stones = np.count_nonzero(board)
#         if num_stones < 20:
#             return "early"
#         elif num_stones < 50:
#             return "mid"
#         else:
#             return "late"

#     def count_mobility(self, board, is_black):
#         """is_blackプレイヤーの合法手数"""
#         val = 1 if is_black else -1
#         moves = 0
#         for r in range(8):
#             for c in range(8):
#                 if board[r,c] == 0:
#                     # 簡易チェック
#                     if can_put(board, r, c, is_black):
#                         moves += 1
#         return moves

#     def count_stable(self, board, is_black):
#         """
#         簡易的に「角や辺の石」を安定石とみなす。
#         実際にはもっと厳密な判定が必要。
#         """
#         val = 1 if is_black else -1
#         stable = 0

#         # 角4つ
#         corners = [(0,0),(0,7),(7,0),(7,7)]
#         for (rr,cc) in corners:
#             if board[rr,cc] == val:
#                 stable += 2  # 角は重めにカウント

#         # 辺
#         for c in range(8):
#             if board[0,c] == val:
#                 stable += 1
#             if board[7,c] == val:
#                 stable += 1
#         for r in range(8):
#             if board[r,0] == val:
#                 stable += 1
#             if board[r,7] == val:
#                 stable += 1

#         return stable

#     def evaluate_complex(self, board, is_black_turn):
#         """
#         複合評価関数 (常に黒視点の値を返す):
#         - 位置重み (self.weights)
#         - mobility (合法手数)
#         - 安定石数
#         - 盤面フェーズ(序盤/中盤/終盤)で重みづけ変更
#         """
#         black_score = np.sum((board ==  1) * self.weights)
#         white_score = np.sum((board == -1) * self.weights)
#         pos_eval = black_score - white_score  # 黒視点

#         # mob(黒), mob(白)
#         black_mob = self.count_mobility(board, True)
#         white_mob = self.count_mobility(board, False)
#         mob_eval = black_mob - white_mob  # 黒視点

#         # 安定石数
#         black_stable = self.count_stable(board, True)
#         white_stable = self.count_stable(board, False)
#         stable_eval = black_stable - white_stable  # 黒視点

#         # 終盤は実石数を加味
#         black_count, white_count = count_stones(board)
#         disc_eval = black_count - white_count

#         phase = self.get_phase(board)
#         if phase == "early":
#             # 序盤: mobility重視
#             score = 0.3 * pos_eval + 3.0 * mob_eval + 1.0 * stable_eval
#         elif phase == "mid":
#             # 中盤: 位置重み + mobility + 安定石
#             score = 0.7 * pos_eval + 2.0 * mob_eval + 2.0 * stable_eval
#         else:
#             # 終盤: 実石数・安定石重視
#             score = 1.0 * pos_eval + 1.0 * mob_eval + 4.0 * stable_eval + 2.0 * disc_eval

#         # 黒視点スコア → 白視点なら符号反転
#         return score if is_black_turn else -score

#     def minimax(self, board, depth, is_black_turn, alpha, beta):
#         """複合評価 + MoveOrdering + α-β 探索"""
#         # 終端条件
#         if depth == 0 or np.all(board != 0):
#             return self.evaluate_complex(board, is_black_turn), None

#         valid_moves = [
#             (r, c) for r in range(8) for c in range(8)
#             if can_put(board, r, c, is_black_turn)
#         ]
#         if not valid_moves:
#             # 相手も置けなければ終局
#             if not any(can_put(board, rr, cc, not is_black_turn) for rr in range(8) for cc in range(8)):
#                 return self.evaluate_complex(board, is_black_turn), None
#             # パス
#             score_pass, _ = self.minimax(board, depth, not is_black_turn, alpha, beta)
#             return score_pass, None

#         # MoveOrdering: 1手打った後の複合評価でソート (黒=降順, 白=昇順)
#         move_evals = []
#         for move in valid_moves:
#             new_bd = board.copy()
#             put(new_bd, move[0], move[1], is_black_turn)
#             move_score = self.evaluate_complex(new_bd, True)  # とりあえず黒視点で
#             move_evals.append((move_score, move))

#         if is_black_turn:  # 黒番(Max層)
#             move_evals.sort(key=lambda x: x[0], reverse=True)
#         else:              # 白番(Min層)
#             move_evals.sort(key=lambda x: x[0])

#         if is_black_turn:
#             max_eval = float('-inf')
#             best_move = None
#             for (mv_sc, mv) in move_evals:
#                 new_bd = board.copy()
#                 put(new_bd, mv[0], mv[1], True)
#                 eval_val, _ = self.minimax(new_bd, depth - 1, False, alpha, beta)
#                 if eval_val > max_eval:
#                     max_eval = eval_val
#                     best_move = mv
#                 alpha = max(alpha, eval_val)
#                 if beta <= alpha:
#                     break
#             return max_eval, best_move
#         else:
#             min_eval = float('inf')
#             best_move = None
#             for (mv_sc, mv) in move_evals:
#                 new_bd = board.copy()
#                 put(new_bd, mv[0], mv[1], False)
#                 eval_val, _ = self.minimax(new_bd, depth - 1, True, alpha, beta)
#                 if eval_val < min_eval:
#                     min_eval = eval_val
#                     best_move = mv
#                 beta = min(beta, eval_val)
#                 if beta <= alpha:
#                     break
#             return min_eval, best_move

#     def choose_move(self, board, is_black_turn):
#         """
#         オープニングブック優先 → なければミニマックス探索
#         """
#         # オープニングブック参照（序盤だけ）
#         if len(self.move_history) < 4:  # 例: 最初の4手まで
#             # tuple化して参照
#             key = tuple(self.move_history)
#             if key in self.opening_book:
#                 next_move = self.opening_book[key]
#                 print(f"[OpeningBook] => {next_move}")
#                 return next_move

#         # ミニマックス探索
#         before_mem = process.memory_info().rss
#         start_time = time.time()

#         eval_val, move = self.minimax(board, self.depth, is_black_turn, float('-inf'), float('inf'))

#         end_time = time.time()
#         after_mem = process.memory_info().rss
#         elapsed  = end_time - start_time
#         mem_used = after_mem - before_mem

#         minimax_inference_times.append(elapsed)
#         minimax_memory_usages.append(mem_used)

#         who = "黒" if is_black_turn else "白"
#         print(f"[MinimaxAI+Enhanced(depth={self.depth}) - {who}番] MemUsed={mem_used} bytes, Time={elapsed:.6f} s (eval={eval_val:.1f})")
#         return move

# ###############################################################################
# # 対戦 (黒=ModelAI, 白=MinimaxAI)
# ###############################################################################
# def play_one_game(model_ai, minimax_ai):
#     board = init_board()
#     is_black_turn = True  # 先手=黒(ModelAI)
#     minimax_ai.move_history = []  # オープニングブック用に履歴をクリア

#     while True:
#         print_board(board)

#         if is_black_turn:
#             print(f"【黒番: Model({model_ai.model_label})AI】")
#             move = model_ai.choose_move(board, True)
#             if move:
#                 minimax_ai.move_history.append(move)  # 相手が打った手も履歴に追加
#         else:
#             print(f"【白番: MinimaxAI+Enhanced(depth={minimax_ai.depth})】")
#             move = minimax_ai.choose_move(board, False)
#             if move:
#                 minimax_ai.move_history.append(move)

#         if move:
#             print(f"→ 着手 {move}\n")
#             put(board, move[0], move[1], is_black_turn)
#         else:
#             print("→ パス\n")
#             # パスの後、相手も打てなければ終了
#             if not any(can_put(board, rr, cc, not is_black_turn) for rr in range(8) for cc in range(8)):
#                 break

#         is_black_turn = not is_black_turn

#     # 対局終了
#     print("===== 対局終了 =====")
#     print_board(board)
#     black, white = count_stones(board)
#     print(f"黒(Model({model_ai.model_label})) = {black}, 白(MinimaxAI+Enhanced) = {white}")

#     if black > white:
#         print("→ 黒(Model)の勝利!\n")
#         return 1
#     elif white > black:
#         print("→ 白(MinimaxAI+Enhanced)の勝利!\n")
#         return -1
#     else:
#         print("→ 引き分け\n")
#         return 0

# ###############################################################################
# # 複数回対戦 & 統計表示
# ###############################################################################
# def simulate_games(num_games=10, minimax_depth=2):
#     global model_inference_times
#     global model_memory_usages
#     global minimax_inference_times
#     global minimax_memory_usages

#     model_inference_times   = []
#     model_memory_usages     = []
#     minimax_inference_times = []
#     minimax_memory_usages   = []

#     # 黒=ModelAI, 白=MinimaxAI
#     model_ai   = ModelAI(model, model_label=model_size)
#     minimax_ai = MinimaxAI(depth=minimax_depth)

#     model_wins   = 0
#     minimax_wins = 0
#     draws        = 0

#     for i in range(num_games):
#         print(f"=== Game {i+1}/{num_games} ===")
#         result = play_one_game(model_ai, minimax_ai)
#         if result == 1:
#             model_wins += 1
#         elif result == -1:
#             minimax_wins += 1
#         else:
#             draws += 1

#     print(f"\n===== 対戦結果({num_games}回, depth={minimax_depth}) =====")
#     print(f"黒(ModelAI) 勝利: {model_wins}")
#     print(f"白(MinimaxAI+Enhanced) 勝利: {minimax_wins}")
#     print(f"引き分け               : {draws}")

#     # ModelAI(黒) の推論統計
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

#     # MinimaxAI(白) の探索統計
#     if minimax_inference_times:
#         avg_time_mini = np.mean(minimax_inference_times)
#         max_time_mini = np.max(minimax_inference_times)
#         avg_mem_mini  = np.mean(minimax_memory_usages)
#         max_mem_mini  = np.max(minimax_memory_usages)
#         print(f"\n--- MinimaxAI+Enhanced(白)の探索統計 ---")
#         print(f"探索時間(秒): 平均={avg_time_mini:.6f}, 最大={max_time_mini:.6f}")
#         print(f"メモリ使用量(bytes): 平均={avg_mem_mini:.0f}, 最大={max_mem_mini:.0f}")
#     else:
#         print("\nMinimaxAI(白)は1回も探索を実行しませんでした。")

# ###############################################################################
# # メイン (例: depth=5 で 500回対戦)
# ###############################################################################
# if __name__ == "__main__":
#     print_gpu_memory_usage(gpu_index=0)
#     simulate_games(num_games=500, minimax_depth=5)
    
