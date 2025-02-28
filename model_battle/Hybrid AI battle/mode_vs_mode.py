import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import psutil
import time
import os

###############################################################################
# GPUの割り当てを制御（必要なGPUデバイスだけを見えるように設定 + メモリ成長を有効化）
###############################################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # ここでは最初のGPUだけを可視化する例
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        
        # 選んだGPUに対してメモリ成長を有効化
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # メモリ成長はプログラム起動時に設定する必要があるため、
        # すでに初期化されている場合はエラーになる可能性があります。
        print(e)

###############################################################################
# グローバル変数: 推論時間・メモリを「黒モデル」「白モデル」別々に記録
###############################################################################
black_inference_times = []
black_memory_usages   = []

white_inference_times = []
white_memory_usages   = []

# psutil で現在のプロセスを取得（メモリ計測に使用）
process = psutil.Process(os.getpid())

###############################################################################
# 2つのモデルをロード
#   先攻(黒)モデル -> data/model_medium (CNNパートに使用)
#   後攻(白)モデル -> data/model_large
# さらにモデルサイズを表す変数を導入
###############################################################################
black_model_size = "small"
white_model_size = "medium"

black_cnn_model = keras.models.load_model('data/model_small')  # 黒番用のCNN
white_model     = keras.models.load_model('data/model_small')   # 白番用

###############################################################################
# オセロ初期化
###############################################################################
def init_board():
    board = np.zeros((8, 8), dtype=int)
    # 黒 = +1, 白 = -1
    board[3, 3] = board[4, 4] = -1  # 白
    board[3, 4] = board[4, 3] =  1  # 黒
    return board

###############################################################################
# 石が置けるかどうかの判定
###############################################################################
def can_put(board, row, col, is_black_turn):
    if board[row, col] != 0:
        return False

    player = 1 if is_black_turn else -1
    opponent = -player

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
            elif board[x, y] == player:
                # 間に相手がいれば置ける
                if found_opponent:
                    return True
                break
            else:
                break
            x += dx
            y += dy
    return False

###############################################################################
# 石を置いて、挟んだ相手の石を反転
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
        # 最後に自分の石があれば反転
        if 0 <= x < 8 and 0 <= y < 8 and board[x, y] == player:
            for fx, fy in stones_to_flip:
                board[fx, fy] = player

###############################################################################
# 盤面表示（デバッグ用）
###############################################################################
def print_board(board):
    symbols = {1: "●", -1: "○", 0: "."}
    for row in board:
        print(" ".join(symbols[v] for v in row))
    print()

###############################################################################
# 黒・白の石数をカウント
###############################################################################
def count_stones(board):
    black = np.sum(board == 1)
    white = np.sum(board == -1)
    return black, white

###############################################################################
# CNNモデルで着手を選ぶ関数 (黒or白兼用)
#   - 予測上位2位を取り、90%で1位・10%で2位からランダム選択
#   - 推論時間 & メモリ使用量をグローバル記録に加える
###############################################################################
def model_move(board, model, is_black_turn, model_size):
    valid_moves = [(r, c) for r in range(8) for c in range(8)
                   if can_put(board, r, c, is_black_turn)]
    if not valid_moves:
        return None

    # (1,2,8,8)形式の入力作成
    board_input = np.zeros((1, 2, 8, 8), dtype='int8')
    if is_black_turn:
        board_input[0, 0] = (board == 1).astype('int8')   # 黒石
        board_input[0, 1] = (board == -1).astype('int8')  # 白石
    else:
        board_input[0, 0] = (board == -1).astype('int8')  # 白石
        board_input[0, 1] = (board == 1).astype('int8')   # 黒石

    # 推論前後でメモリ・時間を測定
    before_mem = process.memory_info().rss
    start_time = time.time()

    predictions = model.predict(board_input, verbose=0)[0]  # shape=(64,)

    end_time = time.time()
    after_mem = process.memory_info().rss

    mem_used = after_mem - before_mem
    elapsed = end_time - start_time

    # 黒番・白番で記録先を切り替え
    global black_inference_times, black_memory_usages
    global white_inference_times, white_memory_usages
    if is_black_turn:
        black_inference_times.append(elapsed)
        black_memory_usages.append(mem_used)
    else:
        white_inference_times.append(elapsed)
        white_memory_usages.append(mem_used)

    # valid_moves に対応する予測値を取得
    move_probs = [predictions[r * 8 + c] for (r, c) in valid_moves]

    # 1位の確率値と2位の確率値を distinct に抽出
    sorted_unique_probs = sorted(list(set(move_probs)), reverse=True)
    max_prob = sorted_unique_probs[0]
    if len(sorted_unique_probs) >= 2:
        second_prob = sorted_unique_probs[1]
    else:
        second_prob = None

    # 1位候補、2位候補をそれぞれリストアップ
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

    # 90%で1位の手、10%で2位の手
    p = random.random()
    if top2_candidates and p < 0.1:
        chosen_move = random.choice(top2_candidates)
    else:
        chosen_move = random.choice(top1_candidates)

    return chosen_move

###############################################################################
# Minimax のクラス定義 (終盤で切り替えるため)
###############################################################################
class MinimaxAI:
    def __init__(self, depth=5, my_color=1):
        """
        depth : 探索の深さ
        my_color : +1(黒) または -1(白)
        """
        self.depth = depth
        self.my_color = my_color

        # 簡易的なウェイト例 (コーナー重視など)
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

    def evaluate_position(self, board):
        me = self.my_color
        opp = -me
        score_me = np.sum((board == me)  * self.weights)
        score_op = np.sum((board == opp) * self.weights)
        return score_me - score_op

    def minimax(self, board, depth, is_black_turn):
        # 終了 or depth=0
        if depth == 0 or np.all(board != 0):
            return self.evaluate_position(board)

        current_player = 1 if is_black_turn else -1
        valid_moves = [
            (r, c) for r in range(8) for c in range(8)
            if can_put(board, r, c, is_black_turn)
        ]

        if not valid_moves:
            # 両者連続パスなら終了
            opponent_moves = [
                (r, c) for r in range(8) for c in range(8)
                if can_put(board, r, c, not is_black_turn)
            ]
            if not opponent_moves:
                return self.evaluate_position(board)
            else:
                # 自分だけ打てないならパスして続行（depth消費しない）
                return self.minimax(board, depth, not is_black_turn)

        if current_player == self.my_color:
            # 最大化
            value = float('-inf')
            for mv in valid_moves:
                new_board = board.copy()
                put(new_board, mv[0], mv[1], is_black_turn)
                value = max(value, self.minimax(new_board, depth-1, not is_black_turn))
            return value
        else:
            # 最小化
            value = float('inf')
            for mv in valid_moves:
                new_board = board.copy()
                put(new_board, mv[0], mv[1], is_black_turn)
                value = min(value, self.minimax(new_board, depth-1, not is_black_turn))
            return value

    def choose_move(self, board, is_black_turn):
        # 先手=黒向けにmy_color=+1で生成した想定
        if not is_black_turn:  # 黒番でなければ動かない
            return None

        valid_moves = [
            (r, c) for r in range(8) for c in range(8)
            if can_put(board, r, c, is_black_turn)
        ]
        if not valid_moves:
            return None

        best_val = float('-inf')
        best_move = None
        for mv in valid_moves:
            new_bd = board.copy()
            put(new_bd, mv[0], mv[1], is_black_turn)
            val = self.minimax(new_bd, self.depth-1, not is_black_turn)
            if val > best_val:
                best_val = val
                best_move = mv
        return best_move

###############################################################################
# ハイブリッド黒番AIクラス
#   - 空きマスが threshold より多い間はCNNモデル、少なくなったらMinimax
###############################################################################
class HybridBlackAI:
    def __init__(self, cnn_model, minimax_ai, threshold=15):
        self.cnn_model = cnn_model
        self.minimax_ai = minimax_ai
        self.threshold = threshold

    def choose_move(self, board, is_black_turn):
        # 黒番でないときはパス扱い
        if not is_black_turn:
            return None

        empty_count = np.count_nonzero(board == 0)
        if empty_count <= self.threshold:
            # 終盤 → Minimax で打つ
            return self.minimax_ai.choose_move(board, is_black_turn)
        else:
            # 中盤までは CNNモデルで打つ
            return model_move(board, self.cnn_model, True, "hybrid(CNN)")

###############################################################################
# 黒モデル vs 白モデルの対局関数
#   - 黒番は HybridBlackAI
#   - 白番は通常の CNN モデル
###############################################################################
def play_game_model_vs_model(black_ai, white_model, show_board=False):
    board = init_board()
    is_black_turn = True

    while True:
        if show_board:
            print_board(board)
            if is_black_turn:
                print("【黒番(Hybrid)】")
            else:
                print("【白番(CNN)】")

        # 現手番に合法手があるか
        valid_exist = any(can_put(board, r, c, is_black_turn)
                          for r in range(8) for c in range(8))
        if not valid_exist:
            # 相手にもなければ終了
            other_exist = any(can_put(board, r, c, not is_black_turn)
                              for r in range(8) for c in range(8))
            if not other_exist:
                break
            # パス
            if show_board:
                print("  → パス\n")
            is_black_turn = not is_black_turn
            continue

        # 着手
        if is_black_turn:
            move = black_ai.choose_move(board, True)
        else:
            # 従来の白番モデル（CNN）のみ
            move = model_move(board, white_model, False, white_model_size)

        if move is not None:
            if show_board:
                print(f"  → 着手 {move}\n")
            put(board, move[0], move[1], is_black_turn)
        else:
            # パス
            if show_board:
                print("  → パス\n")

        is_black_turn = not is_black_turn

    # 終局
    if show_board:
        print("===== 対局終了 =====")
        print_board(board)

    black, white = count_stones(board)
    if show_board:
        print(f"黒(Hybrid) = {black}, 白(CNN) = {white}")

    return board

###############################################################################
# 複数試合をシミュレート
###############################################################################
def simulate_model_vs_model(black_ai, white_model, num_games=100, show_board=False):
    black_wins = 0
    white_wins = 0
    draws = 0

    for i in range(num_games):
        if show_board:
            print(f"=== Game {i+1}/{num_games} ===")

        final_board = play_game_model_vs_model(
            black_ai,
            white_model,
            show_board=show_board
        )
        b, w = count_stones(final_board)
        if b > w:
            black_wins += 1
        elif w > b:
            white_wins += 1
        else:
            draws += 1
        
        if show_board:
            print("----------------------------------------------")

    return black_wins, white_wins, draws

###############################################################################
# メイン例
###############################################################################
if __name__ == "__main__":
    # 黒番AI: CNN + 終盤 Minimax
    minimax_for_black = MinimaxAI(depth=6, my_color=1)  # 黒 = +1
    black_ai = HybridBlackAI(
        cnn_model=black_cnn_model,
        minimax_ai=minimax_for_black,
        threshold=15 # 空きマスが15以下になったらMinimax
    )

    num_simulations = 500

    bw, ww, dw = simulate_model_vs_model(
        black_ai,
        white_model,
        num_games=num_simulations,
        show_board=True  # Falseにすると盤面表示を省略
    )

    print(f"\n{num_simulations} 回対戦結果: (先攻= HybridBlack, 後攻= {white_model_size})")
    print(f"  黒(HybridBlack) 勝利数: {bw}")
    print(f"  白({white_model_size}) 勝利数: {ww}")
    print(f"  引き分け               : {dw}")

    ############################################################################
    # メモリ消費量 & 推論時間の統計を表示 (CNN部分のみ集計)
    ############################################################################
    import numpy as np

    def print_statistics(times, mems, label):
        if len(times) == 0:
            print(f"[{label}] 推論機会がありませんでした。")
            return
        avg_time = np.mean(times)
        max_time = np.max(times)
        avg_mem = np.mean(mems)
        max_mem = np.max(mems)
        print(f"\n--- {label} ---")
        print(f"推論時間(秒): 平均={avg_time:.6f}, 最大={max_time:.6f}")
        print(f"メモリ使用量(bytes): 平均={avg_mem:.0f}, 最大={max_mem:.0f}")

    print_statistics(black_inference_times, black_memory_usages, "黒(Hybrid)のCNN推論")
    print_statistics(white_inference_times, white_memory_usages, f"白モデル ({white_model_size})")


