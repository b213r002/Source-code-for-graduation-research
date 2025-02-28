import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import psutil
import statistics
import time
import os

###############################################################################
# グローバル変数: 推論時間 & メモリ使用量を記録するリスト
###############################################################################
inference_times = []
memory_usages   = []

# psutil で現在のプロセスを取得しておく
process = psutil.Process(os.getpid())

###############################################################################
# 学習済みモデルの読み込み
###############################################################################
# 必要に応じてパスを変えてください
model = keras.models.load_model("data/model_small")
# ここでモデルサイズを示す変数を定義
model_size = "small"

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
    board[3, 4] = board[4, 3] = 1   # 黒
    return board

###############################################################################
# 石を置けるか判定
###############################################################################
def can_put(board, row, col, is_black_turn):
    if board[row, col] != 0:
        return False

    current_player = 1 if is_black_turn else -1
    opponent = -current_player

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
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
# 盤面表示用
###############################################################################
def print_board(board):
    for row in board:
        line = []
        for v in row:
            if v == 1:
                line.append("●")    # 黒
            elif v == -1:
                line.append("○")    # 白
            else:
                line.append(".")    # 空
        print(" ".join(line))
    print()

###############################################################################
# 学習モデルによる着手選択 (変更箇所)
###############################################################################
def model_move(board, model, is_black_turn):
    valid_moves = [(r, c) for r in range(8) for c in range(8)
                   if can_put(board, r, c, is_black_turn)]
    if not valid_moves:
        return None  # パス

    # (1,2,8,8) の入力
    board_input = np.zeros((1, 2, 8, 8), dtype='int8')
    if is_black_turn:
        # 黒番 → チャネル0=黒石, チャネル1=白石
        board_input[0, 0] = (board == 1).astype('int8')
        board_input[0, 1] = (board == -1).astype('int8')
    else:
        # 白番 → チャネル0=白石, チャネル1=黒石
        board_input[0, 0] = (board == -1).astype('int8')
        board_input[0, 1] = (board == 1).astype('int8')

    # --- 推論前後でメモリ＆時間を測定 ---
    before_mem = process.memory_info().rss
    start_time = time.time()

    predictions = model.predict(board_input, verbose=0)[0]  # shape=(64,)

    end_time = time.time()
    after_mem = process.memory_info().rss

    elapsed = end_time - start_time
    mem_used = after_mem - before_mem

    # グローバルリストへ記録
    inference_times.append(elapsed)
    memory_usages.append(mem_used)

    print(f"[model_move] MemUsed={mem_used} bytes, Time={elapsed:.6f} s")

    # 1) valid_moves の推論値を取得
    move_probs = [predictions[r*8 + c] for (r, c) in valid_moves]

    # 2) distinct な確率をソート (上位2つを取得)
    sorted_unique_probs = sorted(list(set(move_probs)), reverse=True)

    if len(sorted_unique_probs) >= 2:
        max_prob = sorted_unique_probs[0]
        second_prob = sorted_unique_probs[1]
    else:
        # 2位の値がない場合は second_prob = None
        max_prob = sorted_unique_probs[0]
        second_prob = None

    # 3) 1位の手、2位の手の候補を取得
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

    # 4) 90%の確率で1位の候補を選ぶ、10%の確率で2位の候補を選ぶ
    p = random.random()
    if top2_candidates and p < 0.1:
        # 10%の確率で2位の手から選ぶ
        return random.choice(top2_candidates)
    else:
        # それ以外(90%) or 2位手が無い場合は1位の手から選ぶ
        return random.choice(top1_candidates)

###############################################################################
# ランダムAI: 合法手の中からランダムに1つ
###############################################################################
def random_move(board, is_black_turn):
    valid_moves = [(r, c) for r in range(8) for c in range(8)
                   if can_put(board, r, c, is_black_turn)]
    if not valid_moves:
        return None
    return random.choice(valid_moves)

###############################################################################
# 石数を数えて勝敗を判定
###############################################################################
def count_stones(board):
    black = np.sum(board == 1)
    white = np.sum(board == -1)
    return black, white

###############################################################################
# 1ゲーム実行 (黒=ランダム, 白=モデル(large))
###############################################################################
def play_one_game(model):
    board = init_board()
    is_black_turn = True  # 先攻=黒

    while True:
        print_board(board)

        if is_black_turn:
            print("【黒番(ランダム)】")
            move = random_move(board, True)
        else:
            # モデルサイズを含めて表示
            print(f"【白番(モデル({model_size}))】")
            move = model_move(board, model, False)

        if move:
            print(f"  → {move}\n")
            put(board, move[0], move[1], is_black_turn)
        else:
            print("  → パス\n")
            # パスしたあと、相手も打てなければ終了
            if not any(can_put(board, r, c, not is_black_turn) for r in range(8) for c in range(8)):
                break

        is_black_turn = not is_black_turn

    print("===== 対局終了 =====")
    print_board(board)
    black, white = count_stones(board)
    # 勝敗表示もモデルサイズを記載
    print(f"黒(ランダム) = {black}, 白(モデル({model_size})) = {white}")

    if black > white:
        print("→ 黒(ランダム)の勝利！\n")
        return 1
    elif white > black:
        print(f"→ 白(モデル({model_size}))の勝利！\n")
        return -1
    else:
        print("→ 引き分け\n")
        return 0

###############################################################################
# 複数ゲームをシミュレートし、勝率および推論統計を表示
###############################################################################
def simulate_games(model, num_games=5):
    global inference_times
    global memory_usages
    inference_times = []
    memory_usages   = []

    model_wins  = 0
    random_wins = 0
    draws       = 0

    for i in range(num_games):
        print(f"=== Game {i+1}/{num_games} ===")
        result = play_one_game(model)
        if result == 1:
            random_wins += 1
        elif result == -1:
            model_wins += 1
        else:
            draws += 1
    
    print(f"\n==== {num_games} 回の対戦結果 ====")
    print(f"ランダムAI(黒) 勝利: {random_wins}")
    # モデルサイズを表示
    print(f"モデル({model_size})(白) 勝利: {model_wins}")
    print(f"引き分け                 : {draws}")

    # 推論統計 (1手あたり)
    times_arr = np.array(inference_times)
    mems_arr  = np.array(memory_usages)

    if len(times_arr) > 0:
        avg_time = np.mean(times_arr)
        max_time = np.max(times_arr)
        avg_mem  = np.mean(mems_arr)
        max_mem  = np.max(mems_arr)

        print("\n===== 推論時間 (秒) =====")
        print(f"平均: {avg_time:.6f} s")
        print(f"最大: {max_time:.6f} s")

        print("\n===== メモリ使用量 (bytes) =====")
        print(f"平均: {avg_mem:.0f} bytes")
        print(f"最大: {max_mem:.0f} bytes")
    else:
        print("推論が行われませんでした。")

###############################################################################
# メイン (例: 1000回の対戦を実施)
###############################################################################
simulate_games(model, num_games=500)
