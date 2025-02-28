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
#   先攻(黒)モデル -> data/model_small
#   後攻(白)モデル -> data/model_medium
# さらにモデルサイズを表す変数を導入
###############################################################################
black_model_size = "medium"
white_model_size = "large"
black_model = keras.models.load_model('data/model_medium')
white_model = keras.models.load_model('data/model_large')

###############################################################################
# オセロ初期化
###############################################################################
def init_board():
    board = np.zeros((8, 8), dtype=int)
    # 黒 = +1, 白 = -1
    board[3, 3] = board[4, 4] = -1  # 白
    board[3, 4] = board[4, 3] = 1   # 黒
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
# 盤面を表示（デバッグ用）
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
# モデルに着手を選ばせる関数
#   ・推論時間＆メモリを「黒用 or 白用」リストに分けて記録
#   ・モデルの出力上位2位を取り、90%で1位・10%で2位を選ぶ。 
#     さらに1位(2位)が複数手ある場合はランダムに選択。
###############################################################################
def model_move(board, model, is_black_turn, model_size):
    valid_moves = [(r, c) for r in range(8) for c in range(8)
                   if can_put(board, r, c, is_black_turn)]
    if not valid_moves:
        return None

    # (1,2,8,8)形式の入力作成
    board_input = np.zeros((1, 2, 8, 8), dtype='int8')
    if is_black_turn:
        # 黒番：チャネル0=黒石, チャネル1=白石
        board_input[0, 0] = (board == 1).astype('int8')
        board_input[0, 1] = (board == -1).astype('int8')
    else:
        # 白番：チャネル0=白石, チャネル1=黒石
        board_input[0, 0] = (board == -1).astype('int8')
        board_input[0, 1] = (board == 1).astype('int8')

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

    # 90%で1位の手(複数あればランダム)、10%で2位の手(複数あればランダム)
    p = random.random()
    if top2_candidates and p < 0.1:
        chosen_move = random.choice(top2_candidates)
    else:
        chosen_move = random.choice(top1_candidates)

    return chosen_move

###############################################################################
# 黒モデル vs 白モデル の対局 (1ゲーム) -- 盤面を表示して着手を確認
###############################################################################
def play_game_model_vs_model(black_model, white_model, 
                             black_model_size, white_model_size, 
                             show_board=False):
    """
    先攻(黒) : black_model
    後攻(白) : white_model
    """
    board = init_board()
    is_black_turn = True

    while True:
        # 毎ターンの盤面表示（show_board=True のとき）
        if show_board:
            print_board(board)
            if is_black_turn:
                print(f"【黒番(モデル:{black_model_size})】")
            else:
                print(f"【白番(モデル:{white_model_size})】")

        # 現在の手番で着手可能か？
        valid_exist = any(can_put(board, r, c, is_black_turn)
                          for r in range(8) for c in range(8))
        if not valid_exist:
            # 相手も打てなければ終了
            other_exist = any(can_put(board, r, c, not is_black_turn)
                              for r in range(8) for c in range(8))
            if not other_exist:
                break
            # パス
            if show_board:
                print("  → パス\n")
            is_black_turn = not is_black_turn
            continue

        # モデルで着手を決定
        if is_black_turn:
            move = model_move(board, black_model, True, black_model_size)
        else:
            move = model_move(board, white_model, False, white_model_size)

        if move is not None:
            if show_board:
                print(f"  → 着手 {move}\n")
            put(board, move[0], move[1], is_black_turn)
        else:
            # パス(通常ここには入らないが念のため)
            if show_board:
                print("  → パス\n")

        # 手番交代
        is_black_turn = not is_black_turn

    # 対局終了後の盤面・結果表示
    if show_board:
        print("===== 対局終了 =====")
        print_board(board)

    black, white = count_stones(board)
    if show_board:
        print(f"黒({black_model_size}) = {black}, 白({white_model_size}) = {white}")

    return board

###############################################################################
# 複数試合をシミュレート
###############################################################################
def simulate_model_vs_model(black_model, white_model, 
                            black_model_size, white_model_size,
                            num_games=100, show_board=False):
    """
    指定回数だけ 黒モデル vs 白モデル の対局を実行し、
    勝敗を集計して返す。
    """
    black_wins = 0
    white_wins = 0
    draws = 0

    for i in range(num_games):
        # 必要に応じて「対局番号」を表示
        if show_board:
            print(f"=== Game {i+1}/{num_games} ===")

        final_board = play_game_model_vs_model(
            black_model,
            white_model,
            black_model_size,
            white_model_size,
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
# メイン例 (50 回対戦 + 結果表示)
###############################################################################
if __name__ == "__main__":
    num_simulations = 500
    
    # show_board=True にすると、各手の盤面が表示されます
    bw, ww, dw = simulate_model_vs_model(
        black_model, white_model,
        black_model_size, white_model_size,
        num_games=num_simulations,
        show_board=True  # 対局途中の盤面を表示
    )

    print(f"\n{num_simulations} 回対戦結果: (先攻= {black_model_size}, 後攻= {white_model_size})")
    print(f"  黒({black_model_size}) 勝利数: {bw}")
    print(f"  白({white_model_size}) 勝利数: {ww}")
    print(f"  引き分け            : {dw}")

    ############################################################################
    # メモリ消費量 & 推論時間の統計を表示 (黒モデル・白モデル別)
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

    print_statistics(black_inference_times, black_memory_usages, f"黒モデル ({black_model_size})")
    print_statistics(white_inference_times, white_memory_usages, f"白モデル ({white_model_size})")


