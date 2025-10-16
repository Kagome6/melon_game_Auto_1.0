# visualize_log.py
import re
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 設定 ---
LOG_FILE_PATH = 'training_log.txt'  # 読み込むログファイル名
MOVING_AVERAGE_WINDOW = 50          # 移動平均を計算するためのウィンドウサイズ（少し長めに設定）

def parse_log_file(file_path):
    """
    指定されたログファイルを解析し、学習データを抽出する。
    """
    # データを格納するための辞書
    data = {
        'episodes': [],
        'scores': [],
        'rewards': [],
        'epsilons': [],
        'losses': [],
        'steps': []
    }

    # 新しいログ形式に対応した正規表現パターン
    # 例: "Episode: 1/5000, Score: 0, TotalReward: -20.00, Epsilon: 1.00, Loss: 0.0000, Steps: 2"
    log_pattern = re.compile(
        r"Episode: (\d+)/\d+, "
        r"Score: (\d+), "
        r"TotalReward: ([-\d.]+), "
        r"Epsilon: ([\d.]+), "
        r"Loss: ([\d.eE+-]+), "  # 指数表記 (例: 1.23e-04) にも対応
        r"Steps: (\d+)"
    )

    print(f"ログファイル '{file_path}' を解析しています...")
    
    if not os.path.exists(file_path):
        print(f"エラー: ログファイル '{file_path}' が見つかりません。")
        print("ヒント: `python your_game_script.py > training_log.txt` のようにリダイレクトしてログを生成してください。")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                # 正規表現でキャプチャした各グループを対応するリストに追加
                data['episodes'].append(int(match.group(1)))
                data['scores'].append(int(match.group(2)))
                data['rewards'].append(float(match.group(3)))
                data['epsilons'].append(float(match.group(4)))
                data['losses'].append(float(match.group(5)))
                data['steps'].append(int(match.group(6)))

    if not data['episodes']:
        print("エラー: ログファイルから有効なデータが見つかりませんでした。")
        print("ログの形式とスクリプト内の正規表現が一致しているか確認してください。")
        return None
    
    print(f"解析完了: {len(data['episodes'])} エピソード分のデータを読み込みました。")
    return data

def plot_training_progress(data):
    """
    解析されたデータを受け取り、学習の進捗をグラフ化する。
    """
    if not data:
        return

    episodes = data['episodes']

    # 移動平均を計算（データがウィンドウサイズより少ない場合は計算しない）
    def calculate_moving_average(values, window):
        if len(values) >= window:
            return np.convolve(values, np.ones(window)/window, mode='valid')
        return None

    scores_ma = calculate_moving_average(data['scores'], MOVING_AVERAGE_WINDOW)
    rewards_ma = calculate_moving_average(data['rewards'], MOVING_AVERAGE_WINDOW)
    losses_ma = calculate_moving_average(data['losses'], MOVING_AVERAGE_WINDOW)

    # --- グラフ描画 ---
    # 4つのグラフを縦に並べて表示
    fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
    fig.suptitle('AI Training Progress Analysis', fontsize=18)

    # 1. スコア (Score)
    ax1 = axes[0]
    ax1.plot(episodes, data['scores'], 'b.', alpha=0.2, label='Raw Score')
    if scores_ma is not None:
        ax1.plot(episodes[MOVING_AVERAGE_WINDOW-1:], scores_ma, 'b-', linewidth=2, label=f'{MOVING_AVERAGE_WINDOW}-ep Moving Avg')
    ax1.set_ylabel('Score')
    ax1.set_title('Score per Episode')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. 合計報酬 (Total Reward)
    ax2 = axes[1]
    ax2.plot(episodes, data['rewards'], 'g.', alpha=0.2, label='Raw Total Reward')
    if rewards_ma is not None:
        ax2.plot(episodes[MOVING_AVERAGE_WINDOW-1:], rewards_ma, 'g-', linewidth=2, label=f'{MOVING_AVERAGE_WINDOW}-ep Moving Avg')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Total Reward per Episode')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. 損失 (Loss)
    ax3 = axes[2]
    ax3.plot(episodes, data['losses'], 'm.', alpha=0.2, label='Raw Loss')
    if losses_ma is not None:
        ax3.plot(episodes[MOVING_AVERAGE_WINDOW-1:], losses_ma, 'm-', linewidth=2, label=f'{MOVING_AVERAGE_WINDOW}-ep Moving Avg')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss per Episode')
    # 損失は急激に変化するため、対数スケールで表示すると傾向が見やすい
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, which='both', linestyle='--', alpha=0.6)

    # 4. イプシロン (Epsilon)
    ax4 = axes[3]
    ax4.plot(episodes, data['epsilons'], 'r-', linewidth=2.5, label='Epsilon (Exploration Rate)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Epsilon')
    ax4.set_title('Epsilon Decay')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # タイトルとの重なりを防ぐ
    plt.show()

if __name__ == '__main__':
    # ログファイルを解析し、データを取得
    training_data = parse_log_file(LOG_FILE_PATH)
    
    # データをグラフ化
    plot_training_progress(training_data)