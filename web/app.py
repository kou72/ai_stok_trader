"""
株価予測モデル学習のWebインターフェース
"""
import os
import sys
import subprocess
import glob
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import threading
import queue

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)

# 実行状態を管理
training_status = {
    'is_running': False,
    'current_log': [],
    'start_time': None,
    'end_time': None,
    'result_dir': None
}

log_queue = queue.Queue()


def get_csv_directories():
    """
    data/csv_* ディレクトリのリストを取得

    Returns:
    --------
    list : ディレクトリパスのリスト
    """
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    csv_dirs = glob.glob(os.path.join(base_path, 'csv_*'))
    csv_dirs = [os.path.basename(d) for d in csv_dirs if os.path.isdir(d)]
    csv_dirs.sort(reverse=True)  # 新しい順に並べる
    return csv_dirs


def run_training(csv_path):
    """
    学習処理を別スレッドで実行

    Parameters:
    -----------
    csv_path : str
        CSVディレクトリのパス
    """
    global training_status

    try:
        training_status['is_running'] = True
        training_status['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        training_status['current_log'] = []
        training_status['end_time'] = None
        training_status['result_dir'] = None

        # プロジェクトルートディレクトリを取得
        project_root = os.path.join(os.path.dirname(__file__), '..')
        main_py_path = os.path.join(project_root, 'src', 'main.py')

        # Pythonコマンドを実行
        cmd = [sys.executable, main_py_path, '--csv', csv_path]

        log_queue.put(f"[{training_status['start_time']}] 学習を開始します...")
        log_queue.put(f"コマンド: {' '.join(cmd)}\n")

        # プロセスを起動
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=project_root,
            universal_newlines=True
        )

        # 出力をリアルタイムで読み取る
        for line in process.stdout:
            line = line.rstrip()
            if line:
                training_status['current_log'].append(line)
                log_queue.put(line)

        # プロセスの終了を待つ
        process.wait()

        training_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if process.returncode == 0:
            log_queue.put(f"\n[{training_status['end_time']}] 学習が正常に完了しました！")

            # 最新のresultディレクトリを取得
            result_base = os.path.join(project_root, 'result')
            result_dirs = glob.glob(os.path.join(result_base, '[0-9]*'))
            if result_dirs:
                latest_result = max(result_dirs, key=os.path.getctime)
                training_status['result_dir'] = os.path.basename(latest_result)
                log_queue.put(f"結果保存先: result/{training_status['result_dir']}")
        else:
            log_queue.put(f"\n[{training_status['end_time']}] エラー: 学習が失敗しました（終了コード: {process.returncode}）")

    except Exception as e:
        training_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_queue.put(f"\n[{training_status['end_time']}] エラー: {str(e)}")

    finally:
        training_status['is_running'] = False


@app.route('/')
def index():
    """
    トップページ
    """
    csv_dirs = get_csv_directories()
    return render_template('index.html', csv_dirs=csv_dirs)


@app.route('/start_training', methods=['POST'])
def start_training():
    """
    学習を開始
    """
    if training_status['is_running']:
        return jsonify({'error': '既に学習が実行中です'}), 400

    csv_dir = request.json.get('csv_dir')
    if not csv_dir:
        return jsonify({'error': 'CSVディレクトリを選択してください'}), 400

    csv_path = os.path.join('data', csv_dir)

    # 学習を別スレッドで実行
    thread = threading.Thread(target=run_training, args=(csv_path,))
    thread.daemon = True
    thread.start()

    return jsonify({'message': '学習を開始しました', 'status': 'started'})


@app.route('/status')
def status():
    """
    学習の状態を取得
    """
    # 新しいログを取得
    new_logs = []
    while not log_queue.empty():
        try:
            new_logs.append(log_queue.get_nowait())
        except queue.Empty:
            break

    return jsonify({
        'is_running': training_status['is_running'],
        'start_time': training_status['start_time'],
        'end_time': training_status['end_time'],
        'result_dir': training_status['result_dir'],
        'new_logs': new_logs
    })


@app.route('/logs')
def logs():
    """
    全ログを取得
    """
    return jsonify({
        'logs': training_status['current_log']
    })


if __name__ == '__main__':
    print("=" * 80)
    print("株価予測モデル学習 Webインターフェース")
    print("=" * 80)
    print("\nブラウザで以下のURLにアクセスしてください:")
    print("  http://localhost:5000")
    print("\n終了するには Ctrl+C を押してください")
    print("=" * 80)

    app.run(debug=True, host='0.0.0.0', port=5000)
