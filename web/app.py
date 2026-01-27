"""
株価予測モデル学習のWebインターフェース
"""
import os
import sys
import subprocess
import glob
import json
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import threading

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from progress import ProgressManager

app = Flask(__name__, static_folder='static/dist', static_url_path='')

# 進捗ファイルのパス
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
PROGRESS_FILE = os.path.join(PROJECT_ROOT, 'progress', 'progress.json')

# 実行状態を管理
training_status = {
    'is_running': False,
    'start_time': None,
    'end_time': None,
    'result_dir': None
}


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
        training_status['end_time'] = None
        training_status['result_dir'] = None

        # 進捗ファイルをリセット
        progress_manager = ProgressManager(PROGRESS_FILE)
        progress_manager.reset()

        # プロジェクトルートディレクトリを取得
        project_root = os.path.join(os.path.dirname(__file__), '..')
        main_py_path = os.path.join(project_root, 'src', 'main.py')

        # Pythonコマンドを実行（進捗ファイルを指定）
        cmd = [sys.executable, main_py_path, '--csv', csv_path, '--progress', PROGRESS_FILE]

        # 環境変数を設定（UTF-8エンコーディングを強制）
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        # プロセスを起動
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=project_root,
            env=env
        )

        # プロセスの終了を待つ
        process.wait()

        training_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if process.returncode == 0:
            # 最新のresultディレクトリを取得
            result_base = os.path.join(project_root, 'result')
            result_dirs = glob.glob(os.path.join(result_base, '[0-9]*'))
            if result_dirs:
                latest_result = max(result_dirs, key=os.path.getctime)
                training_status['result_dir'] = os.path.basename(latest_result)

    except Exception as e:
        training_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"エラー: {str(e)}")

    finally:
        training_status['is_running'] = False


@app.route('/')
def index():
    """
    トップページ（Reactアプリを提供）
    """
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/csv_dirs')
def csv_dirs():
    """
    CSVディレクトリ一覧を取得
    """
    csv_directories = get_csv_directories()
    return jsonify({'csv_dirs': csv_directories})


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
    return jsonify({
        'is_running': training_status['is_running'],
        'start_time': training_status['start_time'],
        'end_time': training_status['end_time'],
        'result_dir': training_status['result_dir']
    })


@app.route('/progress')
def progress():
    """
    学習の進捗を取得
    """
    progress_data = ProgressManager.load(PROGRESS_FILE)
    return jsonify(progress_data)


@app.route('/api/results')
def get_results():
    """
    result/ 配下のディレクトリ一覧を取得
    """
    try:
        project_root = os.path.join(os.path.dirname(__file__), '..')
        result_base = os.path.join(project_root, 'result')

        if not os.path.exists(result_base):
            return jsonify({'results': []})

        result_dirs = glob.glob(os.path.join(result_base, '[0-9]*'))
        results = []

        for result_dir in result_dirs:
            if os.path.isdir(result_dir):
                result_id = os.path.basename(result_dir)

                # タイムスタンプから日時を生成
                timestamp = datetime.strptime(result_id, '%Y%m%d%H%M%S')
                formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')

                # config.json を読み込んで情報を取得
                config_path = os.path.join(result_dir, 'config.json')
                results_path = os.path.join(result_dir, 'results.json')

                config_data = {}
                results_data = {}

                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)

                if os.path.exists(results_path):
                    with open(results_path, 'r', encoding='utf-8') as f:
                        results_data = json.load(f)

                results.append({
                    'id': result_id,
                    'timestamp': formatted_time,
                    'config': config_data,
                    'results': results_data
                })

        # 新しい順にソート
        results.sort(key=lambda x: x['id'], reverse=True)

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/<result_id>')
def get_result_detail(result_id):
    """
    特定のresultディレクトリの詳細データを取得
    """
    try:
        project_root = os.path.join(os.path.dirname(__file__), '..')
        result_dir = os.path.join(project_root, 'result', result_id)

        if not os.path.exists(result_dir):
            return jsonify({'error': 'Result not found'}), 404

        # 各ファイルを読み込む
        config_path = os.path.join(result_dir, 'config.json')
        results_path = os.path.join(result_dir, 'results.json')
        time_log_path = os.path.join(result_dir, 'time_log.json')
        loss_history_path = os.path.join(result_dir, 'loss_history.csv')
        precision_history_path = os.path.join(result_dir, 'precision_history.csv')

        detail = {
            'id': result_id,
            'timestamp': datetime.strptime(result_id, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S'),
            'config': None,
            'results': None,
            'time_log': None,
            'loss_history': None,
            'precision_history': None
        }

        # config.json
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                detail['config'] = json.load(f)

        # results.json
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                detail['results'] = json.load(f)

        # time_log.json
        if os.path.exists(time_log_path):
            with open(time_log_path, 'r', encoding='utf-8') as f:
                detail['time_log'] = json.load(f)

        # loss_history.csv
        if os.path.exists(loss_history_path):
            import csv
            with open(loss_history_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                detail['loss_history'] = list(reader)

        # precision_history.csv
        if os.path.exists(precision_history_path):
            import csv
            with open(precision_history_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                detail['precision_history'] = list(reader)

        return jsonify(detail)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 80)
    print("株価予測モデル学習 Webインターフェース")
    print("=" * 80)
    print("\nブラウザで以下のURLにアクセスしてください:")
    print("  http://localhost:5000")
    print("\n終了するには Ctrl+C を押してください")
    print("=" * 80)

    app.run(debug=True, host='0.0.0.0', port=5000)
