// DOM要素の取得
const csvSelect = document.getElementById('csv-select');
const startBtn = document.getElementById('start-btn');
const clearLogBtn = document.getElementById('clear-log-btn');
const statusText = document.getElementById('status-text');
const startTime = document.getElementById('start-time');
const endTime = document.getElementById('end-time');
const resultDir = document.getElementById('result-dir');
const logOutput = document.getElementById('log-output');

// 状態管理
let statusInterval = null;
let isRunning = false;

// 学習開始ボタンのクリックイベント
startBtn.addEventListener('click', async () => {
    const selectedCsv = csvSelect.value;

    if (!selectedCsv) {
        alert('CSVディレクトリを選択してください');
        return;
    }

    if (isRunning) {
        alert('既に学習が実行中です');
        return;
    }

    try {
        const response = await fetch('/start_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                csv_dir: selectedCsv
            })
        });

        const data = await response.json();

        if (response.ok) {
            // ステータスの監視を開始
            startStatusPolling();
        } else {
            alert(`エラー: ${data.error}`);
        }
    } catch (error) {
        alert(`エラー: ${error.message}`);
    }
});

// ログクリアボタンのクリックイベント
clearLogBtn.addEventListener('click', () => {
    logOutput.textContent = 'ログがクリアされました...';
});

// ステータスの監視開始
function startStatusPolling() {
    if (statusInterval) {
        return; // 既に監視中
    }

    statusInterval = setInterval(async () => {
        try {
            const response = await fetch('/status');
            const data = await response.json();

            updateUI(data);

            // 学習が終了したら監視を停止
            if (!data.is_running && isRunning) {
                stopStatusPolling();
            }

            isRunning = data.is_running;

        } catch (error) {
            console.error('ステータス取得エラー:', error);
        }
    }, 1000); // 1秒ごとにポーリング
}

// ステータスの監視停止
function stopStatusPolling() {
    if (statusInterval) {
        clearInterval(statusInterval);
        statusInterval = null;
    }
}

// UIの更新
function updateUI(data) {
    // 状態表示の更新
    if (data.is_running) {
        statusText.textContent = '実行中';
        statusText.className = 'status-value status-running';
        startBtn.disabled = true;
        startBtn.textContent = '実行中...';
    } else if (data.end_time) {
        if (data.result_dir) {
            statusText.textContent = '完了';
            statusText.className = 'status-value status-completed';
        } else {
            statusText.textContent = 'エラー';
            statusText.className = 'status-value status-error';
        }
        startBtn.disabled = false;
        startBtn.textContent = '学習開始';
    } else {
        statusText.textContent = '待機中';
        statusText.className = 'status-value';
        startBtn.disabled = false;
        startBtn.textContent = '学習開始';
    }

    // 時刻の更新
    startTime.textContent = data.start_time || '-';
    endTime.textContent = data.end_time || '-';
    resultDir.textContent = data.result_dir ? `result/${data.result_dir}` : '-';

    // ログの更新
    if (data.new_logs && data.new_logs.length > 0) {
        const currentLog = logOutput.textContent;
        const newLogText = data.new_logs.join('\n');

        if (currentLog === 'ログが表示されます...' || currentLog === 'ログがクリアされました...') {
            logOutput.textContent = newLogText;
        } else {
            logOutput.textContent = currentLog + '\n' + newLogText;
        }

        // ログコンテナを最下部にスクロール
        const logContainer = document.getElementById('log-container');
        logContainer.scrollTop = logContainer.scrollHeight;
    }
}

// ページ読み込み時にステータスを取得
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        updateUI(data);

        // 実行中の場合は監視を開始
        if (data.is_running) {
            isRunning = true;
            startStatusPolling();
        }
    } catch (error) {
        console.error('初期ステータス取得エラー:', error);
    }
});

// ページを離れるときに監視を停止
window.addEventListener('beforeunload', () => {
    stopStatusPolling();
});
