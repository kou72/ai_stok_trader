import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'

function TrainingPage() {
  const [csvDirs, setCsvDirs] = useState([])
  const [selectedCsv, setSelectedCsv] = useState('')
  const [status, setStatus] = useState({
    is_running: false,
    start_time: null,
    end_time: null,
    result_dir: null,
  })
  const [logs, setLogs] = useState('ログが表示されます...')
  const logContainerRef = useRef(null)
  const statusIntervalRef = useRef(null)

  // 初期化: CSV ディレクトリ一覧を取得
  useEffect(() => {
    fetchCsvDirs()
    fetchInitialStatus()
  }, [])

  const fetchCsvDirs = async () => {
    try {
      const response = await fetch('/csv_dirs')
      const data = await response.json()
      setCsvDirs(data.csv_dirs || [])
    } catch (error) {
      console.error('CSVディレクトリ取得エラー:', error)
    }
  }

  // ステータスのポーリング
  useEffect(() => {
    if (status.is_running) {
      startStatusPolling()
    } else {
      stopStatusPolling()
    }
    return () => stopStatusPolling()
  }, [status.is_running])

  // ログの自動スクロール
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs])

  const fetchInitialStatus = async () => {
    try {
      const response = await fetch('/status')
      const data = await response.json()
      setStatus(data)

      if (data.is_running) {
        // 実行中の場合はログを取得
        const logsResponse = await fetch('/logs')
        const logsData = await logsResponse.json()
        setLogs(logsData.logs.join('\n'))
      }
    } catch (error) {
      console.error('初期ステータス取得エラー:', error)
    }
  }

  const startStatusPolling = () => {
    if (statusIntervalRef.current) return

    statusIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch('/status')
        const data = await response.json()

        setStatus(data)

        // 新しいログを追加
        if (data.new_logs && data.new_logs.length > 0) {
          setLogs(prevLogs => {
            const currentLog = prevLogs === 'ログが表示されます...' || prevLogs === 'ログがクリアされました...'
              ? ''
              : prevLogs + '\n'
            return currentLog + data.new_logs.join('\n')
          })
        }
      } catch (error) {
        console.error('ステータス取得エラー:', error)
      }
    }, 1000)
  }

  const stopStatusPolling = () => {
    if (statusIntervalRef.current) {
      clearInterval(statusIntervalRef.current)
      statusIntervalRef.current = null
    }
  }

  const handleStartTraining = async () => {
    if (!selectedCsv) {
      alert('CSVディレクトリを選択してください')
      return
    }

    if (status.is_running) {
      alert('既に学習が実行中です')
      return
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
      })

      const data = await response.json()

      if (response.ok) {
        setStatus(prev => ({ ...prev, is_running: true }))
      } else {
        alert(`エラー: ${data.error}`)
      }
    } catch (error) {
      alert(`エラー: ${error.message}`)
    }
  }

  const handleClearLog = () => {
    setLogs('ログがクリアされました...')
  }

  const getStatusText = () => {
    if (status.is_running) return '実行中'
    if (status.end_time) {
      return status.result_dir ? '完了' : 'エラー'
    }
    return '待機中'
  }

  const getStatusColor = () => {
    if (status.is_running) return 'text-green-600 animate-pulse-slow'
    if (status.end_time) {
      return status.result_dir ? 'text-blue-600' : 'text-red-600'
    }
    return 'text-gray-600'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 via-purple-700 to-indigo-800">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="bg-white rounded-xl shadow-2xl overflow-hidden">
          {/* ヘッダー */}
          <header className="bg-gradient-to-r from-purple-600 to-indigo-700 text-white py-8 px-6">
            <h1 className="text-4xl font-bold mb-2">株価予測モデル学習</h1>
            <p className="text-lg opacity-90">LSTM株価予測モデルのWebインターフェース</p>
          </header>

          <main className="p-6 space-y-6">
            {/* ナビゲーション */}
            <div className="mb-6">
              <Link
                to="/results"
                className="inline-block px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
              >
                学習結果を見る →
              </Link>
            </div>

            {/* 学習設定セクション */}
            <section className="bg-gray-50 rounded-lg p-6 shadow">
              <h2 className="text-2xl font-bold text-purple-700 mb-4 border-b-2 border-purple-700 pb-2">
                学習設定
              </h2>
              <div className="mb-4">
                <label htmlFor="csv-select" className="block font-semibold text-gray-700 mb-2">
                  データセットを選択:
                </label>
                <select
                  id="csv-select"
                  value={selectedCsv}
                  onChange={(e) => setSelectedCsv(e.target.value)}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-purple-600 transition"
                >
                  <option value="">-- CSVディレクトリを選択 --</option>
                  {csvDirs.map((dir) => (
                    <option key={dir} value={dir}>{dir}</option>
                  ))}
                </select>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={handleStartTraining}
                  disabled={status.is_running}
                  className="px-6 py-3 bg-gradient-to-r from-purple-600 to-indigo-700 text-white font-semibold rounded-lg uppercase tracking-wide transition transform hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                  {status.is_running ? '実行中...' : '学習開始'}
                </button>
                <button
                  onClick={handleClearLog}
                  className="px-6 py-3 bg-gray-600 text-white font-semibold rounded-lg uppercase tracking-wide transition transform hover:scale-105 hover:bg-gray-700"
                >
                  ログクリア
                </button>
              </div>
            </section>

            {/* ステータスセクション */}
            <section className="bg-gray-50 rounded-lg p-6 shadow">
              <h2 className="text-2xl font-bold text-purple-700 mb-4 border-b-2 border-purple-700 pb-2">
                実行状態
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-white rounded-lg p-4 border-l-4 border-purple-600">
                  <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">状態</div>
                  <div className={`text-lg font-semibold ${getStatusColor()}`}>
                    {getStatusText()}
                  </div>
                </div>
                <div className="bg-white rounded-lg p-4 border-l-4 border-purple-600">
                  <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">開始時刻</div>
                  <div className="text-lg font-semibold text-gray-800">
                    {status.start_time || '-'}
                  </div>
                </div>
                <div className="bg-white rounded-lg p-4 border-l-4 border-purple-600">
                  <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">終了時刻</div>
                  <div className="text-lg font-semibold text-gray-800">
                    {status.end_time || '-'}
                  </div>
                </div>
                <div className="bg-white rounded-lg p-4 border-l-4 border-purple-600">
                  <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">結果保存先</div>
                  <div className="text-lg font-semibold text-gray-800">
                    {status.result_dir ? `result/${status.result_dir}` : '-'}
                  </div>
                </div>
              </div>
            </section>

            {/* ログセクション */}
            <section className="bg-gray-50 rounded-lg p-6 shadow">
              <h2 className="text-2xl font-bold text-purple-700 mb-4 border-b-2 border-purple-700 pb-2">
                実行ログ
              </h2>
              <div
                ref={logContainerRef}
                className="bg-gray-900 rounded-lg p-5 max-h-[500px] overflow-y-auto"
              >
                <pre className="text-gray-300 font-mono text-sm whitespace-pre-wrap break-words">
                  {logs}
                </pre>
              </div>
            </section>
          </main>

          <footer className="bg-gray-50 py-5 text-center text-gray-600 text-sm">
            <p>&copy; 2026 AI Stock Trader</p>
          </footer>
        </div>
      </div>
    </div>
  )
}

export default TrainingPage
