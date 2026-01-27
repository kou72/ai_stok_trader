import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'

// 画面サイズ取得用のカスタムフック
function useWindowSize() {
  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  })

  useEffect(() => {
    function handleResize() {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      })
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return windowSize
}

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
  const { height } = useWindowSize()

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

  // 高さ計算
  const headerHeight = 60
  const controlHeight = 60
  const padding = 32
  const logHeight = height - headerHeight - controlHeight - padding

  return (
    <div className="h-screen bg-white flex flex-col overflow-hidden">
      {/* コントロール部分 */}
      <div className="flex-shrink-0 flex items-center gap-2 px-4 py-2 border-b border-gray-200" style={{ height: `${controlHeight}px` }}>
        <select
          value={selectedCsv}
          onChange={(e) => setSelectedCsv(e.target.value)}
          className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-500"
        >
          <option value="">データセットを選択</option>
          {csvDirs.map((dir) => (
            <option key={dir} value={dir}>{dir}</option>
          ))}
        </select>
        <button
          onClick={handleStartTraining}
          disabled={status.is_running || !selectedCsv}
          className="px-6 py-2 bg-blue-500 text-white font-bold rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
        >
          GO
        </button>
      </div>

      {/* ログ表示 */}
      <div
        ref={logContainerRef}
        className="flex-1 bg-gray-900 overflow-y-auto p-4"
        style={{ height: `${logHeight}px` }}
      >
        <pre className="text-gray-300 font-mono text-xs whitespace-pre-wrap break-words">
          {logs}
        </pre>
      </div>
    </div>
  )
}

export default TrainingPage
