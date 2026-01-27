import { useState, useEffect, useRef } from 'react'

function TrainingPage() {
  const [csvDirs, setCsvDirs] = useState([])
  const [selectedCsv, setSelectedCsv] = useState('')
  const [status, setStatus] = useState({
    is_running: false,
    start_time: null,
    end_time: null,
    result_dir: null,
  })
  const [progress, setProgress] = useState({
    is_running: false,
    current_section: 0,
    total_sections: 3,
    section_name: '',
    section_percent: 0,
    section_detail: '',
    epoch: 0,
    total_epochs: 0,
    batch: 0,
    total_batches: 0,
  })
  const statusIntervalRef = useRef(null)
  const progressIntervalRef = useRef(null)

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
      startProgressPolling()
    } else {
      stopStatusPolling()
      stopProgressPolling()
    }
    return () => {
      stopStatusPolling()
      stopProgressPolling()
    }
  }, [status.is_running])

  const fetchInitialStatus = async () => {
    try {
      const response = await fetch('/status')
      const data = await response.json()
      setStatus(data)
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

  const startProgressPolling = () => {
    if (progressIntervalRef.current) return

    progressIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch('/progress')
        const data = await response.json()
        setProgress(data)
      } catch (error) {
        console.error('進捗取得エラー:', error)
      }
    }, 200)
  }

  const stopProgressPolling = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current)
      progressIntervalRef.current = null
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

  // セクション表示用
  const sectionPercent = progress.section_percent || 0

  return (
    <div className="h-screen bg-white flex flex-col">
      {/* コントロール部分 */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-200">
        <select
          value={selectedCsv}
          onChange={(e) => setSelectedCsv(e.target.value)}
          className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-500"
          disabled={status.is_running}
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

      {/* 進捗表示 */}
      {status.is_running && (
        <div className="px-4 py-4 bg-gray-50 border-b border-gray-200">
          <div className="flex justify-between items-center mb-2">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-gray-800">
                {progress.current_section}/{progress.total_sections} {progress.section_name}
              </span>
              {progress.section_detail && (
                <span className="text-xs text-gray-500">
                  - {progress.section_detail}
                </span>
              )}
            </div>
            <span className="text-sm text-blue-500 font-bold">
              {sectionPercent.toFixed(0)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-blue-500 h-3 rounded-full transition-all duration-200"
              style={{ width: `${sectionPercent}%` }}
            />
          </div>
          {progress.current_section === 2 && progress.total_epochs > 0 && (
            <div className="mt-2 text-xs text-gray-500 text-right">
              Epoch {progress.epoch}/{progress.total_epochs}
              {progress.total_batches > 0 && ` - Batch ${progress.batch}/${progress.total_batches}`}
            </div>
          )}
        </div>
      )}

      {/* 待機状態 */}
      {!status.is_running && (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-gray-400 text-lg">データセットを選択して「GO」を押してください</p>
        </div>
      )}
    </div>
  )
}

export default TrainingPage
