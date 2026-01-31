import { useState, useEffect, useRef } from 'react'

// スライダーコンポーネント
function ParamSlider({ label, value, onChange, min, max, step, disabled, format }) {
  const displayValue = format ? format(value) : value
  return (
    <div className="flex flex-col gap-1 max-w-xs mx-auto w-full">
      <div className="flex justify-between items-center">
        <label className="text-xs text-gray-600">{label}</label>
        <span className="text-xs font-semibold text-blue-600">{displayValue}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500 disabled:opacity-50 disabled:cursor-not-allowed slider-thumb"
      />
    </div>
  )
}

function TrainingSection() {
  const [csvDirs, setCsvDirs] = useState([])
  const [models, setModels] = useState([])
  const [selectedCsv, setSelectedCsv] = useState('')
  const [selectedModel, setSelectedModel] = useState('')

  // 学習パラメータ
  const [params, setParams] = useState({
    timeStep: 480,
    epochs: 5,
    batchSize: 128,
    learningRate: 0.001,
    priceThreshold: 1.0,
    hiddenSize: 64,
    numLayers: 2,
    dropout: 0.3,
  })

  const [status, setStatus] = useState({
    is_running: false,
    start_time: null,
    end_time: null,
    result_dir: null,
    running_params: null,
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
  const lastProgressRef = useRef({ section: 0, percent: 0 })

  // ログモーダル用の状態
  const [showLogModal, setShowLogModal] = useState(false)
  const [logContent, setLogContent] = useState('')
  const logContainerRef = useRef(null)
  const logPollingRef = useRef(null)

  useEffect(() => {
    fetchCsvDirs()
    fetchModels()
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

  const fetchModels = async () => {
    try {
      const response = await fetch('/api/models')
      const data = await response.json()
      setModels(data.models || [])
    } catch (error) {
      console.error('モデル取得エラー:', error)
    }
  }

  useEffect(() => {
    // ステータスは常にポーリング（CLI実行検知のため）
    startStatusPolling()

    if (status.is_running) {
      startProgressPolling()
    } else {
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
      // 実行中のパラメータがあれば反映
      if (data.is_running && data.running_params) {
        applyRunningParams(data.running_params)
      }
    } catch (error) {
      console.error('初期ステータス取得エラー:', error)
    }
  }

  const applyRunningParams = (runningParams) => {
    setSelectedCsv(runningParams.csv_dir || '')
    setSelectedModel(runningParams.base_model || '')
    setParams({
      timeStep: runningParams.time_step || 480,
      epochs: runningParams.epochs || 5,
      batchSize: runningParams.batch_size || 128,
      learningRate: runningParams.learning_rate || 0.001,
      priceThreshold: runningParams.price_threshold || 1.0,
      hiddenSize: runningParams.hidden_size || 64,
      numLayers: runningParams.num_layers || 2,
      dropout: runningParams.dropout || 0.3,
    })
  }

  const startStatusPolling = () => {
    if (statusIntervalRef.current) return
    statusIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch('/status')
        const data = await response.json()
        setStatus(data)
        // 実行中のパラメータがあれば反映
        if (data.is_running && data.running_params) {
          applyRunningParams(data.running_params)
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

  const startProgressPolling = () => {
    if (progressIntervalRef.current) return
    // ポーリング開始時に前回の進捗をリセット
    lastProgressRef.current = { section: 0, percent: 0 }
    progressIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch('/progress')
        const data = await response.json()

        // 進捗の後退を防ぐ
        const currentSection = data.current_section || 0
        const currentPercent = data.section_percent || 0
        const last = lastProgressRef.current

        if (currentSection < last.section) {
          // セクションが後退している場合は読み込みエラーとみなして前の値を維持
          // 何も更新しない（前回のprogressをそのまま使う）
          return
        } else if (currentSection === last.section && currentPercent < last.percent) {
          // 同じセクション内で進捗が後退した場合は前の値を維持
          data.section_percent = last.percent
        } else {
          // 進捗を更新
          lastProgressRef.current = { section: currentSection, percent: currentPercent }
        }

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

  const handleParamChange = (key, value) => {
    setParams(prev => ({ ...prev, [key]: value }))
  }

  const fetchLogs = async () => {
    try {
      const response = await fetch('/api/logs')
      const data = await response.json()
      setLogContent(data.logs || '')
      // ログ表示後に自動スクロール
      setTimeout(() => {
        if (logContainerRef.current) {
          logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
        }
      }, 100)
    } catch (error) {
      setLogContent('ログ取得エラー: ' + error.message)
    }
  }

  const handleOpenLogModal = () => {
    setShowLogModal(true)
    fetchLogs()
  }

  // ログモーダルが開いている間、自動更新
  useEffect(() => {
    if (showLogModal) {
      logPollingRef.current = setInterval(() => {
        fetchLogs()
      }, 1000)
    } else {
      if (logPollingRef.current) {
        clearInterval(logPollingRef.current)
        logPollingRef.current = null
      }
    }
    return () => {
      if (logPollingRef.current) {
        clearInterval(logPollingRef.current)
        logPollingRef.current = null
      }
    }
  }, [showLogModal])

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
      const requestBody = {
        csv_dir: selectedCsv,
        time_step: params.timeStep,
        epochs: params.epochs,
        batch_size: params.batchSize,
        learning_rate: params.learningRate,
        price_threshold: params.priceThreshold,
        hidden_size: params.hiddenSize,
        num_layers: params.numLayers,
        dropout: params.dropout,
      }
      if (selectedModel) {
        requestBody.base_model = selectedModel
      }

      const response = await fetch('/start_training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
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

  const sectionPercent = progress.section_percent || 0

  return (
    <div className="h-[85vh] bg-gray-50 p-4">
      <div className="h-full bg-white rounded-xl shadow-sm flex flex-col overflow-hidden">
        {/* プルダウン選択 */}
        <div className="px-4 pt-4 space-y-2">
          <select
            value={selectedCsv}
            onChange={(e) => setSelectedCsv(e.target.value)}
            className="w-full px-2 py-1.5 border border-gray-300 rounded-lg text-xs focus:outline-none focus:border-blue-500"
            disabled={status.is_running}
          >
            <option value="">データセットを選択</option>
            {csvDirs.map((dir) => (
              <option key={dir} value={dir}>{dir}</option>
            ))}
          </select>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full px-2 py-1.5 border border-gray-300 rounded-lg text-xs focus:outline-none focus:border-blue-500"
            disabled={status.is_running}
          >
            <option value="">ベースモデル: なし（新規学習）</option>
            {models.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>

        {/* パラメータスライダー */}
        <div className="flex-1 px-4 pt-4 overflow-y-auto">
          <div className="flex flex-col gap-3">
            <ParamSlider
              label="タイムステップ"
              value={params.timeStep}
              onChange={(v) => handleParamChange('timeStep', v)}
              min={3}
              max={600}
              step={3}
              disabled={status.is_running}
            />
            <ParamSlider
              label="エポック数"
              value={params.epochs}
              onChange={(v) => handleParamChange('epochs', v)}
              min={1}
              max={200}
              step={1}
              disabled={status.is_running}
            />
            <ParamSlider
              label="バッチサイズ"
              value={params.batchSize}
              onChange={(v) => handleParamChange('batchSize', v)}
              min={16}
              max={512}
              step={16}
              disabled={status.is_running}
            />
            <ParamSlider
              label="学習率"
              value={params.learningRate}
              onChange={(v) => handleParamChange('learningRate', v)}
              min={0.0001}
              max={0.01}
              step={0.0001}
              disabled={status.is_running}
              format={(v) => v.toFixed(4)}
            />
            <ParamSlider
              label="株価上昇率(%)"
              value={params.priceThreshold}
              onChange={(v) => handleParamChange('priceThreshold', v)}
              min={0.1}
              max={5.0}
              step={0.1}
              disabled={status.is_running}
              format={(v) => v.toFixed(1)}
            />
            <ParamSlider
              label="隠れ層サイズ"
              value={params.hiddenSize}
              onChange={(v) => handleParamChange('hiddenSize', v)}
              min={16}
              max={256}
              step={16}
              disabled={status.is_running}
            />
            <ParamSlider
              label="LSTM層数"
              value={params.numLayers}
              onChange={(v) => handleParamChange('numLayers', v)}
              min={1}
              max={5}
              step={1}
              disabled={status.is_running}
            />
            <ParamSlider
              label="ドロップアウト"
              value={params.dropout}
              onChange={(v) => handleParamChange('dropout', v)}
              min={0}
              max={0.9}
              step={0.1}
              disabled={status.is_running}
              format={(v) => v.toFixed(1)}
            />
          </div>
        </div>

        {/* GOボタン */}
        <div className="flex justify-center py-4">
          <button
            onClick={handleStartTraining}
            disabled={status.is_running || !selectedCsv}
            className="w-16 h-16 rounded-full bg-blue-500 text-white font-bold text-lg shadow-lg hover:bg-blue-600 active:scale-95 disabled:bg-gray-300 disabled:shadow-none disabled:cursor-not-allowed transition-all flex items-center justify-center"
          >
            GO
          </button>
        </div>

        {/* 進捗表示（カード内最下部） */}
        <div className="px-4 py-3 bg-gray-50 border-t border-gray-200 rounded-b-xl">
          {status.is_running ? (
            <>
              <div className="flex justify-between items-center mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-gray-800">
                    {progress.current_section}/{progress.total_sections} {progress.section_name}
                  </span>
                  {progress.current_section === 2 && progress.total_epochs > 0 && (
                    <span className="text-xs text-gray-500">
                      - Epoch {progress.epoch}/{progress.total_epochs}
                      {progress.total_batches > 0 && ` Batch ${progress.batch}/${progress.total_batches}`}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-blue-500 font-bold">{sectionPercent.toFixed(0)}%</span>
                  <button
                    onClick={handleOpenLogModal}
                    className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
                    title="ログを表示"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </button>
                </div>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-200"
                  style={{ width: `${sectionPercent}%` }}
                />
              </div>
            </>
          ) : (
            <div className="flex justify-between items-center">
              <div className="flex-1 text-xs text-gray-400 text-center">
                学習待機中
              </div>
              <button
                onClick={handleOpenLogModal}
                className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
                title="ログを表示"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ログモーダル */}
      {showLogModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col">
            <div className="flex justify-between items-center px-4 py-3 border-b border-gray-200">
              <h3 className="text-sm font-bold text-gray-800">学習ログ</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={fetchLogs}
                  className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors"
                  title="更新"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
                <button
                  onClick={() => setShowLogModal(false)}
                  className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors"
                  title="閉じる"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
            <div
              ref={logContainerRef}
              className="flex-1 overflow-auto p-4 bg-gray-900 text-gray-100 font-mono text-xs leading-relaxed"
            >
              <pre className="whitespace-pre-wrap break-words">{logContent || 'ログを読み込み中...'}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default TrainingSection
