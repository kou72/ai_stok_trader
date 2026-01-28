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

  const handleParamChange = (key, value) => {
    setParams(prev => ({ ...prev, [key]: value }))
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
              max={100}
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
                <span className="text-xs text-blue-500 font-bold">{sectionPercent.toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-200"
                  style={{ width: `${sectionPercent}%` }}
                />
              </div>
            </>
          ) : (
            <div className="text-xs text-gray-400 text-center">
              学習待機中
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default TrainingSection
