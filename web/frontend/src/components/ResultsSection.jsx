import { useState, useEffect } from 'react'
import { Swiper, SwiperSlide } from 'swiper/react'
import { Pagination } from 'swiper/modules'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Bar, Line } from 'react-chartjs-2'

import 'swiper/css'
import 'swiper/css/pagination'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

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

function ResultSlide({ result, detail, viewportHeight }) {
  const headerHeight = 80
  const paginationHeight = 50
  const padding = 32
  const availableHeight = viewportHeight - headerHeight - paginationHeight - padding

  const idHeight = 20
  const barChartHeight = Math.floor(availableHeight * 0.30)
  const lineChartHeight = Math.floor(availableHeight * 0.45)
  const configHeight = availableHeight - idHeight - barChartHeight - lineChartHeight - 16 * 2

  const precisionChartData = {
    labels: ['訓練', '検証', 'テスト'],
    datasets: [
      {
        label: '正答率 (%)',
        data: [
          detail.results?.['訓練']?.precision * 100 || 0,
          detail.results?.['検証']?.precision * 100 || 0,
          detail.results?.['テスト']?.precision * 100 || 0,
        ],
        backgroundColor: [
          'rgba(99, 102, 241, 0.8)',
          'rgba(139, 92, 246, 0.8)',
          'rgba(168, 85, 247, 0.8)',
        ],
        borderColor: [
          'rgb(99, 102, 241)',
          'rgb(139, 92, 246)',
          'rgb(168, 85, 247)',
        ],
        borderWidth: 2,
      },
    ],
  }

  const precisionChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value) { return value + '%' },
          font: { size: 10 },
        },
      },
      x: {
        ticks: { font: { size: 10 } },
      },
    },
  }

  const historyChartData = {
    labels: detail.loss_history?.map(row => row.epoch) || [],
    datasets: [
      {
        label: '訓練Loss',
        data: detail.loss_history?.map(row => parseFloat(row.train_loss)) || [],
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        yAxisID: 'y',
        borderWidth: 2,
        pointRadius: 2,
      },
      {
        label: '検証Loss',
        data: detail.loss_history?.map(row => parseFloat(row.val_loss)) || [],
        borderColor: 'rgb(249, 115, 22)',
        backgroundColor: 'rgba(249, 115, 22, 0.1)',
        yAxisID: 'y',
        borderWidth: 2,
        pointRadius: 2,
      },
      {
        label: '訓練的中率',
        data: detail.precision_history?.map(row => parseFloat(row.train_precision) * 100) || [],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        yAxisID: 'y1',
        borderWidth: 2,
        pointRadius: 2,
      },
      {
        label: '検証的中率',
        data: detail.precision_history?.map(row => parseFloat(row.val_precision) * 100) || [],
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        yAxisID: 'y1',
        borderWidth: 2,
        pointRadius: 2,
      },
    ],
  }

  const historyChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'top',
        labels: { font: { size: 10 }, boxWidth: 20 },
      },
    },
    scales: {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: { display: true, text: 'Loss', font: { size: 10 } },
        ticks: { font: { size: 9 } },
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: { display: true, text: '的中率 (%)', font: { size: 10 } },
        grid: { drawOnChartArea: false },
        min: 0,
        max: 100,
        ticks: { font: { size: 9 } },
      },
      x: {
        ticks: { font: { size: 9 } },
      },
    },
  }

  return (
    <div className="h-full flex flex-col py-2 overflow-hidden">
      <div className="flex-shrink-0 flex justify-between items-center px-2 mb-2" style={{ height: `${idHeight}px` }}>
        <p className="text-gray-600 text-xs">ID: {result.id}</p>
        <p className="text-blue-500 text-xs font-semibold">
          処理時間: {detail.time_log?.total?.formatted || '-'}
        </p>
      </div>

      <div className="flex-shrink-0 bg-white border border-gray-200 rounded-lg p-3 mb-2 shadow-sm" style={{ height: `${barChartHeight}px` }}>
        <div style={{ height: `${barChartHeight - 80}px` }}>
          <Bar data={precisionChartData} options={precisionChartOptions} />
        </div>
        <div className="mt-2 grid grid-cols-3 gap-2 text-center text-xs">
          {['訓練', '検証', 'テスト'].map((name) => {
            const data = detail.results?.[name]
            if (!data) return null
            return (
              <div key={name} className="bg-gray-50 p-2 rounded">
                <div className="text-blue-500 font-bold text-sm">{data.precision_percent}</div>
                <div className="text-xs text-gray-500">{data.correct_count}/{data.predicted_count}</div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="flex-shrink-0 bg-white border border-gray-200 rounded-lg p-3 mb-2 shadow-sm" style={{ height: `${lineChartHeight}px` }}>
        <div style={{ height: `${lineChartHeight - 24}px` }}>
          <Line data={historyChartData} options={historyChartOptions} />
        </div>
      </div>

      <div className="flex-1 bg-white border border-gray-200 rounded-lg p-3 shadow-sm overflow-y-auto" style={{ height: `${configHeight}px` }}>
        <h3 className="text-sm font-bold text-gray-800 mb-2 border-b pb-1 sticky top-0 bg-white">学習設定</h3>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          {detail.config && Object.entries({
            'エポック数': detail.config.EPOCHS,
            'バッチサイズ': detail.config.BATCH_SIZE,
            '学習率': detail.config.LEARNING_RATE,
            '基準超閾値': detail.config.PRICE_INCREASE_THRESHOLD != null ? `${detail.config.PRICE_INCREASE_THRESHOLD}%` : '-',
            '隠れ層サイズ': detail.config.HIDDEN_SIZE,
            'LSTM層数': detail.config.NUM_LAYERS,
            'Dropout': detail.config.DROPOUT,
            'デバイス': detail.config.DEVICE,
          }).map(([key, value]) => (
            <div key={key} className="flex justify-between py-1 border-b border-gray-100">
              <span className="text-gray-600">{key}:</span>
              <span className="font-semibold text-gray-800">{value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function ResultsSection() {
  const [results, setResults] = useState([])
  const [detailsMap, setDetailsMap] = useState({})
  const [loading, setLoading] = useState(true)
  const { height } = useWindowSize()

  useEffect(() => {
    fetchResults()
  }, [])

  const fetchResults = async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/results')
      const data = await response.json()
      const resultsList = data.results || []
      setResults(resultsList)

      const detailsPromises = resultsList.map(result =>
        fetch(`/api/results/${result.id}`).then(res => res.json())
      )
      const details = await Promise.all(detailsPromises)

      const detailsMapObj = {}
      details.forEach(detail => {
        detailsMapObj[detail.id] = detail
      })
      setDetailsMap(detailsMapObj)
    } catch (error) {
      console.error('結果取得エラー:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="h-full bg-white flex items-center justify-center">
        <div className="text-gray-500">読み込み中...</div>
      </div>
    )
  }

  if (results.length === 0) {
    return (
      <div className="h-full bg-white flex flex-col items-center justify-center px-4">
        <p className="text-gray-400 text-lg mb-4">学習結果がまだありません</p>
        <p className="text-gray-300 text-sm">下にスワイプして学習を開始</p>
      </div>
    )
  }

  return (
    <div className="h-full bg-white overflow-hidden">
      <div className="h-full flex flex-col max-w-6xl mx-auto px-4">
        <Swiper
          modules={[Pagination]}
          spaceBetween={50}
          slidesPerView={1}
          pagination={{ clickable: true }}
          className="results-swiper w-full flex-1"
        >
          {results.map((result) => {
            const detail = detailsMap[result.id]
            if (!detail) return null

            return (
              <SwiperSlide key={result.id}>
                <ResultSlide result={result} detail={detail} viewportHeight={height} />
              </SwiperSlide>
            )
          })}
        </Swiper>
      </div>
    </div>
  )
}

export default ResultsSection
