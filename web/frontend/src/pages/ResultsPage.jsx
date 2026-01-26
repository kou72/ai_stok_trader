import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Swiper, SwiperSlide } from 'swiper/react'
import { Navigation, Pagination } from 'swiper/modules'
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

// Swiper styles
import 'swiper/css'
import 'swiper/css/navigation'
import 'swiper/css/pagination'

// Chart.js の登録
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

function ResultsPage() {
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

      // 各結果の詳細を取得
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
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-gray-800 text-2xl">読み込み中...</div>
      </div>
    )
  }

  if (results.length === 0) {
    return (
      <div className="min-h-screen bg-white">
        <div className="container mx-auto px-4 py-8">
          <Link
            to="/"
            className="inline-block mb-6 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition"
          >
            ← 学習ページに戻る
          </Link>
          <div className="text-center py-12">
            <p className="text-gray-500 text-lg">学習結果がまだありません</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen bg-white overflow-hidden">
      <div className="container mx-auto px-4 h-full flex flex-col max-w-6xl">
        {/* カルーセル */}
        <Swiper
          modules={[Navigation, Pagination]}
          spaceBetween={50}
          slidesPerView={1}
          navigation
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

function ResultSlide({ result, detail, viewportHeight }) {
  // 画面サイズに応じた高さ計算
  const headerHeight = 80
  const paginationHeight = 50
  const padding = 32
  const availableHeight = viewportHeight - headerHeight - paginationHeight - padding

  // セクションの高さ配分（vh単位ではなくpx）
  const idHeight = 20
  const barChartHeight = Math.floor(availableHeight * 0.35)
  const lineChartHeight = Math.floor(availableHeight * 0.35)
  const configHeight = availableHeight - idHeight - barChartHeight - lineChartHeight - 16 * 2 // gap分を引く

  // 棒グラフデータ（results.json）
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
      legend: {
        display: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value) {
            return value + '%'
          },
          font: {
            size: 10,
          },
        },
      },
      x: {
        ticks: {
          font: {
            size: 10,
          },
        },
      },
    },
  }

  // 折れ線グラフデータ（loss + precision history）
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
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: {
            size: 10,
          },
          boxWidth: 20,
        },
      },
    },
    scales: {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Loss',
          font: {
            size: 10,
          },
        },
        ticks: {
          font: {
            size: 9,
          },
        },
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: '的中率 (%)',
          font: {
            size: 10,
          },
        },
        grid: {
          drawOnChartArea: false,
        },
        min: 0,
        max: 100,
        ticks: {
          font: {
            size: 9,
          },
        },
      },
      x: {
        ticks: {
          font: {
            size: 9,
          },
        },
      },
    },
  }

  return (
    <div className="h-full flex flex-col py-2 overflow-hidden">
      {/* ヘッダー（戻るボタン + ID） */}
      <div className="flex-shrink-0 flex justify-between items-center px-2 mb-2" style={{ height: `${idHeight}px` }}>
        <p className="text-gray-600 text-xs">ID: {result.id}</p>
      </div>

      {/* 棒グラフ（正答率） */}
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
                <div className="text-purple-600 font-bold text-sm">
                  {data.precision_percent}
                </div>
                <div className="text-xs text-gray-500">
                  {data.correct_count}/{data.predicted_count}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* 折れ線グラフ（LossとPrecision） */}
      <div className="flex-shrink-0 bg-white border border-gray-200 rounded-lg p-3 mb-2 shadow-sm" style={{ height: `${lineChartHeight}px` }}>
        <div style={{ height: `${lineChartHeight - 24}px` }}>
          <Line data={historyChartData} options={historyChartOptions} />
        </div>
      </div>

      {/* Config と TimeLog（2列、スクロール可能） */}
      <div className="flex-1 grid grid-cols-2 gap-2 overflow-hidden" style={{ height: `${configHeight}px` }}>
        {/* Config */}
        <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-sm overflow-y-auto">
          <h3 className="text-sm font-bold text-gray-800 mb-2 border-b pb-1 sticky top-0 bg-white">学習設定</h3>
          <div className="space-y-1 text-xs">
            {detail.config && Object.entries({
              'エポック数': detail.config.EPOCHS,
              'バッチサイズ': detail.config.BATCH_SIZE,
              '学習率': detail.config.LEARNING_RATE,
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

        {/* TimeLog */}
        <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-sm overflow-y-auto">
          <h3 className="text-sm font-bold text-gray-800 mb-2 border-b pb-1 sticky top-0 bg-white">処理時間</h3>
          <div className="space-y-1 text-xs">
            {detail.time_log && Object.entries({
              'CSV読込': detail.time_log['1_load_csv']?.formatted,
              '前処理': detail.time_log['2_preprocess_and_normalize']?.formatted,
              'シーケンス': detail.time_log['3_create_sequences']?.formatted,
              'データ分割': detail.time_log['4_split_data']?.formatted,
              'モデル構築': detail.time_log['6_build_model']?.formatted,
              '訓練': detail.time_log['7_train']?.formatted,
              '評価': detail.time_log['8_evaluate']?.formatted,
              '可視化': detail.time_log['9_visualize']?.formatted,
              '合計': detail.time_log.total?.formatted,
            }).map(([key, value]) => {
              if (!value) return null
              return (
                <div key={key} className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600">{key}:</span>
                  <span className={`font-semibold ${key === '合計' ? 'text-purple-600' : 'text-gray-800'}`}>
                    {value}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResultsPage
