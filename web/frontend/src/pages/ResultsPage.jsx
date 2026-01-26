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

function ResultsPage() {
  const [results, setResults] = useState([])
  const [detailsMap, setDetailsMap] = useState({})
  const [loading, setLoading] = useState(true)

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

  return (
    <div className="min-h-screen bg-white">
      <div className="container mx-auto px-4 py-8 max-w-6xl">

        {/* カルーセル */}
        <Swiper
          modules={[Navigation, Pagination]}
          spaceBetween={50}
          slidesPerView={1}
          navigation
          pagination={{ clickable: true }}
          className="results-swiper"
        >
          {results.map((result) => {
            const detail = detailsMap[result.id]
            if (!detail) return null

            return (
              <SwiperSlide key={result.id}>
                <ResultSlide result={result} detail={detail} />
              </SwiperSlide>
            )
          })}
        </Swiper>
      </div>
    </div>
  )
}

function ResultSlide({ result, detail }) {
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
      title: {
        display: true,
        text: 'データセット別正答率',
        font: {
          size: 18,
          weight: 'bold',
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value) {
            return value + '%'
          }
        }
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
      },
      {
        label: '検証Loss',
        data: detail.loss_history?.map(row => parseFloat(row.val_loss)) || [],
        borderColor: 'rgb(249, 115, 22)',
        backgroundColor: 'rgba(249, 115, 22, 0.1)',
        yAxisID: 'y',
      },
      {
        label: '訓練的中率',
        data: detail.precision_history?.map(row => parseFloat(row.train_precision) * 100) || [],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        yAxisID: 'y1',
      },
      {
        label: '検証的中率',
        data: detail.precision_history?.map(row => parseFloat(row.val_precision) * 100) || [],
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        yAxisID: 'y1',
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
      },
      title: {
        display: true,
        text: 'Lossと的中率の推移',
        font: {
          size: 18,
          weight: 'bold',
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
        },
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: '的中率 (%)',
        },
        grid: {
          drawOnChartArea: false,
        },
        min: 0,
        max: 100,
      },
    },
  }

  return (
    <div className="pb-16">
      {/* ID */}
      <div className="text-center mb-6">
        <p className="text-gray-600 text-sm mt-1">ID: {result.id}</p>
      </div>

      {/* 棒グラフ（正答率） */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6 shadow-sm">
        <div className="h-64">
          <Bar data={precisionChartData} options={precisionChartOptions} />
        </div>
        <div className="mt-4 grid grid-cols-3 gap-4 text-center text-sm">
          {['訓練', '検証', 'テスト'].map((name) => {
            const data = detail.results?.[name]
            if (!data) return null
            return (
              <div key={name} className="bg-gray-50 p-3 rounded">
                <div className="font-semibold text-gray-700">{name}</div>
                <div className="text-purple-600 font-bold text-lg">
                  {data.precision_percent}
                </div>
                <div className="text-xs text-gray-500">
                  {data.correct_count}/{data.predicted_count}件的中
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* 折れ線グラフ（LossとPrecision） */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6 shadow-sm">
        <div className="h-80">
          <Line data={historyChartData} options={historyChartOptions} />
        </div>
      </div>

      {/* Config と TimeLog（2列） */}
      <div className="grid grid-cols-2 gap-6">
        {/* Config */}
        <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <h3 className="text-lg font-bold text-gray-800 mb-4 border-b pb-2">学習設定</h3>
          <div className="space-y-2 text-sm">
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
        <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <h3 className="text-lg font-bold text-gray-800 mb-4 border-b pb-2">処理時間</h3>
          <div className="space-y-2 text-sm">
            {detail.time_log && Object.entries({
              'CSVデータ読み込み': detail.time_log['1_load_csv']?.formatted,
              'データ前処理': detail.time_log['2_preprocess_and_normalize']?.formatted,
              'シーケンス作成': detail.time_log['3_create_sequences']?.formatted,
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
