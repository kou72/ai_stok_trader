import ResultsSection from '../components/ResultsSection'
import TrainingSection from '../components/TrainingSection'

function HomePage() {
  return (
    <div className="h-screen overflow-y-auto snap-y snap-mandatory">
      {/* 結果ページ（上） */}
      <div className="h-screen snap-start snap-always">
        <ResultsSection />
      </div>

      {/* 学習ページ（下） */}
      <div className="h-screen snap-start snap-always">
        <TrainingSection />
      </div>
    </div>
  )
}

export default HomePage
