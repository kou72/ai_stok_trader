import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import TrainingPage from './pages/TrainingPage'
import ResultsPage from './pages/ResultsPage'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<TrainingPage />} />
        <Route path="/results" element={<ResultsPage />} />
      </Routes>
    </Router>
  )
}

export default App
