import './App.css'
import MNISTTrainer from './components/MNISTTrainer'
import ErrorBoundary from './components/ErrorBoundary'

function App() {
  return (
    <div className="App">
      <ErrorBoundary>
        <MNISTTrainer />
      </ErrorBoundary>
    </div>
  )
}

export default App
