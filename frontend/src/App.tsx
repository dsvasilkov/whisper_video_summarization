import { useState } from 'react'
import { InferencePage } from './InferencePage'
import { TrainingPage } from './TrainingPage'
import './App.css'

type Mode = 'inference' | 'training'

function App() {
  const [mode, setMode] = useState<Mode>('inference')

  return (
    <div className="app">
      <header className="header">
        <h1>Whisper Video Summarization</h1>
        <nav className="nav">
          <button
            type="button"
            className={mode === 'inference' ? 'active' : ''}
            onClick={() => setMode('inference')}
          >
            Инференс видео
          </button>
          <button
            type="button"
            className={mode === 'training' ? 'active' : ''}
            onClick={() => setMode('training')}
          >
            Обучение
          </button>
        </nav>
      </header>
      <main className="main">
        {mode === 'inference' ? <InferencePage /> : <TrainingPage />}
      </main>
    </div>
  )
}

export default App
