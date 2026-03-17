import { useState } from 'react'
import { startTraining, uploadDataset } from './api'

export function TrainingPage() {
  const [configPath, setConfigPath] = useState('configs/train.yaml')
  const [datasetFile, setDatasetFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const runTraining = async () => {
    setLoading(true)
    setError(null)
    setMessage(null)
    try {
      let datasetPath: string | null = null
      if (datasetFile) {
        const { path } = await uploadDataset(datasetFile)
        datasetPath = path
      }
      const res = await startTraining(configPath, datasetPath)
      setMessage(res.status === 'training started' ? 'Обучение запущено.' : JSON.stringify(res))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h2>Обучение суммаризатора</h2>
      <div className="input-group">
        <label htmlFor="config-path">Путь к конфигу Hydra</label>
        <input
          id="config-path"
          type="text"
          value={configPath}
          onChange={(e) => setConfigPath(e.target.value)}
        />
      </div>
      <div className="input-group">
        <label>Датасет (Gazeta, jsonl/csv)</label>
        <input
          type="file"
          accept=".jsonl,.csv"
          onChange={(e) => setDatasetFile(e.target.files?.[0] ?? null)}
        />
      </div>
      <button
        type="button"
        className="btn"
        disabled={loading}
        onClick={runTraining}
      >
        {loading ? 'Запуск…' : 'Запустить обучение'}
      </button>
      {error && <p className="error">{error}</p>}
      {message && <p style={{ marginTop: '0.5rem', color: '#a1a1aa' }}>{message}</p>}
    </div>
  )
}
