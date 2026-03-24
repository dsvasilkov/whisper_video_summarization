import { useCallback, useEffect, useRef, useState } from 'react'
import { getTaskStatus, inferVideoUpload } from './api'

const ACCEPT = '.mp4,.wav,.mp3,.mkv'

export function InferencePage() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{ transcription: string; summary: string } | null>(null)
  const [statusLabel, setStatusLabel] = useState<string | null>(null)
  const [dragover, setDragover] = useState(false)

  const onFileChange = useCallback((f: File | null) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    if (!f) {
      setFile(null)
      return
    }
    setFile(f)
    if (f.type.startsWith('video/') || f.type.startsWith('audio/')) {
      setPreviewUrl(URL.createObjectURL(f))
    }
  }, [previewUrl])

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    onFileChange(f ?? null)
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragover(false)
    const f = e.dataTransfer.files?.[0]
    if (f) onFileChange(f)
  }

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragover(true)
  }

  const onDragLeave = () => setDragover(false)

  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const runInference = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    setStatusLabel('Постановка в очередь…')
    try {
      const { task_id } = await inferVideoUpload(file)
      setStatusLabel('В очереди. Ожидание результата…')
      const poll = () => {
        getTaskStatus(task_id).then((t) => {
          if (t.status === 'processing') setStatusLabel('Обработка…')
          if (t.status === 'completed' && t.result_transcription != null && t.result_summary != null) {
            setResult({ transcription: t.result_transcription, summary: t.result_summary })
            setStatusLabel(null)
            setLoading(false)
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
          } else if (t.status === 'failed') {
            setError(t.error_message || 'Ошибка инференса')
            setStatusLabel(null)
            setLoading(false)
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
          }
        }).catch((e) => {
          setError(e instanceof Error ? e.message : String(e))
          setStatusLabel(null)
          setLoading(false)
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
        })
      }
      poll()
      pollIntervalRef.current = setInterval(poll, 2000)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatusLabel(null)
      setLoading(false)
    }
  }

  useEffect(() => () => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
  }, [])

  return (
    <div className="card">
      <h2>Инференс видео</h2>
      <div
        className={`upload-area ${dragover ? 'dragover' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => document.getElementById('inference-file')?.click()}
      >
        <input
          id="inference-file"
          type="file"
          accept={ACCEPT}
          onChange={onInputChange}
        />
        {file ? file.name : 'Загрузите файл (mp4, wav, mp3, mkv)'}
      </div>
      {previewUrl && (
        <video
          src={previewUrl}
          controls
          className="video-preview"
        />
      )}
      <button
        type="button"
        className="btn"
        disabled={!file || loading}
        onClick={runInference}
      >
        {loading ? 'Обработка…' : 'Запустить транскрибацию и суммаризацию'}
      </button>
      {error && <p className="error">{error}</p>}
      {loading && <p className="loading">{statusLabel ?? 'Подождите, это может занять несколько минут.'}</p>}
      {result && (
        <>
          <div className="result-block">
            <h3>Транскрипция</h3>
            <pre>{result.transcription}</pre>
          </div>
          <div className="result-block">
            <h3>Суммаризация</h3>
            <pre>{result.summary}</pre>
          </div>
        </>
      )}
    </div>
  )
}
