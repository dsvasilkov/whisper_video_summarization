import { useCallback, useEffect, useRef, useState } from 'react'
import type { TaskStatusResponse } from './api'
import { getTaskStatus, uploadAudio } from './api'
import { useAuth } from './AuthContext'
import { convertMediaToWav } from './audio'
import { ResultViewSwitch } from './TranscriptViewer'

const ACCEPT = '.mp4,.wav,.mp3,.mkv'

type TranscriptMeta = {
  asr_done?: boolean
  diarization_ready?: boolean
  diarization_skipped?: boolean
  merge_done?: boolean
}

function extractTranscriptMeta(transcription: unknown): TranscriptMeta {
  if (!transcription || typeof transcription !== 'object') return {}
  const maybeMeta = (transcription as { _meta?: unknown })._meta
  if (!maybeMeta || typeof maybeMeta !== 'object') return {}
  return maybeMeta as TranscriptMeta
}

function shouldKeepPollingAfterCompleted(task: TaskStatusResponse): boolean {
  if (task.status !== 'completed') return false
  const meta = extractTranscriptMeta(task.result_transcription)
  const diarizationExpected =
    meta.asr_done === true ||
    meta.diarization_ready === true ||
    meta.merge_done === true ||
    meta.diarization_skipped === true

  if (!diarizationExpected) return false
  if (meta.diarization_skipped === true) return false
  return meta.merge_done !== true
}

function taskProgressMessage(task: TaskStatusResponse | null, fallback: string | null): string | null {
  if (task == null) return fallback
  if (task.status === 'pending') return 'В очереди…'
  if (task.status === 'failed') return null

  const meta = extractTranscriptMeta(task.result_transcription)
  const hasTranscription = task.result_transcription != null
  const hasSummary = task.result_summary != null

  if (task.status === 'processing') {
    if (!meta.asr_done) return 'Распознавание аудио…'
    if (meta.diarization_skipped === true) {
      return hasSummary ? 'Завершение обработки…' : 'Готовится суммаризация…'
    }
    if (!meta.diarization_ready) return 'Ожидание диаризации…'
    if (!meta.merge_done) return 'Сопоставление спикеров…'
    if (!hasSummary) return 'Готовится суммаризация…'
    return 'Завершение обработки…'
  }

  // Rare race: backend marked completed, but diarization merge not yet visible in payload.
  if (task.status === 'completed' && shouldKeepPollingAfterCompleted(task)) {
    if (!hasTranscription) return 'Финализация транскрипции…'
    return 'Финализация диаризации…'
  }
  return null
}

export function InferencePage() {
  const { token, logout } = useAuth()
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [jobActive, setJobActive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [liveTask, setLiveTask] = useState<TaskStatusResponse | null>(null)
  const [result, setResult] = useState<{
    taskId: string
    transcription: unknown
    summary: string | null
  } | null>(null)
  const [statusLabel, setStatusLabel] = useState<string | null>(null)
  const [dragover, setDragover] = useState(false)
  const [forceDisableDiarization, setForceDisableDiarization] = useState(false)

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
    const lower = f.name.toLowerCase()
    const isMkv =
      lower.endsWith('.mkv') ||
      lower.endsWith('.mka') ||
      f.type === 'video/x-matroska'
    // Chrome often cannot decode MKV in <video>; avoid extra reads / error noise.
    if (!isMkv && (f.type.startsWith('video/') || f.type.startsWith('audio/'))) {
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

  const stopPoll = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = null
    }
  }

  const runInference = async () => {
    if (!file || !token) return
    setJobActive(true)
    setError(null)
    setResult(null)
    setLiveTask(null)
    setStatusLabel('Постановка в очередь…')
    try {
      setStatusLabel('Конвертация в WAV…')
      const wavFile = await convertMediaToWav(file)
      setStatusLabel('Загрузка аудио…')
      const { task_id } = await uploadAudio(wavFile, token, {
        forceDisableDiarization,
      })
      setStatusLabel('В очереди. Ожидание результата…')
      const poll = () => {
        getTaskStatus(task_id, token).then((t) => {
          setLiveTask(t)
          setResult((prev) => ({
            taskId: task_id,
            transcription: t.result_transcription ?? prev?.transcription ?? null,
            summary: t.result_summary ?? prev?.summary ?? null,
          }))

          if (t.status === 'failed') {
            if (t.error_message) setError(t.error_message)
            else setError('Ошибка инференса')
            setStatusLabel(null)
            setJobActive(false)
            stopPoll()
            return
          }

          if (t.status === 'completed') {
            if (shouldKeepPollingAfterCompleted(t)) {
              setStatusLabel('Финализация диаризации…')
            } else {
              setStatusLabel(null)
              setJobActive(false)
              stopPoll()
            }
          }
        }).catch((e) => {
          const msg = e instanceof Error ? e.message : String(e)
          if (msg.includes('Not authenticated') || msg.includes('Invalid or expired token')) {
            logout()
          }
          setError(msg)
          setStatusLabel(null)
          setJobActive(false)
          stopPoll()
        })
      }
      poll()
      pollIntervalRef.current = setInterval(poll, 2000)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      if (msg.includes('Not authenticated') || msg.includes('Invalid or expired token')) {
        logout()
      }
      setError(msg)
      setStatusLabel(null)
      setJobActive(false)
    }
  }

  const progressMessage = taskProgressMessage(liveTask, statusLabel)

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
        disabled={!file || jobActive || !token}
        onClick={runInference}
      >
        {jobActive ? 'Обработка…' : 'Запустить транскрибацию и суммаризацию'}
      </button>
      <label style={{ display: 'block', marginTop: 12 }}>
        <input
          type="checkbox"
          checked={forceDisableDiarization}
          onChange={(e) => setForceDisableDiarization(e.target.checked)}
          disabled={jobActive}
          style={{ marginRight: 8 }}
        />
        Принудительно выключить диаризацию
      </label>
      {error && <p className="error">{error}</p>}
      {jobActive && (
        <p className="loading">
          {progressMessage ?? 'Подождите, это может занять несколько минут.'}
        </p>
      )}
      {result && (result.transcription != null || result.summary != null) && (
        <div className="result-block">
          <h3>Результат</h3>
          <ResultViewSwitch
            key={result.taskId}
            transcription={result.transcription}
            summary={result.summary}
          />
        </div>
      )}
    </div>
  )
}
