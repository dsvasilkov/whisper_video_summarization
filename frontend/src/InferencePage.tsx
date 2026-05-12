import { useCallback, useEffect, useRef, useState } from 'react'
import type { TaskStatusResponse, TopicGraphPayload } from './api'
import { getTaskStatus, subscribeTaskEvents, uploadAudio } from './api'
import { useAuth } from './AuthContext'
import { convertMediaToWav } from './audio'
import { TaskQaForm } from './TaskQaForm'
import { isTopicGraph, ResultViewSwitch } from './TranscriptViewer'

const ACCEPT = '.mp4,.wav,.mp3,.mkv'

type TranscriptMeta = {
  asr_done?: boolean
  diarization_ready?: boolean
  diarization_skipped?: boolean
  merge_done?: boolean
}

function hasTranscriptionContent(transcription: unknown): boolean {
  if (!transcription) return false
  if (typeof transcription === 'string') return transcription.trim().length > 0
  if (typeof transcription !== 'object') return false
  const t = transcription as { text?: unknown; segments?: unknown; _meta?: unknown }
  if (typeof t.text === 'string' && t.text.trim().length > 0) return true
  return Array.isArray(t.segments) && t.segments.length > 0
}

function extractTranscriptMeta(transcription: unknown): TranscriptMeta {
  if (!transcription || typeof transcription !== 'object') return {}
  const maybeMeta = (transcription as { _meta?: unknown })._meta
  if (!maybeMeta || typeof maybeMeta !== 'object') return {}
  return maybeMeta as TranscriptMeta
}

/**
 * Диаризация включена автоматически на стороне бэкенда.
 *
 * Иногда при `completed` UI может получить payload, где транскрипт ещё не подгрузился
 * или флаг merge ещё не отражён в `_meta`. В этом случае показываем "финализацию".
 */
function shouldWaitForDiarizationFinalization(task: TaskStatusResponse): boolean {
  if (task.status !== 'completed') return false
  if (task.result_transcription == null) return true

  const meta = extractTranscriptMeta(task.result_transcription)
  if (meta.diarization_skipped === true) return false

  // If ASR is done but speaker merge not reflected yet, keep showing finalization state.
  if (meta.asr_done === true && meta.merge_done !== true) return true
  return false
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
  if (task.status === 'completed' && shouldWaitForDiarizationFinalization(task)) {
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
    topicGraph: TopicGraphPayload | null
  } | null>(null)
  const [statusLabel, setStatusLabel] = useState<string | null>(null)
  const [dragover, setDragover] = useState(false)

  const activeTaskIdRef = useRef<string | null>(null)
  const taskEventsCloseRef = useRef<null | (() => void)>(null)
  const completedWaitStartedAtRef = useRef<number | null>(null)
  /** Сбрасывает устаревшие ответы getTaskStatus при всплеске SSE (иначе поздний JSON может затереть промежуточный статус). */
  const taskStatusFetchGenRef = useRef(0)
  const taskFetchDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const stopLiveUpdates = () => {
    if (taskFetchDebounceRef.current) {
      clearTimeout(taskFetchDebounceRef.current)
      taskFetchDebounceRef.current = null
    }
    if (taskEventsCloseRef.current) {
      taskEventsCloseRef.current()
      taskEventsCloseRef.current = null
    }
  }

  const onFileChange = useCallback((f: File | null) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    // Если выбирают новый файл во время обработки другой задачи — возвращаем UI в начальное состояние.
    stopLiveUpdates()
    activeTaskIdRef.current = null
    completedWaitStartedAtRef.current = null
    setJobActive(false)
    setLiveTask(null)
    setStatusLabel(null)
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
    // Позволяет выбрать тот же файл повторно (иначе onChange не сработает).
    e.target.value = ''
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

  const runInference = async () => {
    if (!file || !token) return
    // Safety: never allow multiple concurrent SSE subscriptions.
    stopLiveUpdates()
    activeTaskIdRef.current = null
    setJobActive(true)
    setError(null)
    setResult(null)
    setLiveTask(null)
    setStatusLabel('Постановка в очередь…')
    try {
      setStatusLabel('Конвертация в WAV…')
      const wavFile = await convertMediaToWav(file)
      setStatusLabel('Загрузка аудио…')
      const { task_id } = await uploadAudio(wavFile, token)
      activeTaskIdRef.current = task_id
      setStatusLabel('В очереди. Ожидание результата…')

      const finishOk = () => {
        setStatusLabel(null)
        setJobActive(false)
        activeTaskIdRef.current = null
        completedWaitStartedAtRef.current = null
        stopLiveUpdates()
      }

      const ingestFullTask = (t: TaskStatusResponse) => {
        if (activeTaskIdRef.current !== task_id) return

        setLiveTask(t)
        setResult((prev) => ({
          taskId: task_id,
          transcription: hasTranscriptionContent(t.result_transcription)
            ? (t.result_transcription ?? null)
            : (prev?.transcription ?? null),
          summary: t.result_summary ?? prev?.summary ?? null,
          topicGraph: t.result_topic_graph ?? prev?.topicGraph ?? null,
        }))

        if (t.status === 'failed') {
          setError(t.error_message || 'Ошибка инференса')
          finishOk()
          return
        }

        if (t.status === 'completed') {
          if (!shouldWaitForDiarizationFinalization(t)) {
            finishOk()
            return
          }
          // Backend says completed; avoid hanging forever on UI-side "finalization" heuristics.
          const now = Date.now()
          if (completedWaitStartedAtRef.current == null) completedWaitStartedAtRef.current = now
          if (now - completedWaitStartedAtRef.current > 15_000) {
            finishOk()
          }
        } else {
          completedWaitStartedAtRef.current = null
        }
      }

      const fetchAndIngest = () => {
        const gen = ++taskStatusFetchGenRef.current
        void getTaskStatus(task_id, token)
          .then((t) => {
            if (activeTaskIdRef.current !== task_id) return
            if (gen !== taskStatusFetchGenRef.current) return
            ingestFullTask(t)
          })
          .catch((e) => {
            const msg = e instanceof Error ? e.message : String(e)
            if (msg.includes('Not authenticated') || msg.includes('Invalid or expired token')) {
              logout()
              finishOk()
              return
            }
            setError(msg)
          })
      }

      const scheduleFetchFromSse = () => {
        if (taskFetchDebounceRef.current) clearTimeout(taskFetchDebounceRef.current)
        taskFetchDebounceRef.current = setTimeout(() => {
          taskFetchDebounceRef.current = null
          fetchAndIngest()
        }, 80)
      }

      // Сначала SSE, чтобы не пропустить publish до SUBSCRIBE (воркер может быстро дать processing).
      const sub = subscribeTaskEvents(
        task_id,
        token,
        (ev) => {
          if (activeTaskIdRef.current !== task_id) return

          setLiveTask((prev) => {
            if (!prev) {
              return {
                task_id: ev.task_id,
                status: ev.status,
                task_type: ev.task_type,
                result_transcription: null,
                result_summary: null,
                result_topic_graph: null,
                error_message: ev.error_message,
                created_at: new Date().toISOString(),
                updated_at: ev.updated_at ?? new Date().toISOString(),
              }
            }
            return {
              ...prev,
              status: ev.status,
              task_type: ev.task_type,
              error_message: ev.error_message,
              updated_at: ev.updated_at ?? prev.updated_at,
            }
          })

          if (ev.status === 'failed') {
            setError(ev.error_message || 'Ошибка инференса')
            if (taskFetchDebounceRef.current) {
              clearTimeout(taskFetchDebounceRef.current)
              taskFetchDebounceRef.current = null
            }
            fetchAndIngest()
            finishOk()
            return
          }

          scheduleFetchFromSse()
        },
        (e) => {
          const msg = e instanceof Error ? e.message : String(e)
          if (msg.includes('Not authenticated') || msg.includes('Invalid or expired token')) {
            logout()
            finishOk()
            return
          }
        },
      )
      taskEventsCloseRef.current = sub.close
      fetchAndIngest()
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      if (msg.includes('Not authenticated') || msg.includes('Invalid or expired token')) {
        logout()
      }
      setError(msg)
      setStatusLabel(null)
      setJobActive(false)
      activeTaskIdRef.current = null
    }
  }

  const progressMessage = taskProgressMessage(liveTask, statusLabel)

  useEffect(() => () => {
    stopLiveUpdates()
    activeTaskIdRef.current = null
  }, [])

  return (
    <div className="card">
      <h2>Инференс видео</h2>
      <div
        className={`upload-area ${dragover ? 'dragover' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => {
          const el = document.getElementById('inference-file') as HTMLInputElement | null
          if (el) {
            // Чтобы выбор "того же" файла снова триггерил onChange.
            el.value = ''
            el.click()
          }
        }}
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
      {error && <p className="error">{error}</p>}
      {jobActive && (
        <p className="loading">
          {progressMessage ?? 'Подождите, это может занять несколько минут.'}
        </p>
      )}
      {result &&
        (result.transcription != null || result.summary != null || isTopicGraph(result.topicGraph)) && (
        <div className="result-block">
          <h3>Результат</h3>
          <ResultViewSwitch
            key={result.taskId}
            transcription={result.transcription}
            summary={result.summary}
            topicGraph={result.topicGraph}
          />
        </div>
      )}
      {token &&
        liveTask?.status === 'completed' &&
        result?.taskId &&
        (result.transcription != null || result.summary != null) && (
          <TaskQaForm
            taskId={result.taskId}
            token={token}
            onAuthError={() => logout()}
          />
        )}
    </div>
  )
}
