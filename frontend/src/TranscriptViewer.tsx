import { useMemo, useState } from 'react'

interface TranscriptEntry {
  id: string
  speaker: string
  timeLabel: string
  text: string
}

interface ApiTranscriptSegment {
  speaker?: unknown
  time_label?: unknown
  text?: unknown
}

interface ApiTranscriptPayload {
  segments?: unknown
}

const TIME_RANGE_RE = /^(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)$/
const TIME_SINGLE_RE = /^(\d{1,2}:\d{2}(?::\d{2})?)$/

function parseTranscript(raw: string): TranscriptEntry[] {
  const rows = raw
    .replace(/\r/g, '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)

  if (rows.length === 0) return []

  const entries: TranscriptEntry[] = []
  let currentSpeaker = 'Unknown'
  let fragmentIdx = 1

  for (const row of rows) {
    const bracketTokens = [...row.matchAll(/\[([^\]]+)\]/g)].map((m) => m[1].trim())
    let remainder = row.replace(/\[[^\]]+\]/g, '').trim()
    let speaker = currentSpeaker
    let timeLabel = `Фрагмент ${fragmentIdx}`

    for (const token of bracketTokens) {
      if (TIME_RANGE_RE.test(token) || TIME_SINGLE_RE.test(token)) {
        timeLabel = token
      } else {
        speaker = token
      }
    }

    if (!remainder && bracketTokens.length === 1 && !TIME_RANGE_RE.test(bracketTokens[0]) && !TIME_SINGLE_RE.test(bracketTokens[0])) {
      currentSpeaker = bracketTokens[0]
      continue
    }

    if (!remainder) remainder = row
    currentSpeaker = speaker
    entries.push({
      id: `seg-${entries.length}`,
      speaker,
      timeLabel,
      text: remainder,
    })
    fragmentIdx += 1
  }

  if (entries.length === 0) {
    return [
      {
        id: 'seg-0',
        speaker: 'Unknown',
        timeLabel: 'Фрагмент 1',
        text: raw.trim(),
      },
    ]
  }

  return entries
}

function fromJsonPayload(payload: ApiTranscriptPayload): TranscriptEntry[] {
  const rawSegments = payload.segments
  if (!Array.isArray(rawSegments)) return []

  const entries: TranscriptEntry[] = []
  rawSegments.forEach((raw, idx) => {
    const seg = raw as ApiTranscriptSegment
    const text = typeof seg.text === 'string' ? seg.text.trim() : ''
    if (!text) return
    const speaker = typeof seg.speaker === 'string' && seg.speaker.trim() ? seg.speaker : 'Unknown'
    const timeLabel = typeof seg.time_label === 'string' && seg.time_label.trim()
      ? seg.time_label
      : `Фрагмент ${idx + 1}`
    entries.push({
      id: `seg-${idx}`,
      speaker,
      timeLabel,
      text,
    })
  })

  return entries
}

function toTranscriptEntries(transcription: unknown): TranscriptEntry[] {
  if (!transcription) return []

  if (typeof transcription === 'string') {
    const raw = transcription.trim()
    if (!raw) return []
    try {
      const parsed = JSON.parse(raw) as ApiTranscriptPayload
      const fromJson = fromJsonPayload(parsed)
      if (fromJson.length > 0) return fromJson
    } catch {
      // plain text legacy format
    }
    return parseTranscript(raw)
  }

  if (typeof transcription === 'object') {
    const fromJson = fromJsonPayload(transcription as ApiTranscriptPayload)
    if (fromJson.length > 0) return fromJson
  }

  return []
}

function summaryValueToText(value: unknown): string {
  if (value == null) return ''
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' || typeof value === 'boolean') return String(value).trim()
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return String(value)
    }
  }
  return String(value).trim()
}

function parseSummary(summary: unknown): string {
  return summaryValueToText(summary)
}

export function TranscriptViewer({ transcription }: { transcription: unknown }) {
  const entries = useMemo(() => toTranscriptEntries(transcription), [transcription])
  const speakers = useMemo(
    () => Array.from(new Set(entries.map((x) => x.speaker))).filter(Boolean),
    [entries],
  )
  const [selectedSpeakers, setSelectedSpeakers] = useState<Set<string>>(new Set())
  const [openEntryId, setOpenEntryId] = useState<string | null>(entries[0]?.id ?? null)

  const filtered = useMemo(() => {
    if (selectedSpeakers.size === 0) return entries
    return entries.filter((x) => selectedSpeakers.has(x.speaker))
  }, [entries, selectedSpeakers])

  const toggleSpeaker = (speaker: string) => {
    setSelectedSpeakers((prev) => {
      const next = prev.size === 0 ? new Set(speakers) : new Set(prev)
      if (next.has(speaker)) next.delete(speaker)
      else next.add(speaker)
      return next
    })
  }

  const selectAll = () => setSelectedSpeakers(new Set(speakers))
  const clearAll = () => setSelectedSpeakers(new Set())

  if (entries.length === 0) {
    return <p className="muted">Транскрипция пуста.</p>
  }

  return (
    <div className="transcript-viewer">
      <div className="transcript-toolbar">
        <div className="speaker-chips">
          {speakers.map((speaker) => {
            const active = selectedSpeakers.size === 0 || selectedSpeakers.has(speaker)
            return (
              <button
                key={speaker}
                type="button"
                className={`speaker-chip ${active ? 'active' : ''}`}
                onClick={() => toggleSpeaker(speaker)}
              >
                {speaker}
              </button>
            )
          })}
        </div>
        <div className="transcript-toolbar-actions">
          <button type="button" className="btn btn-secondary btn-small" onClick={selectAll}>
            Все
          </button>
          <button type="button" className="btn btn-secondary btn-small" onClick={clearAll}>
            Сбросить
          </button>
        </div>
      </div>

      {filtered.length === 0 ? (
        <p className="muted small">Нет реплик для выбранных спикеров.</p>
      ) : (
        <div className="transcript-accordion">
          {filtered.map((entry) => {
            const expanded = openEntryId === entry.id
            return (
              <div key={entry.id} className={`transcript-item ${expanded ? 'expanded' : ''}`}>
                <button
                  type="button"
                  className="transcript-item-header"
                  onClick={() => setOpenEntryId(expanded ? null : entry.id)}
                >
                  <span className="transcript-time">{entry.timeLabel}</span>
                  <span className="transcript-speaker">{entry.speaker}</span>
                </button>
                {expanded && <div className="transcript-text">{entry.text}</div>}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export function SummaryViewer({ summary }: { summary: unknown }) {
  const parsedText = useMemo(() => parseSummary(summary), [summary])

  if (!parsedText) {
    return <p className="muted">Суммаризация пуста.</p>
  }

  return <pre className="transcript-raw-fallback">{parsedText}</pre>
}

type ResultViewTab = 'transcript' | 'summary'

export function ResultViewSwitch({
  transcription,
  summary,
}: {
  transcription: unknown
  summary: unknown
}) {
  const hasTranscript = transcription != null
  const hasSummary = summary != null

  const [tab, setTab] = useState<ResultViewTab>('transcript')

  if (!hasTranscript && !hasSummary) {
    return null
  }

  if (hasTranscript && !hasSummary) {
    return <TranscriptViewer transcription={transcription} />
  }

  if (!hasTranscript && hasSummary) {
    return <SummaryViewer summary={summary} />
  }

  return (
    <div className="result-view-switch">
      <div className="segmented-control" role="tablist" aria-label="Результат">
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'transcript'}
          className={`segmented-control-btn ${tab === 'transcript' ? 'active' : ''}`}
          onClick={() => setTab('transcript')}
        >
          Транскрипция
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'summary'}
          className={`segmented-control-btn ${tab === 'summary' ? 'active' : ''}`}
          onClick={() => setTab('summary')}
        >
          Суммаризация
        </button>
      </div>
      <div className="result-view-panel" role="tabpanel">
        {tab === 'transcript' ? (
          <TranscriptViewer transcription={transcription} />
        ) : (
          <SummaryViewer summary={summary} />
        )}
      </div>
    </div>
  )
}
