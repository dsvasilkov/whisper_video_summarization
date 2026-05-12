import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from 'react'
import type { TopicGraphNode, TopicGraphPayload } from './api'
import { LlmMarkdown } from './LlmMarkdown'
import { TopicMindMap } from './TopicMindMap'

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
      /* not JSON — одна строка целиком */
    }
    return [
      { id: 'seg-0', speaker: 'Unknown', timeLabel: 'Фрагмент 1', text: raw },
    ]
  }

  if (typeof transcription === 'object') {
    const fromJson = fromJsonPayload(transcription as ApiTranscriptPayload)
    if (fromJson.length > 0) return fromJson
  }

  return []
}

function TranscriptAccordionRow({
  entry,
  expanded,
  expandAll,
  setOpenEntryId,
  setExpandAll,
}: {
  entry: TranscriptEntry
  expanded: boolean
  expandAll: boolean
  setOpenEntryId: Dispatch<SetStateAction<string | null>>
  setExpandAll: Dispatch<SetStateAction<boolean>>
}) {
  const onHeaderClick = () => {
    if (expandAll) {
      setExpandAll(false)
      setOpenEntryId(entry.id)
      return
    }
    setOpenEntryId((id) => (id === entry.id ? null : entry.id))
  }

  return (
    <div className={`transcript-item ${expanded ? 'expanded' : ''}`}>
      <button
        type="button"
        className="transcript-item-header"
        onClick={onHeaderClick}
      >
        <span className="transcript-time">{entry.timeLabel}</span>
        <span className="transcript-speaker">{entry.speaker}</span>
      </button>
      {expanded ? <div className="transcript-text">{entry.text}</div> : null}
    </div>
  )
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

function formatMediaSeconds(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) return '—'
  const s = Math.max(0, sec)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const rs = Math.floor(s % 60)
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(rs).padStart(2, '0')}`
  return `${m}:${String(rs).padStart(2, '0')}`
}

function formatTimeRange(n: TopicGraphNode): string {
  const a = formatMediaSeconds(n.communityTimeStart ?? null)
  const b = formatMediaSeconds(n.communityTimeEnd ?? null)
  if (a === '—' && b === '—') return '—'
  return `${a} — ${b}`
}

function sortByCommunityStart(a: TopicGraphNode, b: TopicGraphNode): number {
  const ta = typeof a.communityTimeStart === 'number' ? a.communityTimeStart : Number.POSITIVE_INFINITY
  const tb = typeof b.communityTimeStart === 'number' ? b.communityTimeStart : Number.POSITIVE_INFINITY
  if (ta !== tb) return ta - tb
  return (a.label || '').localeCompare(b.label || '', 'ru')
}

function getLectureNode(g: TopicGraphPayload): TopicGraphNode | null {
  return g.nodes.find((n) => n.kind === 'lecture') ?? null
}

function getThemeNodes(graph: TopicGraphPayload): TopicGraphNode[] {
  const lec = getLectureNode(graph)
  if (lec) {
    const list = graph.nodes.filter((n) => n.kind === 'theme' && n.parentId === lec.id)
    if (list.length) return [...list].sort(sortByCommunityStart)
  }
  const themes = graph.nodes.filter((n) => n.kind === 'theme')
  return [...themes].sort(sortByCommunityStart)
}

function getSubtopicsForTheme(graph: TopicGraphPayload, themeId: string): TopicGraphNode[] {
  return graph.nodes
    .filter((n) => n.kind === 'subtopic' && n.parentId === themeId)
    .sort(sortByCommunityStart)
}

function communityBodyText(n: TopicGraphNode): string {
  return (n.communityBody ?? '').trim()
}

/** Текст вкладки «резюме» (как buildTooltipText в mind map). */
function nodeSummaryTooltipText(n: TopicGraphNode): string {
  const sum = (n.summary ?? '').trim()
  const desc = (n.description ?? '').trim()
  const kws = (n.keywords ?? []).map((x) => String(x).trim()).filter(Boolean)
  const pieces: (string | null)[] = [sum || null]
  if (desc && desc !== sum) pieces.push(desc)
  if (kws.length > 0) pieces.push(`Ключевые слова: ${kws.join(', ')}`)
  const body = pieces.filter(Boolean).join('\n\n').trim()
  return body || 'Нет текста резюме для узла.'
}

function SummarySourceToggle({
  tab,
  onTab,
  hasSource,
}: {
  tab: 'summary' | 'source'
  onTab: (t: 'summary' | 'source') => void
  hasSource: boolean
}) {
  if (!hasSource) return null
  return (
    <div className="mindmap-node-tooltip-toggle hierarchy-inline-toggle" role="tablist" aria-label="Режим текста">
      <button
        type="button"
        role="tab"
        aria-selected={tab === 'summary'}
        title="Резюме"
        className={`mindmap-tooltip-dot ${tab === 'summary' ? 'mindmap-tooltip-dot--active' : ''}`}
        onClick={(e) => {
          e.stopPropagation()
          onTab('summary')
        }}
      />
      <button
        type="button"
        role="tab"
        aria-selected={tab === 'source'}
        title="Текст сообщества"
        className={`mindmap-tooltip-dot ${tab === 'source' ? 'mindmap-tooltip-dot--active' : ''}`}
        onClick={(e) => {
          e.stopPropagation()
          onTab('source')
        }}
      />
    </div>
  )
}

/** Если LLM вернул пустую строку, показываем темы из mind map как текстовое резюме. */
function textFromTopicGraph(g: TopicGraphPayload): string {
  return g.nodes
    .filter(
      (n) =>
        n.kind !== 'macro' &&
        n.kind !== 'lecture' &&
        n.kind !== 'theme' &&
        n.kind !== 'subtopic',
    )
    .map((n) => {
      const title = (n.label || '').trim()
      const body = (n.summary || n.description || '').trim()
      if (title && body && body !== title) return `${title}\n${body}`
      return title || body
    })
    .filter(Boolean)
    .join('\n\n')
}

export function TranscriptViewer({ transcription }: { transcription: unknown }) {
  const entries = useMemo(() => toTranscriptEntries(transcription), [transcription])
  const speakers = useMemo(
    () => Array.from(new Set(entries.map((x) => x.speaker))).filter(Boolean),
    [entries],
  )
  const [selectedSpeakers, setSelectedSpeakers] = useState<Set<string>>(new Set())
  const [openEntryId, setOpenEntryId] = useState<string | null>(entries[0]?.id ?? null)
  const [expandAll, setExpandAll] = useState(false)

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
          <button
            type="button"
            className="btn btn-secondary btn-small"
            onClick={() => setExpandAll((x) => !x)}
          >
            {expandAll ? 'Свернуть всё' : 'Раскрыть всё'}
          </button>
        </div>
      </div>

      {filtered.length === 0 ? (
        <p className="muted small">Нет реплик для выбранных спикеров.</p>
      ) : (
        <div className="transcript-accordion">
          {filtered.map((entry) => {
            const expanded = expandAll || openEntryId === entry.id
            return (
              <TranscriptAccordionRow
                key={entry.id}
                entry={entry}
                expanded={expanded}
                expandAll={expandAll}
                setOpenEntryId={setOpenEntryId}
                setExpandAll={setExpandAll}
              />
            )
          })}
        </div>
      )}
    </div>
  )
}

type SummaryDepth = 0 | 1 | 2

/** Подпись уровня ползунка (подсказка слева и aria-valuetext). */
const SUMMARY_DEPTH_LABEL: readonly string[] = ['Подтемы', 'Темы', 'Итог лекции']

function HierarchySubtopicRow({
  node,
  expanded,
  expandAll,
  setOpenId,
  setExpandAll,
}: {
  node: TopicGraphNode
  expanded: boolean
  expandAll: boolean
  setOpenId: Dispatch<SetStateAction<string | null>>
  setExpandAll: Dispatch<SetStateAction<boolean>>
}) {
  const [tab, setTab] = useState<'summary' | 'source'>('summary')
  const summaryMd = nodeSummaryTooltipText(node)
  const source = communityBodyText(node)
  const hasSource = !!source

  useEffect(() => {
    if (!expanded) setTab('summary')
  }, [expanded])

  const onHeaderClick = () => {
    if (expandAll) {
      setExpandAll(false)
      setOpenId(node.id)
      return
    }
    setOpenId((id) => (id === node.id ? null : node.id))
  }

  return (
    <div className={`transcript-item hierarchy-item ${expanded ? 'expanded' : ''}`}>
      <button
        type="button"
        className="transcript-item-header"
        onClick={onHeaderClick}
      >
        <span className="transcript-speaker hierarchy-item-label hierarchy-subtopic-title">
          {node.label}
        </span>
        <span className="transcript-time hierarchy-subtopic-times">{formatTimeRange(node)}</span>
      </button>
      {expanded ? (
        <div className="hierarchy-expanded-block">
          <div className="mindmap-node-tooltip-body hierarchy-expanded-scroll">
            {tab === 'summary' ? (
              <LlmMarkdown text={summaryMd} className="mindmap-tooltip-md" />
            ) : (
              <div className="mindmap-tooltip-plain hierarchy-source-text">{source}</div>
            )}
          </div>
          <SummarySourceToggle tab={tab} onTab={setTab} hasSource={hasSource} />
        </div>
      ) : null}
    </div>
  )
}

function SummarySubtopicsPanel({
  graph,
  expandAll,
  setExpandAll,
}: {
  graph: TopicGraphPayload
  expandAll: boolean
  setExpandAll: Dispatch<SetStateAction<boolean>>
}) {
  const themes = useMemo(() => getThemeNodes(graph), [graph])
  const [openId, setOpenId] = useState<string | null>(null)

  const totalSubs = useMemo(
    () => themes.reduce((acc, th) => acc + getSubtopicsForTheme(graph, th.id).length, 0),
    [graph, themes],
  )

  if (!themes.length) {
    return <p className="muted small">В графе нет тем.</p>
  }
  if (totalSubs === 0) {
    return <p className="muted small">Подтемы не найдены.</p>
  }

  return (
    <div className="hierarchy-accordion">
      {themes.flatMap((theme) => {
        const subs = getSubtopicsForTheme(graph, theme.id)
        if (!subs.length) return []
        const themeTitle = (theme.label || '').trim() || 'Тема'
        return [
          <section key={theme.id} className="hierarchy-theme-block">
            <h4 className="hierarchy-theme-heading">
              {themeTitle}
            </h4>
            <div className="transcript-accordion">
              {subs.map((st) => {
                const expanded = expandAll || openId === st.id
                return (
                  <HierarchySubtopicRow
                    key={st.id}
                    node={st}
                    expanded={expanded}
                    expandAll={expandAll}
                    setOpenId={setOpenId}
                    setExpandAll={setExpandAll}
                  />
                )
              })}
            </div>
          </section>,
        ]
      })}
    </div>
  )
}

function HierarchyThemeRow({
  node,
  expanded,
  expandAll,
  setOpenId,
  setExpandAll,
}: {
  node: TopicGraphNode
  expanded: boolean
  expandAll: boolean
  setOpenId: Dispatch<SetStateAction<string | null>>
  setExpandAll: Dispatch<SetStateAction<boolean>>
}) {
  const [tab, setTab] = useState<'summary' | 'source'>('summary')
  const summaryMd = nodeSummaryTooltipText(node)
  const source = communityBodyText(node)
  const hasSource = !!source

  useEffect(() => {
    if (!expanded) setTab('summary')
  }, [expanded])

  const onHeaderClick = () => {
    if (expandAll) {
      setExpandAll(false)
      setOpenId(node.id)
      return
    }
    setOpenId((id) => (id === node.id ? null : node.id))
  }

  return (
    <div className={`transcript-item hierarchy-item ${expanded ? 'expanded' : ''}`}>
      <button
        type="button"
        className="transcript-item-header hierarchy-item-header"
        onClick={onHeaderClick}
      >
        <div className="hierarchy-item-header-top">
          <span className="hierarchy-row-theme">{node.label}</span>
          <span className="hierarchy-row-times">
            Тема: {formatMediaSeconds(node.communityTimeStart)} — {formatMediaSeconds(node.communityTimeEnd)}
          </span>
        </div>
      </button>
      {expanded ? (
        <div className="hierarchy-expanded-block">
          <div className="mindmap-node-tooltip-body hierarchy-expanded-scroll">
            {tab === 'summary' ? (
              <LlmMarkdown text={summaryMd} className="mindmap-tooltip-md" />
            ) : (
              <div className="mindmap-tooltip-plain hierarchy-source-text">{source}</div>
            )}
          </div>
          <SummarySourceToggle tab={tab} onTab={setTab} hasSource={hasSource} />
        </div>
      ) : null}
    </div>
  )
}

function SummaryThemesPanel({
  graph,
  expandAll,
  setExpandAll,
}: {
  graph: TopicGraphPayload
  expandAll: boolean
  setExpandAll: Dispatch<SetStateAction<boolean>>
}) {
  const themes = useMemo(() => getThemeNodes(graph), [graph])
  const [openId, setOpenId] = useState<string | null>(null)

  if (!themes.length) {
    return <p className="muted small">В графе нет тем.</p>
  }

  return (
    <div className="transcript-accordion hierarchy-accordion">
      {themes.map((t) => {
        const expanded = expandAll || openId === t.id
        return (
          <HierarchyThemeRow
            key={t.id}
            node={t}
            expanded={expanded}
            expandAll={expandAll}
            setOpenId={setOpenId}
            setExpandAll={setExpandAll}
          />
        )
      })}
    </div>
  )
}

function SummaryLecturePanel({
  graph,
  fallbackText,
}: {
  graph: TopicGraphPayload
  fallbackText: string
}) {
  const lec = useMemo(() => getLectureNode(graph), [graph])
  const [tab, setTab] = useState<'summary' | 'source'>('summary')

  const summaryMd = useMemo(() => {
    if (lec) return nodeSummaryTooltipText(lec)
    return (fallbackText || '').trim()
  }, [lec, fallbackText])

  const source = lec ? communityBodyText(lec) : ''
  const hasSource = !!source

  if (!lec) {
    const t = (fallbackText || '').trim()
    if (!t) {
      return <p className="muted small">Нет итогового текста лекции.</p>
    }
    return (
      <div className="transcript-md-panel hierarchy-lecture-md">
        <LlmMarkdown text={t} />
      </div>
    )
  }

  return (
    <div className="hierarchy-lecture-panel">
      <div className="hierarchy-item-header-top hierarchy-lecture-header-top">
        <span className="hierarchy-row-theme">{(lec.label || '').trim() || 'Лекция'}</span>
        <span className="hierarchy-row-times">
          Лекция: {formatMediaSeconds(lec.communityTimeStart)} — {formatMediaSeconds(lec.communityTimeEnd)}
        </span>
      </div>
      <div className="hierarchy-expanded-block">
        <div className="mindmap-node-tooltip-body hierarchy-expanded-scroll hierarchy-lecture-scroll">
          {tab === 'summary' ? (
            <LlmMarkdown text={summaryMd} className="mindmap-tooltip-md" />
          ) : (
            <div className="mindmap-tooltip-plain hierarchy-source-text">{source || 'Нет текста сообщества.'}</div>
          )}
        </div>
        <SummarySourceToggle tab={tab} onTab={setTab} hasSource={hasSource} />
      </div>
    </div>
  )
}

export function isTopicGraph(v: unknown): v is TopicGraphPayload {
  if (!v || typeof v !== 'object') return false
  const o = v as { nodes?: unknown }
  return Array.isArray(o.nodes) && o.nodes.length > 0
}

export function SummaryViewer({
  summary,
  topicGraph,
}: {
  summary: unknown
  topicGraph?: TopicGraphPayload | null
}) {
  const parsedText = useMemo(() => parseSummary(summary), [summary])
  const hasGraph = isTopicGraph(topicGraph)
  const displayText = useMemo(() => {
    const t = parsedText.trim()
    if (t) return t
    if (hasGraph && topicGraph) return textFromTopicGraph(topicGraph).trim()
    return ''
  }, [parsedText, hasGraph, topicGraph])

  const [summaryDepth, setSummaryDepth] = useState<SummaryDepth>(0)
  const [mindMapOpen, setMindMapOpen] = useState(false)
  const [hierarchyExpandAll, setHierarchyExpandAll] = useState(false)

  const graphKey = topicGraph?.nodes.map((n) => n.id).join('|') ?? ''
  useEffect(() => {
    setHierarchyExpandAll(false)
  }, [summaryDepth, graphKey])

  if (!displayText && !hasGraph) {
    return <p className="muted">Суммаризация пуста.</p>
  }

  return (
    <div
      className={`summary-viewer ${hasGraph && mindMapOpen ? 'summary-viewer--mindmap-full' : ''}`}
    >
      {hasGraph && (
        <div className="summary-toolbar" role="toolbar" aria-label="Суммаризация">
          <div className="summary-depth-group">
            <p className="summary-depth-hint muted small" aria-live="polite">
              {SUMMARY_DEPTH_LABEL[summaryDepth] ?? ''}
            </p>
            <div className="summary-depth-rail">
              <input
                type="range"
                min={0}
                max={2}
                step={1}
                value={summaryDepth}
                onChange={(e) => {
                  setSummaryDepth(Number(e.target.value) as SummaryDepth)
                  setMindMapOpen(false)
                }}
                className="summary-depth-slider"
                aria-label="Уровень суммаризации"
                aria-valuenow={summaryDepth}
                aria-valuemin={0}
                aria-valuemax={2}
                aria-valuetext={SUMMARY_DEPTH_LABEL[summaryDepth] ?? ''}
              />
              <div className="summary-depth-markers" aria-hidden>
                <span />
                <span />
                <span />
              </div>
            </div>
          </div>
          {!mindMapOpen && summaryDepth !== 2 ? (
            <button
              type="button"
              className="btn btn-secondary btn-small summary-expand-all-btn"
              onClick={() => setHierarchyExpandAll((x) => !x)}
            >
              {hierarchyExpandAll ? 'Свернуть всё' : 'Раскрыть всё'}
            </button>
          ) : null}
          <button
            type="button"
            className="btn btn-secondary btn-small summary-mindmap-btn"
            onClick={() => setMindMapOpen(true)}
          >
            Mind map
          </button>
        </div>
      )}
      {mindMapOpen && hasGraph && topicGraph ? (
        <TopicMindMap graph={topicGraph} onExitViewportFullscreen={() => setMindMapOpen(false)} />
      ) : hasGraph && topicGraph && summaryDepth === 0 ? (
        <SummarySubtopicsPanel
          graph={topicGraph}
          expandAll={hierarchyExpandAll}
          setExpandAll={setHierarchyExpandAll}
        />
      ) : hasGraph && topicGraph && summaryDepth === 1 ? (
        <SummaryThemesPanel
          graph={topicGraph}
          expandAll={hierarchyExpandAll}
          setExpandAll={setHierarchyExpandAll}
        />
      ) : hasGraph && topicGraph && summaryDepth === 2 ? (
        <SummaryLecturePanel graph={topicGraph} fallbackText={displayText} />
      ) : displayText ? (
        <div className="transcript-md-panel">
          <LlmMarkdown text={displayText} />
        </div>
      ) : (
        <p className="muted small">Переключите уровень ползунка или дождитесь суммаризации.</p>
      )}
    </div>
  )
}

type ResultViewTab = 'transcript' | 'summary'

export function ResultViewSwitch({
  transcription,
  summary,
  topicGraph,
}: {
  transcription: unknown
  summary: unknown
  topicGraph?: TopicGraphPayload | null
}) {
  const hasTranscript = transcription != null
  const hasSummary = summary != null || isTopicGraph(topicGraph)

  const [tab, setTab] = useState<ResultViewTab>('transcript')

  if (!hasTranscript && !hasSummary) {
    return null
  }

  if (hasTranscript && !hasSummary) {
    return <TranscriptViewer transcription={transcription} />
  }

  if (!hasTranscript && hasSummary) {
    return <SummaryViewer summary={summary} topicGraph={topicGraph} />
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
          <SummaryViewer summary={summary} topicGraph={topicGraph} />
        )}
      </div>
    </div>
  )
}
