import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from 'react'
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
  type NodeProps,
} from 'reactflow'
import 'reactflow/dist/style.css'
import type { TopicGraphNode, TopicGraphPayload } from './api'
import { LlmMarkdown } from './LlmMarkdown'

/** Узлам без координат из радиальной функции сохраняем позиции с сервера. */
function computeRadialPositions(
  graph: TopicGraphPayload,
  visibleIds: Set<string>,
): Map<string, { x: number; y: number }> {
  const pos = new Map<string, { x: number; y: number }>()
  const byId = new Map(graph.nodes.map((n) => [n.id, n]))
  const children = new Map<string, TopicGraphNode[]>()
  for (const n of graph.nodes) {
    const p = n.parentId
    if (!p || !byId.has(p)) continue
    if (!children.has(p)) children.set(p, [])
    children.get(p)!.push(n)
  }

  const sortSiblings = (arr: TopicGraphNode[]) =>
    [...arr].sort((a, b) => {
      const ay = a.position?.y ?? 0
      const by = b.position?.y ?? 0
      if (ay !== by) return ay - by
      return (a.position?.x ?? 0) - (b.position?.x ?? 0)
    })

  interface PlaceCtx {
    centerX: number
    centerY: number
  }

  function radiusFor(parentKind: TopicGraphNode['kind'] | undefined): number {
    if (parentKind === 'lecture') return 400
    if (parentKind === 'theme') return 280
    if (parentKind === 'subtopic') return 200
    if (parentKind === 'macro') return 260
    return 220
  }

  function placeChildren(
    parentId: string,
    px: number,
    py: number,
    parentKind: TopicGraphNode['kind'] | undefined,
    ctx: PlaceCtx,
  ) {
    const ch = sortSiblings(children.get(parentId) ?? []).filter((c) => visibleIds.has(c.id))
    if (!ch.length) return
    const n = ch.length
    const R = radiusFor(parentKind)
    const atOrigin = Math.hypot(px - ctx.centerX, py - ctx.centerY) < 1e-3
    const fullRing = parentKind === 'lecture' || atOrigin

    if (fullRing) {
      for (let i = 0; i < n; i++) {
        const ang = -Math.PI / 2 + (2 * Math.PI * i) / Math.max(n, 1)
        const x = px + R * Math.cos(ang)
        const y = py + R * Math.sin(ang)
        const child = ch[i]
        pos.set(child.id, { x, y })
        placeChildren(child.id, x, y, child.kind ?? 'topic', ctx)
      }
      return
    }

    const outward = Math.atan2(py - ctx.centerY, px - ctx.centerX)
    const spread = Math.min(Math.PI * 0.9, Math.PI * 0.45 + 0.18 * Math.max(n - 1, 0))
    const startAng = outward - spread / 2
    const endAng = outward + spread / 2
    for (let i = 0; i < n; i++) {
      const child = ch[i]
      const t = n === 1 ? 0.5 : i / (n - 1)
      const ang = startAng + t * (endAng - startAng)
      const x = px + R * Math.cos(ang)
      const y = py + R * Math.sin(ang)
      pos.set(child.id, { x, y })
      placeChildren(child.id, x, y, child.kind ?? 'topic', ctx)
    }
  }

  const roots = graph.nodes.filter(
    (node) => visibleIds.has(node.id) && (!node.parentId || !byId.has(node.parentId)),
  )
  if (!roots.length) return pos

  const lecture = roots.find((r) => r.kind === 'lecture')
  const ctx: PlaceCtx = { centerX: 0, centerY: 0 }

  if (lecture) {
    pos.set(lecture.id, { x: 0, y: 0 })
    placeChildren(lecture.id, 0, 0, 'lecture', ctx)
    return pos
  }

  if (roots.length === 1) {
    const r = roots[0]
    pos.set(r.id, { x: 0, y: 0 })
    placeChildren(r.id, 0, 0, r.kind ?? 'topic', ctx)
    return pos
  }

  const Rr = 420
  roots.forEach((r, i) => {
    const ang = -Math.PI / 2 + (2 * Math.PI * i) / roots.length
    const x = Rr * Math.cos(ang)
    const y = Rr * Math.sin(ang)
    pos.set(r.id, { x, y })
    placeChildren(r.id, x, y, r.kind ?? 'topic', ctx)
  })
  return pos
}

const KW_NODE_SEP = '::keyword::'

function formatMediaSeconds(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) return '—'
  const s = Math.max(0, sec)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const rs = Math.floor(s % 60)
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(rs).padStart(2, '0')}`
  return `${m}:${String(rs).padStart(2, '0')}`
}

const COMMUNITY_COLORS = [
  '#3b82f6',
  '#22c55e',
  '#a855f7',
  '#f97316',
  '#ec4899',
  '#14b8a6',
  '#eab308',
  '#64748b',
]

type TopicNodeData = {
  label: string
  summary: string
  /** Полный текст для подсказки по двойному щелчку */
  tooltipText: string
  /** Реплики сообщества подтемы (если есть с бэкенда) — вкладка «источник» во всплывающем окне */
  communityBody?: string
  /** Временной диапазон узла в записи (секунды). Доступен для подтем, тем и узла «Лекция». */
  communityTimeStart?: number | null
  communityTimeEnd?: number | null
  community: number
  keywords?: string[]
  kind?: 'lecture' | 'theme' | 'subtopic' | 'micro' | 'macro' | 'topic' | 'keyword'
  expandable?: boolean
  expanded?: boolean
  onToggleExpand?: () => void
}

function TopicFlowNode({ data, selected }: NodeProps<TopicNodeData>) {
  const c = COMMUNITY_COLORS[Math.abs(data.community) % COMMUNITY_COLORS.length]
  const isKeyword = data.kind === 'keyword'
  const isMacro = data.kind === 'macro'
  const branchVariant =
    data.kind === 'lecture'
      ? 'topic-flow-node--lecture'
      : data.kind === 'theme'
        ? 'topic-flow-node--theme'
        : data.kind === 'subtopic'
          ? 'topic-flow-node--subtopic'
          : isMacro
            ? 'topic-flow-node--macro'
            : ''
  const collapsedBranch =
    data.expandable && data.expanded === false &&
    (data.kind === 'lecture' ||
      data.kind === 'theme' ||
      data.kind === 'subtopic' ||
      isMacro)
      ? ' topic-flow-node--branch-collapsed'
      : ''

  return (
    <div
      className={`topic-flow-node topic-flow-node--circle ${branchVariant} ${isKeyword ? 'topic-flow-node--keyword' : ''} ${selected ? 'topic-flow-node--selected' : ''} ${data.expandable ? 'topic-flow-node--expandable' : ''} ${data.expandable ? 'topic-flow-node--click-toggle' : ''}${collapsedBranch}`}
      style={{
        borderColor: isKeyword
          ? '#71717a'
          : data.kind === 'lecture'
            ? '#e2e8f0'
            : data.kind === 'theme'
              ? '#94a3b8'
              : data.kind === 'subtopic'
                ? '#64748b'
                : isMacro
                  ? '#94a3b8'
                  : c,
      }}
    >
      <Handle type="target" position={Position.Left} className="topic-flow-handle" />
      <div className="topic-flow-node-title">{data.label}</div>
      <Handle type="source" position={Position.Right} className="topic-flow-handle" />
    </div>
  )
}

const nodeTypes = { topic: TopicFlowNode }

function buildTooltipText(node: TopicGraphNode): string {
  const sum = (node.summary ?? '').trim()
  const desc = (node.description ?? '').trim()
  const kws = (node.keywords ?? []).map((x) => String(x).trim()).filter(Boolean)
  const pieces: (string | null)[] = [sum || null]
  if (desc && desc !== sum) pieces.push(desc)
  if (kws.length > 0) pieces.push(`Ключевые слова: ${kws.join(', ')}`)
  const body = pieces.filter(Boolean).join('\n\n').trim()
  return body || 'Нет текста резюме для узла.'
}

function _hasHierarchyContainers(graph: TopicGraphPayload): boolean {
  return graph.nodes.some(
    (n) =>
      (n.kind === 'lecture' ||
        n.kind === 'theme' ||
        n.kind === 'subtopic' ||
        n.kind === 'macro') &&
      graph.nodes.some((c) => c.parentId === n.id),
  )
}

/** Узел виден, если раскрыты все предки-контейнеры (лекция → тема → подтема/макро). */
function buildVisibleTopicIds(graph: TopicGraphPayload, expandedBranches: ReadonlySet<string>): Set<string> {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]))
  if (!_hasHierarchyContainers(graph)) {
    const out = new Set<string>()
    graph.nodes.forEach((n) => out.add(n.id))
    return out
  }
  const visible = new Set<string>()
  for (const n of graph.nodes) {
    let pid: string | null | undefined = n.parentId
    let show = true
    while (pid) {
      if (!expandedBranches.has(pid)) {
        show = false
        break
      }
      const p = byId.get(pid)
      pid = p?.parentId ?? null
    }
    if (show) visible.add(n.id)
  }
  return visible
}

function graphToFlow(
  graph: TopicGraphPayload,
  visibleIds: Set<string>,
  expandedBranches: Set<string>,
  expandedTopics: Set<string>,
  toggleBranch: (id: string) => void,
  toggleTopicExpand: (id: string) => void,
  linearLectureLayout: boolean,
): { nodes: Node<TopicNodeData>[]; edges: Edge[] } {
  const rawLinks = (graph.links ?? []).filter((e) => {
    if (linearLectureLayout) return true
    const t = String(e.type || '')
    return t !== 'timeline' && t !== 'follows'
  })
  const radialPos = linearLectureLayout ? null : computeRadialPositions(graph, visibleIds)
  const nodes: Node<TopicNodeData>[] = []
  const extraEdges: Edge[] = []

  for (const n of graph.nodes) {
    if (!visibleIds.has(n.id)) continue
    const fallback = { x: n.position?.x ?? 0, y: n.position?.y ?? 0 }
    const rp = radialPos?.get(n.id)
    const x = rp?.x ?? fallback.x
    const y = rp?.y ?? fallback.y
    const isMacro = n.kind === 'macro'
    const isLecture = n.kind === 'lecture'
    const isTheme = n.kind === 'theme'
    const isSubtopic = n.kind === 'subtopic'
    const isMicro = n.kind === 'micro'
    const isTopic = (n.kind ?? 'topic') === 'topic'
    const hasChildren = graph.nodes.some((ch) => ch.parentId === n.id)
    const isBranch =
      (isMacro || isLecture || isTheme || isSubtopic) && hasChildren
    /** Отдельные круги-ключевые слова только у topic/micro (не у subtopic — тогда 4-й уровень). */
    const expandKeywordsAsChildNodes = isTopic || isMicro
    const kws = (n.keywords ?? []).filter(Boolean)
    const branchExp = expandedBranches.has(n.id)
    const topicDetailExp = expandedTopics.has(n.id)

    const showsTimeSpan = isLecture || isTheme || isSubtopic
    nodes.push({
      id: n.id,
      type: 'topic',
      position: { x, y },
      data: {
        label: n.label,
        summary: (n.summary ?? '').trim(),
        tooltipText: buildTooltipText(n),
        communityBody: isSubtopic ? (n.communityBody ?? '').trim() || undefined : undefined,
        communityTimeStart:
          showsTimeSpan && typeof n.communityTimeStart === 'number' ? n.communityTimeStart : undefined,
        communityTimeEnd:
          showsTimeSpan && typeof n.communityTimeEnd === 'number' ? n.communityTimeEnd : undefined,
        community: n.community ?? 0,
        keywords: kws,
        kind: (n.kind ?? 'topic') as TopicNodeData['kind'],
        expandable: !!(isBranch || (expandKeywordsAsChildNodes && kws.length > 0)),
        expanded: isBranch
          ? branchExp
          : expandKeywordsAsChildNodes && kws.length > 0
            ? topicDetailExp
            : undefined,
        onToggleExpand: isBranch
          ? () => toggleBranch(n.id)
          : expandKeywordsAsChildNodes && kws.length > 0
            ? () => toggleTopicExpand(n.id)
            : undefined,
      },
    })

    if (expandKeywordsAsChildNodes && kws.length > 0 && topicDetailExp) {
      kws.forEach((kw, idx) => {
        const kid = `${n.id}${KW_NODE_SEP}${idx}`
        const kwTip = `${kw}\n\nТема: «${n.label}».${(n.summary ?? '').trim() ? `\n\n${(n.summary ?? '').trim()}` : ''}`
        nodes.push({
          id: kid,
          type: 'topic',
          position: { x: x + 270, y: y + idx * 48 },
          data: {
            label: kw,
            summary: '',
            tooltipText: kwTip.trim(),
            community: n.community ?? 0,
            kind: 'keyword',
            expandable: false,
          },
        })
        extraEdges.push({
          id: `sub-${kid}`,
          source: n.id,
          target: kid,
          markerEnd: { type: MarkerType.ArrowClosed, color: '#6ee7b7', width: 14, height: 14 },
          style: { stroke: '#34d399', strokeWidth: 1.25 },
        })
      })
    }
  }

  const edgeStyleRelated = {
    stroke: '#a78bfa',
    strokeWidth: 1.25,
    strokeDasharray: '5 6',
  }
  const edgeStyleTimeline = {
    stroke: '#64748b',
    strokeWidth: 1.35,
    strokeDasharray: '4 6',
    opacity: 0.92,
  }
  const edgeStyleDefault = {
    stroke: '#52525b',
    strokeWidth: 1.45,
  }
  const visibleForServerEdge = new Set(visibleIds)
  /* keyword-ребра учитывают только видимые серверные id; дочерние kw-* не в visibleIds */
  const edges: Edge[] = rawLinks
    .filter((e) => {
      const src = String(e.source)
      const tgt = String(e.target)
      if (!visibleForServerEdge.has(src) || !visibleForServerEdge.has(tgt)) return false
      return true
    })
    .map((e, i) => {
      const t = String(e.type || '')
      const isRelated = t === 'related'
      const isTimeline = t === 'timeline' || t === 'follows'
      return {
        id: `e-${i}-${e.source}-${e.target}`,
        source: String(e.source),
        target: String(e.target),
        animated: isRelated,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: isRelated ? '#a78bfa' : isTimeline ? '#94a3b8' : '#71717a',
          width: 18,
          height: 18,
        },
        style: isRelated ? edgeStyleRelated : isTimeline ? edgeStyleTimeline : edgeStyleDefault,
        label: isRelated ? 'семантика' : undefined,
        labelStyle: {
          fill: isRelated ? '#c4b5fd' : '#a1a1aa',
          fontSize: 10,
        },
        labelBgPadding: [4, 2] as [number, number],
        labelBgBorderRadius: 4,
        labelBgStyle: { fill: '#18181b', fillOpacity: 0.95 },
      }
    })

  return { nodes, edges: [...edges, ...extraEdges] }
}

export function TopicMindMap({
  graph,
  onExitViewportFullscreen,
}: {
  graph: TopicGraphPayload
  /** При выходе из режима «на весь экран» — например, переключение на вкладку «Текст». */
  onExitViewportFullscreen?: () => void
}) {
  const branchContainerIds = useMemo(() => {
    const parentRefs = new Set(
      graph.nodes.map((n) => n.parentId).filter((x): x is string => !!x),
    )
    const ids = graph.nodes
      .filter((n) => {
        const k = n.kind
        const isBranch =
          k === 'lecture' || k === 'theme' || k === 'subtopic' || k === 'macro'
        return isBranch && parentRefs.has(n.id)
      })
      .map((n) => n.id)
    return [...new Set(ids)].sort()
  }, [graph.nodes])

  const graphStructuralKey = useMemo(
    () => graph.nodes.map((n) => `${n.id}:${(n.keywords ?? []).length}`).join('|'),
    [graph.nodes],
  )

  const [expandedBranches, setExpandedBranches] = useState<Set<string>>(() => new Set())
  const [expandedTopics, setExpandedTopics] = useState<Set<string>>(() => new Set())
  const [linearLectureLayout, setLinearLectureLayout] = useState(false)

  const branchKey = useMemo(() => branchContainerIds.join(','), [branchContainerIds])
  const prevBranchKeyRef = useRef<string | null>(null)

  useEffect(() => {
    const isNewGraph = prevBranchKeyRef.current !== branchKey
    prevBranchKeyRef.current = branchKey

    if (linearLectureLayout) {
      setExpandedBranches(new Set(branchContainerIds))
      return
    }
    if (isNewGraph) {
      const lecture = graph.nodes.find((n) => n.kind === 'lecture')
      if (lecture && branchContainerIds.includes(lecture.id)) {
        setExpandedBranches(new Set([lecture.id]))
      } else {
        setExpandedBranches(new Set(branchContainerIds))
      }
    }
  }, [branchKey, branchContainerIds, graph.nodes, linearLectureLayout])

  useEffect(() => {
    setExpandedTopics(new Set())
  }, [graphStructuralKey])

  const toggleBranch = useCallback((id: string) => {
    setExpandedBranches((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const toggleTopicExpand = useCallback((id: string) => {
    setExpandedTopics((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const visibleIds = useMemo(
    () => buildVisibleTopicIds(graph, expandedBranches),
    [graph, expandedBranches],
  )

  const { nodes: initNodes, edges: initEdges } = useMemo(
    () =>
      graphToFlow(
        graph,
        visibleIds,
        expandedBranches,
        expandedTopics,
        toggleBranch,
        toggleTopicExpand,
        linearLectureLayout,
      ),
    [
      graph,
      visibleIds,
      expandedBranches,
      expandedTopics,
      toggleBranch,
      toggleTopicExpand,
      linearLectureLayout,
    ],
  )
  const [nodes, setNodes, onNodesChange] = useNodesState(initNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initEdges)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [nodeTooltip, setNodeTooltip] = useState<{
    anchorX: number
    anchorY: number
    title: string
    summaryBody: string
    communityBody: string | null
    communityTimeStart: number | null
    communityTimeEnd: number | null
    kind: TopicNodeData['kind']
  } | null>(null)
  const [tooltipFloat, setTooltipFloat] = useState<{ left: number; top: number }>({ left: 0, top: 0 })
  const tooltipRef = useRef<HTMLDivElement | null>(null)
  const [tooltipTab, setTooltipTab] = useState<'summary' | 'source'>('summary')
  const [viewportFullscreen, setViewportFullscreen] = useState(true)

  useEffect(() => {
    const built = graphToFlow(
      graph,
      visibleIds,
      expandedBranches,
      expandedTopics,
      toggleBranch,
      toggleTopicExpand,
      linearLectureLayout,
    )
    setNodes(built.nodes)
    setEdges(built.edges)
  }, [
    graph,
    visibleIds,
    expandedBranches,
    expandedTopics,
    toggleBranch,
    toggleTopicExpand,
    linearLectureLayout,
    setNodes,
    setEdges,
  ])

  const onNodeClick = useCallback(
    (e: ReactMouseEvent, node: Node<TopicNodeData>) => {
      if (e.detail >= 2) return
      const kd = node.data.kind
      const hasKids = graph.nodes.some((x) => x.parentId === node.id)
      if (
        hasKids &&
        (kd === 'lecture' || kd === 'theme' || kd === 'subtopic' || kd === 'macro')
      ) {
        toggleBranch(node.id)
      } else if (
        (kd === 'topic' || kd === 'micro') &&
        (node.data.keywords?.length ?? 0) > 0
      ) {
        toggleTopicExpand(node.id)
      }
      setSelectedId(node.id)
    },
    [graph.nodes, toggleBranch, toggleTopicExpand],
  )

  useEffect(() => {
    if (nodeTooltip) setTooltipTab('summary')
  }, [nodeTooltip])

  useLayoutEffect(() => {
    if (!nodeTooltip) return
    const el = tooltipRef.current
    if (!el) return
    const pad = 12
    const r = el.getBoundingClientRect()
    let dl = 0
    let dt = 0
    if (r.right > window.innerWidth - pad) dl = window.innerWidth - pad - r.right
    if (r.bottom > window.innerHeight - pad) dt = window.innerHeight - pad - r.bottom
    if (r.left + dl < pad) dl = pad - r.left
    if (r.top + dt < pad) dt = pad - r.top
    if (dl !== 0 || dt !== 0) {
      setTooltipFloat((prev) => ({ left: prev.left + dl, top: prev.top + dt }))
    }
  }, [nodeTooltip, tooltipTab])

  const onNodeDoubleClick = useCallback((_e: ReactMouseEvent, node: Node<TopicNodeData>) => {
    const raw = node.data.communityBody?.trim()
    const ax = _e.clientX
    const ay = _e.clientY
    setTooltipFloat({ left: ax + 12, top: ay + 12 })
    setNodeTooltip({
      anchorX: ax,
      anchorY: ay,
      title: node.data.label,
      summaryBody: node.data.tooltipText,
      communityBody: raw ? raw : null,
      communityTimeStart:
        typeof node.data.communityTimeStart === 'number' ? node.data.communityTimeStart : null,
      communityTimeEnd:
        typeof node.data.communityTimeEnd === 'number' ? node.data.communityTimeEnd : null,
      kind: node.data.kind ?? 'topic',
    })
  }, [])

  const onPaneClick = useCallback(() => {
    setSelectedId(null)
    setNodeTooltip(null)
    setTooltipFloat({ left: 0, top: 0 })
  }, [])

  const leaveViewportFullscreen = useCallback(() => {
    if (onExitViewportFullscreen) {
      onExitViewportFullscreen()
      return
    }
    setViewportFullscreen(false)
  }, [onExitViewportFullscreen])

  useEffect(() => {
    if (!viewportFullscreen) return
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
    }
  }, [viewportFullscreen])

  useEffect(() => {
    if (!viewportFullscreen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') leaveViewportFullscreen()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [viewportFullscreen, leaveViewportFullscreen])

  const selected = useMemo(() => {
    const base = graph.nodes.find((n) => n.id === selectedId) ?? null
    if (base) return base
    if (selectedId?.includes(KW_NODE_SEP)) {
      const [parentId, rest] = selectedId.split(KW_NODE_SEP)
      const idx = Number(rest)
      const parent = graph.nodes.find((n) => n.id === parentId)
      if (parent && Number.isFinite(idx)) {
        const kw = parent.keywords?.[idx]
        if (kw) {
          return {
            ...parent,
            id: selectedId,
            label: kw,
            summary: `Ключевая строка темы «${parent.label}»`,
          }
        }
      }
    }
    return null
  }, [graph.nodes, selectedId])

  if (!graph.nodes.length) {
    return <p className="muted small">Нет узлов графа тем.</p>
  }

  return (
    <div
      className={`mindmap-panel mindmap-panel--fullscreen ${viewportFullscreen ? 'mindmap-panel--viewport-fs' : ''}`}
    >
      <div className="mindmap-toolbar">
        {viewportFullscreen ? (
          <button
            type="button"
            className="btn btn-secondary btn-small mindmap-fs-exit"
            onClick={leaveViewportFullscreen}
          >
            Выйти из полного экрана
          </button>
        ) : !onExitViewportFullscreen ? (
          <button
            type="button"
            className="btn btn-secondary btn-small"
            onClick={() => setViewportFullscreen(true)}
          >
            На весь экран
          </button>
        ) : null}
        <label className="mindmap-toggle">
          <input
            type="checkbox"
            checked={linearLectureLayout}
            onChange={(e) => setLinearLectureLayout(e.target.checked)}
          />
          <span>Линейно по ходу лекции</span>
        </label>
      </div>
      <div className="mindmap-canvas-wrap">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          onNodeClick={onNodeClick}
          onNodeDoubleClick={onNodeDoubleClick}
          onPaneClick={onPaneClick}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          minZoom={0.2}
          maxZoom={1.5}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#3f3f46" />
          <Controls className="mindmap-controls" />
          <MiniMap
            nodeStrokeWidth={2}
            zoomable
            pannable
            maskColor="rgba(15, 15, 18, 0.85)"
            className="mindmap-minimap"
            nodeColor={(n) => {
              const c = (n.data as TopicNodeData)?.community ?? 0
              return COMMUNITY_COLORS[Math.abs(c) % COMMUNITY_COLORS.length]
            }}
          />
        </ReactFlow>
      </div>
      {nodeTooltip ? (
        <div
          ref={tooltipRef}
          role="tooltip"
          className="mindmap-node-tooltip"
          style={{ left: tooltipFloat.left, top: tooltipFloat.top }}
          onMouseLeave={() => {
            setNodeTooltip(null)
            setTooltipFloat({ left: 0, top: 0 })
          }}
        >
          <div className="mindmap-node-tooltip-title">{nodeTooltip.title}</div>
          {(nodeTooltip.communityTimeStart != null || nodeTooltip.communityTimeEnd != null) ? (
            <div className="mindmap-node-tooltip-times">
              {nodeTooltip.kind === 'lecture'
                ? 'Лекция: '
                : nodeTooltip.kind === 'theme'
                  ? 'Тема: '
                  : 'Фрагмент: '}
              {formatMediaSeconds(nodeTooltip.communityTimeStart)} —{' '}
              {formatMediaSeconds(nodeTooltip.communityTimeEnd)}
            </div>
          ) : null}
          <div className="mindmap-node-tooltip-body">
            {tooltipTab === 'summary' ? (
              <LlmMarkdown text={nodeTooltip.summaryBody} className="mindmap-tooltip-md" />
            ) : (
              <div className="mindmap-tooltip-plain">
                {nodeTooltip.communityBody || 'Нет текста сообщества для этого узла.'}
              </div>
            )}
          </div>
          {nodeTooltip.communityBody ? (
            <div className="mindmap-node-tooltip-toggle" role="tablist" aria-label="Режим подсказки">
              <button
                type="button"
                role="tab"
                aria-selected={tooltipTab === 'summary'}
                title="Резюме и ключевые слова"
                className={`mindmap-tooltip-dot ${tooltipTab === 'summary' ? 'mindmap-tooltip-dot--active' : ''}`}
                onClick={(e) => {
                  e.stopPropagation()
                  setTooltipTab('summary')
                }}
              />
              <button
                type="button"
                role="tab"
                aria-selected={tooltipTab === 'source'}
                title="Фактический текст сообщества"
                className={`mindmap-tooltip-dot ${tooltipTab === 'source' ? 'mindmap-tooltip-dot--active' : ''}`}
                onClick={(e) => {
                  e.stopPropagation()
                  setTooltipTab('source')
                }}
              />
            </div>
          ) : null}
        </div>
      ) : null}
      {!viewportFullscreen ? (
        <p className="muted small mindmap-expand-hint">
          {branchContainerIds.length > 0
            ? 'Три уровня: лекция → темы → подтемы. Один клик — свернуть/развернуть ветку. Двойной клик — всплывающее окно; у подтем внизу два кружка: резюме и фактический текст сообщества. Режим «Линейно по ходу лекции» — колонка и timeline.'
            : 'Двойной клик по узлу — карточка с текстом; у подтем два кружка внизу — резюме или исходные реплики. «Линейно по ходу лекции» — шкала и семантические рёбра.'}
        </p>
      ) : null}
      {selected && !viewportFullscreen ? (
        <div className="mindmap-detail">
          <h5 className="mindmap-detail-title">{selected.label}</h5>
          {(() => {
            const sum = (selected.summary ?? '').trim()
            const desc = (selected.description ?? '').trim()
            const kws = (selected.keywords ?? []).map((x) => String(x).trim()).filter(Boolean)
            const main =
              sum ? (
                <div className="mindmap-detail-body mindmap-detail-md">
                  <LlmMarkdown text={sum} />
                </div>
              ) : desc ? (
                <div className="mindmap-detail-body mindmap-detail-md muted">
                  <LlmMarkdown text={desc} />
                </div>
              ) : (
                <p className="muted small">Нет текста резюме для узла.</p>
              )
            if (!kws.length) return main
            return (
              <>
                {main}
                <p className="mindmap-detail-body muted mindmap-detail-keywords">
                  Ключевые слова: {kws.join(', ')}
                </p>
              </>
            )
          })()}
        </div>
      ) : null}
    </div>
  )
}
