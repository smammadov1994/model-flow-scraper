import { useCallback, useEffect, useRef, useState } from 'react'
import { ReactFlowProvider } from '@xyflow/react'
import * as api from './api'
import demoPipeline from './demo/pipeline.json'
import { DEMO_ASK, demoPlan } from './demo/plan'
import type { ChatMessage, Deployment, Pipeline } from './types'
import ChatPanel from './components/ChatPanel'
import Canvas from './components/Canvas'
import Drawer from './components/Drawer'
import TopBar from './components/TopBar'

const params = new URLSearchParams(window.location.search)
const DEMO = params.get('demo') === '1'
const DEMO_PRELOADED = DEMO && params.get('pipeline') === '1'
const DEMO_SESSION = 'demo'
const TARGET_KEY = 'studio.target'

function loadTarget(): Deployment | null {
  try {
    const v = window.localStorage.getItem(TARGET_KEY)
    return v === 'local' || v === 'phone' || v === 'cloud' ? v : null
  } catch {
    return null
  }
}

function saveTarget(t: Deployment | null) {
  try {
    if (t) window.localStorage.setItem(TARGET_KEY, t)
    else window.localStorage.removeItem(TARGET_KEY)
  } catch {
    /* storage unavailable; the choice just won't persist */
  }
}

function setUrlSession(id: string | null) {
  const u = new URL(window.location.href)
  if (id) u.searchParams.set('s', id)
  else u.searchParams.delete('s')
  window.history.replaceState(null, '', u.toString())
}

function errText(e: unknown): string {
  return e instanceof Error && e.message ? e.message : 'Something went wrong. Please try again.'
}

function clone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x)) as T
}

function download(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

/** Demo-only: swap a model locally and recompute the budget like the backend would. */
function demoSwap(p: Pipeline, nodeId: string, modelId: string): Pipeline {
  const next = clone(p)
  if (next.budget.mode === 'cloud' || next.budget.ram_cap_gb == null) return next
  const node = next.nodes.find((n) => n.id === nodeId)
  const alt = node?.alternatives.find((a) => a.id === modelId)
  if (!node || !alt) return next
  const old = node.model
  node.alternatives = [...(old ? [old] : []), ...node.alternatives.filter((a) => a.id !== modelId)]
  node.model = alt
  const total = next.nodes.reduce((s, n) => s + (n.model?.ram_gb ?? 0), 0)
  const cap = next.budget.ram_cap_gb
  next.budget.total_ram_gb = Math.round(total * 100) / 100
  next.budget.fits = total <= cap
  next.budget.note = `Models loaded at once need about ${next.budget.total_ram_gb} GB of the ${next.device.label}'s roughly ${cap} GB budget.`
  for (const n of next.nodes) if (n.type === 'model' && n.model) n.status = next.budget.fits ? 'ok' : 'over_budget'
  return next
}

function demoN8n(p: Pipeline) {
  return {
    name: p.title,
    nodes: p.nodes.map((n) => ({
      id: n.id,
      name: n.title,
      type: 'n8n-nodes-base.noOp',
      typeVersion: 1,
      position: [n.position.x, n.position.y],
      parameters: {},
      notes: n.model?.id ?? n.step ?? '',
    })),
    connections: Object.fromEntries(
      p.nodes.map((n) => {
        const t = p.edges.filter((e) => e.source === n.id).map((e) => ({ node: p.nodes.find((x) => x.id === e.target)?.title ?? e.target, type: 'main', index: 0 }))
        return [n.title, { main: [t] }]
      }),
    ),
    settings: {},
  }
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(DEMO_PRELOADED ? DEMO_SESSION : null)
  const [target, setTargetState] = useState<Deployment | null>(loadTarget)
  const [hardwareLabel, setHardwareLabel] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    DEMO_PRELOADED
      ? [
          { role: 'user', text: (demoPipeline as unknown as Pipeline).goal ?? 'A voice assistant that runs on my iPhone' },
          {
            role: 'assistant',
            text: "Here's a four-step plan: listen, transcribe, think, speak. Everything fits in about 1.9 GB of your iPhone's budget. Click a model to see why I picked it or swap in an alternative.",
          },
        ]
      : [],
  )
  const [pipeline, setPipeline] = useState<Pipeline | null>(() => (DEMO_PRELOADED ? (demoPipeline as unknown as Pipeline) : null))
  const [pending, setPending] = useState(false)
  const [loadingSession, setLoadingSession] = useState(!DEMO && !!params.get('s'))
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [swapping, setSwapping] = useState(false)
  const sessionRef = useRef(sessionId)
  sessionRef.current = sessionId

  const setTarget = useCallback((t: Deployment | null) => {
    setTargetState(t)
    saveTarget(t)
  }, [])

  // This computer's hardware, shown next to the "This computer" option (omitted if unavailable)
  useEffect(() => {
    if (DEMO) {
      setHardwareLabel('Apple M1 Pro · 16 GB')
      return
    }
    let cancelled = false
    api
      .getHardware()
      .then((h) => {
        if (!cancelled && typeof h?.label === 'string' && h.label) setHardwareLabel(h.label)
      })
      .catch(() => {
        /* optional nicety */
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Restore a session from ?s=<id>
  useEffect(() => {
    const id = params.get('s')
    if (DEMO || !id) return
    let cancelled = false
    api
      .getSession(id)
      .then((s) => {
        if (cancelled) return
        setSessionId(s.session_id ?? id)
        setMessages(
          (s.messages ?? []).map((m) => ({ role: m.role === 'user' ? 'user' : 'assistant', text: m.text })),
        )
        setPipeline(s.pipeline)
      })
      .catch((e) => {
        if (cancelled) return
        setUrlSession(null)
        setMessages([{ role: 'error', text: `Couldn't restore that session. ${errText(e)}` }])
      })
      .finally(() => {
        if (!cancelled) setLoadingSession(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const targetRef = useRef(target)
  targetRef.current = target

  const send = useCallback(async (text: string, deployment?: Deployment) => {
    const dep = deployment ?? targetRef.current
    setMessages((m) => [...m, { role: 'user', text }])
    setPending(true)
    try {
      if (DEMO) {
        await new Promise((r) => window.setTimeout(r, 900))
        if (!dep) {
          setMessages((m) => [...m, { role: 'assistant', text: DEMO_ASK.question, ask: DEMO_ASK }])
          return
        }
        const plan = demoPlan(dep)
        setSessionId(DEMO_SESSION)
        setMessages((m) => [...m, { role: 'assistant', text: plan.reply }])
        setPipeline(plan.pipeline)
        setSelectedId(null)
        return
      }
      const res = await api.chat(sessionRef.current, text, dep ?? null)
      setSessionId(res.session_id)
      setUrlSession(res.session_id)
      const reply = res.reply?.trim() ?? ''
      if (res.ask) {
        setMessages((m) => [...m, { role: 'assistant', text: reply || res.ask!.question, ask: res.ask }])
      } else {
        setMessages((m) => [...m, { role: 'assistant', text: reply }])
      }
      if (res.pipeline) {
        setPipeline(res.pipeline)
        setSelectedId(null)
      }
    } catch (e) {
      setMessages((m) => [...m, { role: 'error', text: errText(e) }])
    } finally {
      setPending(false)
    }
  }, [])

  const newPipeline = useCallback(() => {
    setSessionId(null)
    setPipeline(null)
    setMessages([])
    setSelectedId(null)
    setUrlSession(null)
  }, [])

  const doSwap = useCallback(async (nodeId: string, modelId: string) => {
    setSwapping(true)
    try {
      if (DEMO) {
        setPipeline((p) => (p ? demoSwap(p, nodeId, modelId) : p))
      } else if (sessionRef.current) {
        const res = await api.swap(sessionRef.current, nodeId, modelId)
        setPipeline(res.pipeline)
      }
    } catch (e) {
      setMessages((m) => [...m, { role: 'error', text: `Couldn't swap that model. ${errText(e)}` }])
    } finally {
      setSwapping(false)
    }
  }, [])

  const onMove = useCallback((positions: Record<string, { x: number; y: number }>) => {
    // keep local copy in sync so a later refresh of the same pipeline doesn't jump nodes back
    setPipeline((p) =>
      p
        ? { ...p, nodes: p.nodes.map((n) => (positions[n.id] ? { ...n, position: positions[n.id] } : n)) }
        : p,
    )
    if (DEMO || !sessionRef.current) return
    api.move(sessionRef.current, positions).catch(() => {
      /* layout is a nicety; ignore failures */
    })
  }, [])

  const onDemoDownload = useCallback(
    (kind: 'n8n' | 'pipeline') => {
      if (!pipeline) return
      if (kind === 'n8n') download(`${pipeline.id}-n8n.json`, demoN8n(pipeline))
      else download(`${pipeline.id}.json`, pipeline)
    },
    [pipeline],
  )

  const selectedNode = pipeline?.nodes.find((n) => n.id === selectedId) ?? null
  const drawerNode = selectedNode && selectedNode.type === 'model' ? selectedNode : null
  const closeDrawer = useCallback(() => setSelectedId(null), [])

  return (
    <div className="app">
      <ChatPanel
        messages={messages}
        pending={pending}
        onSend={send}
        onNew={newPipeline}
        canReset={!!pipeline || messages.length > 0}
        target={target}
        onTarget={setTarget}
        hardwareLabel={hardwareLabel}
      />
      <main className="stage">
        <TopBar pipeline={pipeline} sessionId={sessionId} demo={DEMO} onDemoDownload={onDemoDownload} />
        <div className="canvas-wrap">
          {pipeline && pipeline.nodes.length > 0 ? (
            <ReactFlowProvider>
              <Canvas pipeline={pipeline} selectedId={selectedId} onSelect={setSelectedId} onMove={onMove} />
            </ReactFlowProvider>
          ) : (
            <div className="canvas-empty">
              <div className="empty-art" aria-hidden="true">
                <span />
                <i />
                <span />
                <i />
                <span />
              </div>
              <p className="empty-title">{loadingSession ? 'Loading your pipeline…' : pending ? 'Working on it…' : 'Nothing here yet'}</p>
              <p className="muted">
                {pending
                  ? 'Your pipeline will appear here in a few seconds.'
                  : "Describe what you want to build — I'll pick models and lay them out here."}
              </p>
            </div>
          )}
          {drawerNode ? <Drawer node={drawerNode} swapping={swapping} onSwap={doSwap} onClose={closeDrawer} /> : null}
        </div>
      </main>
    </div>
  )
}
