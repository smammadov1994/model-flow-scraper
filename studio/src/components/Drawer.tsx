import { useEffect, useRef, useState } from 'react'
import type { ModelInfo, PipelineNode, Provider } from '../types'
import { taskLabel } from '../taskLabels'
import { HostingBadge, StatusBadge, fitPercent } from './nodes'

interface Props {
  node: PipelineNode
  swapping: boolean
  onSwap: (nodeId: string, modelId: string) => void
  onClose: () => void
}

function formatDownloads(n: number | null | undefined): string | null {
  if (typeof n !== 'number') return null
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, '')}M`
  if (n >= 1_000) return `${Math.round(n / 1_000)}K`
  return String(n)
}

function money(n: number): string {
  const two = Math.abs(n * 100 - Math.round(n * 100)) < 1e-6
  return `$${n.toFixed(two ? 2 : 3)}`
}

export function providerPrice(p: Provider): string | null {
  const parts: string[] = []
  if (typeof p.price_in === 'number') parts.push(`${money(p.price_in)} in`)
  if (typeof p.price_out === 'number') parts.push(`${money(p.price_out)} out`)
  return parts.length ? `${parts.join(' / ')} per 1M tokens` : null
}

function providerSpeed(p: Provider): string | null {
  const parts: string[] = []
  if (typeof p.tokens_per_s === 'number') parts.push(`${Math.round(p.tokens_per_s)} tokens/s`)
  if (typeof p.latency_ms === 'number') parts.push(`${Math.round(p.latency_ms)} ms latency`)
  return parts.length ? parts.join(' · ') : null
}

function safeUrl(u: string | null | undefined): string | null {
  if (!u) return null
  try {
    const parsed = new URL(u)
    return parsed.protocol === 'https:' || parsed.protocol === 'http:' ? parsed.href : null
  } catch {
    return null
  }
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.setAttribute('readonly', '')
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    let ok = false
    try {
      ok = document.execCommand('copy')
    } catch {
      ok = false
    }
    document.body.removeChild(ta)
    return ok
  }
}

function Alt({ m, disabled, onUse }: { m: ModelInfo; disabled: boolean; onUse: () => void }) {
  const fit = fitPercent(m.fit)
  return (
    <li className="alt">
      <div className="alt-main">
        <div className="alt-id" title={m.id}>
          {m.id}
        </div>
        <div className="chips">
          {m.params_h ? <span className="chip">{m.params_h}</span> : null}
          {typeof m.ram_gb === 'number' ? <span className="chip">~{m.ram_gb} GB RAM</span> : null}
          {m.format ? <span className="chip">{m.format}</span> : null}
          {fit ? <span className="chip chip-fit">fit {fit}</span> : null}
        </div>
      </div>
      <button type="button" className="btn btn-small" disabled={disabled} onClick={onUse} aria-label={`Use ${m.id}`}>
        Use this
      </button>
    </li>
  )
}

export default function Drawer({ node, swapping, onSwap, onClose }: Props) {
  const m = node.model
  const [copied, setCopied] = useState(false)
  const closeRef = useRef<HTMLButtonElement>(null)
  const url = safeUrl(m?.url)
  const downloads = formatDownloads(m?.downloads)
  const fit = fitPercent(m?.fit)

  useEffect(() => {
    setCopied(false)
  }, [node.id, m?.id])

  useEffect(() => {
    closeRef.current?.focus()
  }, [node.id])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const copy = async () => {
    if (!m?.how_to_run) return
    if (await copyText(m.how_to_run)) {
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    }
  }

  return (
    <aside className="drawer" role="dialog" aria-label={`Details for ${node.title}`}>
      <header className="drawer-head">
        <div>
          <div className="drawer-task">{taskLabel(node.task)}</div>
          <h2 className="drawer-title">{node.title}</h2>
        </div>
        <button ref={closeRef} type="button" className="icon-btn" onClick={onClose} aria-label="Close details">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <path d="M6 6l12 12M18 6L6 18" />
          </svg>
        </button>
      </header>

      <div className="drawer-scroll">
        {node.step ? <p className="drawer-step">{node.step}</p> : null}
        <div className="drawer-status">
          <StatusBadge status={node.status} />
        </div>
        {node.notes ? <p className="drawer-note">{node.notes}</p> : null}

        {m ? (
          <>
            <section className="drawer-section">
              <h3>Chosen model</h3>
              {url ? (
                <a className="model-link" href={url} target="_blank" rel="noopener noreferrer">
                  {m.id}
                  <span aria-hidden="true"> ↗</span>
                  <span className="sr-only"> (opens in a new tab)</span>
                </a>
              ) : (
                <div className="model-link plain">{m.id}</div>
              )}
              <div className="chips">
                {m.params_h ? <span className="chip">{m.params_h}</span> : null}
                {typeof m.ram_gb === 'number' ? <span className="chip">~{m.ram_gb} GB RAM</span> : null}
                {m.format ? <span className="chip">{m.format}</span> : null}
                {m.variant ? <span className="chip">{m.variant}</span> : null}
                {fit ? <span className="chip chip-fit">fit {fit}</span> : null}
                <HostingBadge hosting={m.hosting} />
              </div>
            </section>

            {m.providers && m.providers.length > 0 ? (
              <section className="drawer-section">
                <h3>Hosted APIs</h3>
                <ul className="providers">
                  {m.providers.map((pr) => {
                    const price = providerPrice(pr)
                    const speed = providerSpeed(pr)
                    return (
                      <li key={pr.name} className="provider">
                        <div className="provider-name">{pr.name}</div>
                        {price ? <div className="provider-line">{price}</div> : null}
                        {speed ? <div className="provider-line">{speed}</div> : null}
                      </li>
                    )
                  })}
                </ul>
              </section>
            ) : null}

            {m.why ? (
              <section className="drawer-section">
                <h3>Why this one</h3>
                <p>{m.why}</p>
              </section>
            ) : null}

            {m.how_to_run ? (
              <section className="drawer-section">
                <h3>How to run it</h3>
                <div className="callout">
                  <pre>{m.how_to_run}</pre>
                  <button type="button" className="btn btn-small" onClick={copy}>
                    {copied ? 'Copied' : 'Copy'}
                  </button>
                </div>
                <span className="sr-only" role="status" aria-live="polite">
                  {copied ? 'Copied to clipboard' : ''}
                </span>
              </section>
            ) : null}

            <section className="drawer-section">
              <h3>License and popularity</h3>
              <dl className="facts">
                <dt>License</dt>
                <dd>
                  {m.license || 'unknown'}
                  {m.open_license === true ? <span className="tag tag-open">open</span> : null}
                  {m.open_license === false ? <span className="tag tag-closed">restricted</span> : null}
                </dd>
                {downloads ? (
                  <>
                    <dt>Downloads</dt>
                    <dd>{downloads}</dd>
                  </>
                ) : null}
                {m.variant_repo ? (
                  <>
                    <dt>Files from</dt>
                    <dd className="mono">{m.variant_repo}</dd>
                  </>
                ) : null}
              </dl>
            </section>
          </>
        ) : (
          <p className="drawer-note">No model has been picked for this step yet.</p>
        )}

        <section className="drawer-section">
          <h3>Alternatives</h3>
          {node.alternatives.length ? (
            <ul className="alts">
              {node.alternatives.map((a) => (
                <Alt key={a.id} m={a} disabled={swapping} onUse={() => onSwap(node.id, a.id)} />
              ))}
            </ul>
          ) : (
            <p className="muted">No other candidates were found for this step.</p>
          )}
          {swapping ? <p className="muted">Swapping and re-checking memory…</p> : null}
        </section>
      </div>
    </aside>
  )
}
