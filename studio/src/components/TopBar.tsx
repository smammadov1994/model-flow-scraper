import type { Pipeline } from '../types'
import { n8nExportUrl, pipelineExportUrl } from '../api'
import { Glyph } from './Glyph'

interface Props {
  pipeline: Pipeline | null
  sessionId: string | null
  demo: boolean
  onDemoDownload: (kind: 'n8n' | 'pipeline') => void
}

function fmt(n: number): string {
  return (Math.round(n * 10) / 10).toString()
}

export function budgetTone(b: Pipeline['budget']): 'ok' | 'warn' | 'bad' {
  const cap = b.ram_cap_gb
  if (typeof cap !== 'number' || cap <= 0) return 'ok'
  if (!b.fits || b.total_ram_gb > cap) return 'bad'
  return b.total_ram_gb / cap >= 0.8 ? 'warn' : 'ok'
}

function deviceChip(p: Pipeline) {
  const mode = p.deployment?.mode
  const glyph = mode === 'local' || mode === 'phone' || mode === 'cloud' ? mode : null
  const label = p.deployment?.label?.trim()
  if (label) return { glyph, text: label, title: `${label}${p.device?.label ? ` (${p.device.label})` : ''}` }
  const cap = typeof p.device.ram_cap_gb === 'number' ? ` · ${fmt(p.device.ram_cap_gb)} GB budget` : ''
  return { glyph, text: `${p.device.label}${cap}`, title: `${p.device.label} (${p.device.kind})` }
}

export default function TopBar({ pipeline, sessionId, demo, onDemoDownload }: Props) {
  const canExport = !!pipeline && (!!sessionId || demo)
  return (
    <header className="topbar">
      <div className="topbar-main">
        <h1 className="topbar-title">{pipeline ? pipeline.title : 'Your pipeline'}</h1>
        {pipeline
          ? (() => {
              const c = deviceChip(pipeline)
              return (
                <span className="chip chip-device" title={c.title}>
                  {c.glyph ? (
                    <span className="dev-glyph">
                      <Glyph name={c.glyph} size={13} />
                    </span>
                  ) : null}
                  {c.text}
                </span>
              )
            })()
          : null}
      </div>

      {pipeline && pipeline.budget.mode === 'cloud' ? (
        <span className="chip chip-cloud" title={pipeline.budget.note ?? undefined} tabIndex={0}>
          <span className="dev-glyph">
            <Glyph name="cloud" size={13} />
          </span>
          Cloud
          {typeof pipeline.budget.total_ram_gb === 'number' && pipeline.budget.total_ram_gb > 0
            ? ` · GPU ~${fmt(pipeline.budget.total_ram_gb)} GB`
            : ''}
        </span>
      ) : pipeline && typeof pipeline.budget.ram_cap_gb === 'number' && pipeline.budget.ram_cap_gb > 0 ? (
        <div
          className={`budget budget-${budgetTone(pipeline.budget)}`}
          title={pipeline.budget.note ?? undefined}
        >
          <div className="budget-text">
            Memory <strong>{fmt(pipeline.budget.total_ram_gb)}</strong> / {fmt(pipeline.budget.ram_cap_gb)} GB
            {!pipeline.budget.fits ? <span className="budget-flag"> · over budget</span> : null}
          </div>
          <div
            className="budget-track"
            role="progressbar"
            aria-label="Memory use compared with device budget"
            aria-valuemin={0}
            aria-valuemax={pipeline.budget.ram_cap_gb}
            aria-valuenow={Math.min(pipeline.budget.total_ram_gb, pipeline.budget.ram_cap_gb)}
            aria-valuetext={`${fmt(pipeline.budget.total_ram_gb)} of ${fmt(pipeline.budget.ram_cap_gb)} GB`}
          >
            <div
              className="budget-fill"
              style={{
                width: `${Math.min(100, Math.max(2, (pipeline.budget.total_ram_gb / pipeline.budget.ram_cap_gb) * 100))}%`,
              }}
            />
          </div>
        </div>
      ) : null}

      <div className="topbar-actions">
        {canExport && sessionId && !demo ? (
          <>
            <a className="btn" href={n8nExportUrl(sessionId)} download>
              Export to n8n
            </a>
            <a className="btn btn-ghost" href={pipelineExportUrl(sessionId)} download>
              Download pipeline JSON
            </a>
          </>
        ) : (
          <>
            <button type="button" className="btn" disabled={!canExport} onClick={() => onDemoDownload('n8n')}>
              Export to n8n
            </button>
            <button type="button" className="btn btn-ghost" disabled={!canExport} onClick={() => onDemoDownload('pipeline')}>
              Download pipeline JSON
            </button>
          </>
        )}
      </div>
    </header>
  )
}
