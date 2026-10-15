import type { Pipeline } from '../types'
import { n8nExportUrl, pipelineExportUrl } from '../api'

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
  if (!b.fits || b.total_ram_gb > b.ram_cap_gb) return 'bad'
  return b.total_ram_gb / b.ram_cap_gb >= 0.8 ? 'warn' : 'ok'
}

export default function TopBar({ pipeline, sessionId, demo, onDemoDownload }: Props) {
  const canExport = !!pipeline && (!!sessionId || demo)
  return (
    <header className="topbar">
      <div className="topbar-main">
        <h1 className="topbar-title">{pipeline ? pipeline.title : 'Your pipeline'}</h1>
        {pipeline ? (
          <span className="chip chip-device" title={`${pipeline.device.label} (${pipeline.device.kind})`}>
            {pipeline.device.label} · {fmt(pipeline.device.ram_cap_gb)} GB budget
          </span>
        ) : null}
      </div>

      {pipeline ? (
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
