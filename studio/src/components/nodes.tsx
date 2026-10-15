import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import type { NodeStatus, PipelineNode } from '../types'
import { taskLabel } from '../taskLabels'
import { Glyph, glyphFor } from './Glyph'

export type StudioNodeData = { p: PipelineNode } & Record<string, unknown>
export type StudioNode = Node<StudioNodeData>

export const STATUS_LABEL: Record<NodeStatus, string> = {
  ok: 'Fits',
  over_budget: 'Over budget',
  no_match: 'No match',
  pending: 'Pending',
}

export function StatusBadge({ status }: { status: NodeStatus }) {
  return <span className={`badge badge-${status}`}>{STATUS_LABEL[status] ?? status}</span>
}

export function fitPercent(fit: number | null | undefined): string | null {
  if (typeof fit !== 'number' || Number.isNaN(fit)) return null
  return `${Math.round(fit * 100)}%`
}

export function ModelNodeCard({ data, selected }: NodeProps<StudioNode>) {
  const p = data.p
  const m = p.model
  const fit = fitPercent(m?.fit)
  return (
    <div className={`node node-model status-${p.status}${selected ? ' is-selected' : ''}`}>
      <Handle type="target" position={Position.Left} />
      <div className="node-head">
        <span className="node-glyph">
          <Glyph name="model" size={15} />
        </span>
        <span className="node-task">{taskLabel(p.task)}</span>
        <StatusBadge status={p.status} />
      </div>
      <div className="node-body">
        <div className="node-title">{p.title}</div>
        {m ? (
          <>
            <div className="node-model-id" title={m.id}>
              {m.id}
            </div>
            <div className="chips">
              {m.params_h ? <span className="chip">{m.params_h}</span> : null}
              {typeof m.ram_gb === 'number' ? <span className="chip">~{m.ram_gb} GB RAM</span> : null}
              {m.format ? <span className="chip">{m.format}</span> : null}
              {fit ? <span className="chip chip-fit">fit {fit}</span> : null}
            </div>
          </>
        ) : (
          <div className="node-empty">{p.notes || 'No model picked for this step yet.'}</div>
        )}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

export function SimpleNodeCard({ data, selected }: NodeProps<StudioNode>) {
  const p = data.p
  const hasIn = p.type !== 'input'
  const hasOut = p.type !== 'output'
  const types = p.type === 'input' ? p.outputs : p.type === 'output' ? p.inputs : [...p.inputs, ...p.outputs]
  const kindLabel = p.type === 'input' ? 'Input' : p.type === 'output' ? 'Output' : 'Logic'
  return (
    <div className={`node node-simple node-${p.type}${selected ? ' is-selected' : ''}`}>
      {hasIn ? <Handle type="target" position={Position.Left} /> : null}
      <span className="simple-glyph">
        <Glyph name={glyphFor(p.type, types)} size={20} />
      </span>
      <div className="simple-text">
        <div className="simple-kind">{kindLabel}</div>
        <div className="node-title">{p.title}</div>
        {p.step ? <div className="simple-step">{p.step}</div> : null}
      </div>
      {hasOut ? <Handle type="source" position={Position.Right} /> : null}
    </div>
  )
}

export const nodeTypes = {
  model: ModelNodeCard,
  input: SimpleNodeCard,
  output: SimpleNodeCard,
  logic: SimpleNodeCard,
}
