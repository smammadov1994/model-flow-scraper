export type NodeKind = 'input' | 'model' | 'logic' | 'output'
export type NodeStatus = 'ok' | 'no_match' | 'over_budget' | 'pending'

export interface ModelInfo {
  id: string
  url?: string | null
  params_h?: string | null
  ram_gb?: number | null
  variant?: string | null
  variant_repo?: string | null
  format?: string | null
  license?: string | null
  open_license?: boolean | null
  downloads?: number | null
  how_to_run?: string | null
  why?: string | null
  fit?: number | null
}

export interface PipelineNode {
  id: string
  type: NodeKind
  title: string
  step?: string | null
  task?: string | null
  inputs: string[]
  outputs: string[]
  position: { x: number; y: number }
  model?: ModelInfo | null
  alternatives: ModelInfo[]
  status: NodeStatus
  notes?: string | null
}

export interface PipelineEdge {
  id: string
  source: string
  target: string
  label?: string | null
}

export interface Pipeline {
  id: string
  title: string
  goal?: string
  device: { kind: string; label: string; ram_cap_gb: number }
  nodes: PipelineNode[]
  edges: PipelineEdge[]
  budget: { ram_cap_gb: number; total_ram_gb: number; fits: boolean; note?: string | null }
  updated_at?: string
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'error'
  text: string
}

export interface ChatResponse {
  session_id: string
  reply: string
  pipeline: Pipeline
  timing_ms?: Record<string, number>
  planner?: string
}

export interface SessionResponse {
  session_id: string
  messages: { role: string; text: string }[]
  pipeline: Pipeline | null
}
