export type NodeKind = 'input' | 'model' | 'logic' | 'output'
export type NodeStatus = 'ok' | 'no_match' | 'over_budget' | 'pending'

export type Deployment = 'local' | 'phone' | 'cloud'
export type Hosting = 'local' | 'hosted_api' | 'gpu_server'

export interface Provider {
  name: string
  price_in?: number | null
  price_out?: number | null
  tokens_per_s?: number | null
  latency_ms?: number | null
}

export interface HardwareInfo {
  chip?: string | null
  ram_gb?: number | null
  gpu?: string | null
  vram_gb?: number | null
  unified?: boolean | null
}

export interface PipelineDeployment {
  mode: Deployment | 'other'
  label?: string | null
  hardware?: HardwareInfo | null
}

export interface AskOption {
  label: string
  value: Deployment
  detail?: string | null
}

export interface Ask {
  question: string
  options: AskOption[]
}

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
  hosting?: Hosting | null
  providers?: Provider[] | null
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
  device: { kind: string; label: string; ram_cap_gb: number | null }
  deployment?: PipelineDeployment | null
  nodes: PipelineNode[]
  edges: PipelineEdge[]
  budget: {
    ram_cap_gb: number | null
    total_ram_gb: number
    fits: boolean
    note?: string | null
    mode?: string | null
  }
  updated_at?: string
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'error'
  text: string
  ask?: Ask | null
}

export interface ChatResponse {
  session_id: string
  reply?: string | null
  pipeline: Pipeline | null
  ask?: Ask | null
  timing_ms?: Record<string, number>
  planner?: string
}

export interface SessionResponse {
  session_id: string
  messages: { role: string; text: string }[]
  pipeline: Pipeline | null
}

export interface HardwareResponse {
  label: string
  kind?: string
  chip?: string | null
  ram_gb?: number | null
  gpu?: string | null
  vram_gb?: number | null
  unified?: boolean | null
  budget_gb?: number | null
}
