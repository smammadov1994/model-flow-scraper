import type { ChatResponse, Deployment, HardwareResponse, Pipeline, SessionResponse } from './types'

const UNREACHABLE = "Can't reach the Model Studio server. Is it running?"

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, init)
  } catch {
    throw new Error(UNREACHABLE)
  }
  let data: unknown = null
  try {
    data = await res.json()
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    const msg =
      data && typeof data === 'object' && 'error' in data && typeof (data as { error: unknown }).error === 'string'
        ? (data as { error: string }).error
        : `Something went wrong (HTTP ${res.status}).`
    throw new Error(msg)
  }
  return data as T
}

const post = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const chat = (sessionId: string | null, message: string, deployment: Deployment | null = null) =>
  request<ChatResponse>(
    '/api/studio/chat',
    post(deployment ? { session_id: sessionId, message, deployment } : { session_id: sessionId, message }),
  )

export const getHardware = () => request<HardwareResponse>('/api/studio/hardware')

export const getSession = (id: string) =>
  request<SessionResponse>(`/api/studio/sessions/${encodeURIComponent(id)}`)

export const swap = (sessionId: string, nodeId: string, modelId: string) =>
  request<{ pipeline: Pipeline }>('/api/studio/swap', post({ session_id: sessionId, node_id: nodeId, model_id: modelId }))

export const move = (sessionId: string, positions: Record<string, { x: number; y: number }>) =>
  request<{ ok: boolean }>('/api/studio/move', post({ session_id: sessionId, positions }))

export const n8nExportUrl = (sessionId: string) => `/api/studio/export/n8n/${encodeURIComponent(sessionId)}`
export const pipelineExportUrl = (sessionId: string) => `/api/studio/export/pipeline/${encodeURIComponent(sessionId)}`
