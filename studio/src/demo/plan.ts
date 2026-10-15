import type { Ask, Deployment, Pipeline } from '../types'
import basePipeline from './pipeline.json'

/** Demo-only helpers: fake the harness's "where should this run?" question and the resulting plans. */

export const DEMO_ASK: Ask = {
  question: 'Where should this run?',
  options: [
    { label: 'This computer', value: 'local', detail: 'Apple M1 Pro · 16 GB — only models that fit here' },
    { label: 'My phone', value: 'phone', detail: 'on-device models under ~4 GB' },
    { label: 'Cloud', value: 'cloud', detail: 'hosted APIs or a rented GPU — bigger models' },
  ],
}

function clone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x)) as T
}

export function demoPlan(deployment: Deployment): { reply: string; pipeline: Pipeline } {
  const p = clone(basePipeline) as unknown as Pipeline
  if (deployment === 'phone') {
    return {
      reply:
        "Here's a four-step plan: listen, transcribe, think, speak. Everything fits in about 1.9 GB of your iPhone's budget. Click a model to see why I picked it or swap in an alternative.",
      pipeline: p,
    }
  }
  if (deployment === 'local') {
    p.device = { kind: 'mac', label: 'MacBook Pro', ram_cap_gb: 11.2 }
    p.deployment = { mode: 'local', label: 'This Mac · Apple M1 Pro · 16 GB', hardware: { chip: 'Apple M1 Pro', ram_gb: 16, gpu: 'Apple M1 Pro (16-core GPU)', vram_gb: null, unified: true } }
    p.budget = { ...p.budget, ram_cap_gb: 11.2, mode: 'local', note: "All three models loaded at once need about 1.9 GB of this Mac's roughly 11 GB budget." }
    return {
      reply: 'Planned for this computer: all three models run locally and fit with plenty of room (about 1.9 of 11 GB).',
      pipeline: p,
    }
  }
  p.device = { kind: 'any', label: 'Cloud', ram_cap_gb: null }
  p.deployment = { mode: 'cloud', label: 'Cloud · hosted APIs + 1 GPU server', hardware: null }
  p.budget = { ram_cap_gb: null, total_ram_gb: 0.6, fits: true, mode: 'cloud', note: 'The language model uses a hosted API; the speech models need a small rented GPU (about 1 GB of VRAM).' }
  for (const n of p.nodes) {
    if (n.type !== 'model' || !n.model) continue
    n.model.hosting = n.id === 'n3' ? 'hosted_api' : 'gpu_server'
  }
  return {
    reply: 'Planned for the cloud: the language model runs on a hosted API (cheapest provider first) and the speech steps on a small rented GPU.',
    pipeline: p,
  }
}
