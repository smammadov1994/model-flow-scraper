import type { NodeKind } from '../types'

// Small hand-drawn glyphs (24x24, stroke only). No icon library.
const PATHS: Record<string, string> = {
  mic: 'M12 3a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM6 11a6 6 0 0 0 12 0M12 17v4M9 21h6',
  speaker: 'M4 9v6h4l5 4V5L8 9H4zM16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12',
  text: 'M5 6h14M12 6v13M9 19h6',
  image: 'M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4M9 9.5h.01',
  video: 'M3 6h12v12H3zM15 10l6-3v10l-6-3',
  data: 'M12 4c4.4 0 8 1.1 8 2.5S16.4 9 12 9 4 7.9 4 6.5 7.6 4 12 4zM4 6.5v5C4 12.9 7.6 14 12 14s8-1.1 8-2.5v-5M4 11.5v5C4 17.9 7.6 19 12 19s8-1.1 8-2.5v-5',
  logic: 'M6 4v6a4 4 0 0 0 4 4h4a4 4 0 0 1 4 4v2M6 4h.01M18 20h.01M14 14l4-4M18 10v4h-4',
  model: 'M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3M7 6h10a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1zM10 10h4v4h-4z',
  local: 'M4 5h16v10H4zM2 19h20M9 15v4M15 15v4',
  phone: 'M8 3h8a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM11 18h2',
  cloud: 'M7 18a4.5 4.5 0 0 1-.6-8.96A6 6 0 0 1 18 10a4 4 0 0 1 0 8H7z',
  arrow: 'M4 12h14M13 6l6 6-6 6',
}

export function glyphFor(kind: NodeKind, dataTypes: string[]): string {
  const t = (dataTypes[0] ?? '').toLowerCase()
  if (kind === 'logic') return 'logic'
  if (kind === 'model') return 'model'
  if (t.includes('audio') || t.includes('speech') || t.includes('voice')) return kind === 'input' ? 'mic' : 'speaker'
  if (t.includes('image')) return 'image'
  if (t.includes('video')) return 'video'
  if (t.includes('text')) return 'text'
  if (t.includes('json') || t.includes('data') || t.includes('file')) return 'data'
  return 'arrow'
}

export function Glyph({ name, size = 18 }: { name: string; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d={PATHS[name] ?? PATHS.arrow} />
    </svg>
  )
}
