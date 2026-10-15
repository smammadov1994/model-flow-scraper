import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  useEdgesState,
  useNodesState,
  type Edge,
  type NodeMouseHandler,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { Pipeline, PipelineNode } from '../types'
import { nodeTypes, type StudioNode } from './nodes'

interface Props {
  pipeline: Pipeline
  selectedId: string | null
  onSelect: (id: string | null) => void
  onMove: (positions: Record<string, { x: number; y: number }>) => void
}

function toFlowNode(n: PipelineNode, i: number, selected: boolean): StudioNode {
  const pos = n.position ?? { x: i * 320, y: 120 }
  return {
    id: n.id,
    type: n.type,
    position: { x: pos.x, y: pos.y },
    data: { p: n },
    selected,
  }
}

function signature(p: Pipeline): string {
  return `${p.id}|${p.nodes.map((n) => n.id).join(',')}|${p.edges.map((e) => e.id).join(',')}`
}

/** Remounts when the set of nodes/edges changes, so the new graph is fitted into view. */
export default function Canvas(props: Props) {
  return <CanvasInner key={signature(props.pipeline)} {...props} />
}

function CanvasInner({ pipeline, selectedId, onSelect, onMove }: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState<StudioNode>(
    pipeline.nodes.map((n, i) => toFlowNode(n, i, n.id === selectedId)),
  )
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null)
  const [, , onEdgesChange] = useEdgesState<Edge>([])
  const dirty = useRef<Record<string, { x: number; y: number }>>({})
  const timer = useRef<number | undefined>(undefined)
  const onMoveRef = useRef(onMove)
  onMoveRef.current = onMove

  // New node data (e.g. after swapping a model) while keeping where the user dragged things.
  useEffect(() => {
    setNodes((prev) =>
      pipeline.nodes.map((n, i) => {
        const old = prev.find((o) => o.id === n.id)
        const fresh = toFlowNode(n, i, n.id === selectedId)
        return old ? { ...fresh, position: old.position } : fresh
      }),
    )
  }, [pipeline, selectedId, setNodes])

  const edges = useMemo<Edge[]>(
    () =>
      pipeline.edges.map((e) => {
        const active = e.id === selectedEdge || e.source === selectedId || e.target === selectedId
        return {
          id: e.id,
          source: e.source,
          target: e.target,
          type: 'smoothstep',
          label: e.label ?? undefined,
          animated: active,
          selected: e.id === selectedEdge,
          className: active ? 'edge-active' : undefined,
          markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
        }
      }),
    [pipeline.edges, selectedId, selectedEdge],
  )

  const flush = useCallback(() => {
    const batch = dirty.current
    dirty.current = {}
    if (Object.keys(batch).length) onMoveRef.current(batch)
  }, [])

  useEffect(
    () => () => {
      window.clearTimeout(timer.current)
      flush()
    },
    [flush],
  )

  const onNodeDragStop = useCallback(
    (_: unknown, __: unknown, dragged: StudioNode[]) => {
      for (const n of dragged) dirty.current[n.id] = { x: Math.round(n.position.x), y: Math.round(n.position.y) }
      window.clearTimeout(timer.current)
      timer.current = window.setTimeout(flush, 500)
    },
    [flush],
  )

  const onNodeClick = useCallback<NodeMouseHandler<StudioNode>>(
    (_, node) => {
      setSelectedEdge(null)
      onSelect(node.id)
    },
    [onSelect],
  )

  return (
    <ReactFlow
      className="flow"
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={onNodeClick}
      onEdgeClick={(_, e) => {
        setSelectedEdge(e.id)
      }}
      onPaneClick={() => {
        setSelectedEdge(null)
        onSelect(null)
      }}
      onNodeDragStop={(e, n, all) => onNodeDragStop(e, n, all as StudioNode[])}
      nodesConnectable={false}
      deleteKeyCode={null}
      colorMode="system"
      fitView
      fitViewOptions={{ padding: 0.2, maxZoom: 1.1 }}
      minZoom={0.2}
      proOptions={{ hideAttribution: true }}
      aria-label="Pipeline canvas"
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1.4} />
      <Controls showInteractive={false} />
      <MiniMap pannable zoomable ariaLabel="Pipeline overview" nodeStrokeWidth={2} />
    </ReactFlow>
  )
}
