"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import type { GraphData, GraphNode } from "@/lib/graph";
import { getCategoryAccent } from "@/lib/data";

interface SimNode extends SimulationNodeDatum, GraphNode {}
type SimLink = SimulationLinkDatum<SimNode>;

interface ExploreGraphProps {
  graph: GraphData;
  query: string;
  onSelectTool: (node: GraphNode) => void;
}

interface Transform {
  x: number;
  y: number;
  k: number;
}

const CATEGORY_RADIUS = 20;
const TOOL_RADIUS = 7;
const SHARED_TOOL_RADIUS = 9;

function nodeRadius(node: GraphNode): number {
  if (node.type === "category") return CATEGORY_RADIUS;
  return node.categoryIds.length > 1 ? SHARED_TOOL_RADIUS : TOOL_RADIUS;
}

function nodeAccent(node: GraphNode): string {
  return getCategoryAccent(node.categoryIds[0]);
}

export default function ExploreGraph({
  graph,
  query,
  onSelectTool,
}: ExploreGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [, setVersion] = useState(0);
  const [transform, setTransform] = useState<Transform>({ x: 0, y: 0, k: 1 });
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  // Deterministic initial layout: categories on a ring, tools scattered
  // around their category, then settled with a synchronous simulation run.
  // No Math.random, so server and client render identical positions.
  const { nodes, links, simulation } = useMemo(() => {
    const categoryCount = graph.nodes.filter(
      (node) => node.type === "category"
    ).length;
    let categoryIndex = 0;
    const categoryAngle = new Map<string, number>();

    const nodes: SimNode[] = graph.nodes.map((node) => {
      if (node.type === "category") {
        const angle = (categoryIndex++ / categoryCount) * 2 * Math.PI;
        categoryAngle.set(node.categoryIds[0], angle);
        return { ...node, x: Math.cos(angle) * 320, y: Math.sin(angle) * 320 };
      }
      return { ...node };
    });
    let toolIndex = 0;
    for (const node of nodes) {
      if (node.type === "tool") {
        const angle = categoryAngle.get(node.categoryIds[0]) ?? 0;
        const jitter = (toolIndex++ % 7) - 3;
        node.x = Math.cos(angle + jitter * 0.08) * (380 + (toolIndex % 5) * 22);
        node.y = Math.sin(angle + jitter * 0.08) * (380 + (toolIndex % 5) * 22);
      }
    }

    const links: SimLink[] = graph.links.map((link) => ({ ...link }));

    const simulation: Simulation<SimNode, SimLink> = forceSimulation(nodes)
      .force(
        "link",
        forceLink<SimNode, SimLink>(links)
          .id((node) => node.id)
          .distance((link) =>
            (link.target as SimNode).categoryIds.length > 1 ? 140 : 75
          )
          .strength(0.6)
      )
      .force(
        "charge",
        forceManyBody<SimNode>().strength((node) =>
          node.type === "category" ? -900 : -80
        )
      )
      .force(
        "collide",
        forceCollide<SimNode>().radius((node) => nodeRadius(node) + 16)
      )
      .force("x", forceX(0).strength(0.03))
      .force("y", forceY(0).strength(0.03))
      .stop();

    simulation.tick(300);
    return { nodes, links, simulation };
  }, [graph]);

  const adjacency = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const link of graph.links) {
      if (!map.has(link.source)) map.set(link.source, new Set());
      if (!map.has(link.target)) map.set(link.target, new Set());
      map.get(link.source)!.add(link.target);
      map.get(link.target)!.add(link.source);
    }
    return map;
  }, [graph]);

  // Re-render on simulation ticks (only runs while dragging a node).
  useEffect(() => {
    simulation.on("tick", () => setVersion((v) => v + 1));
    return () => {
      simulation.on("tick", null);
      simulation.stop();
    };
  }, [simulation]);

  // Fit the settled layout to the container on mount / graph change.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const { width, height } = container.getBoundingClientRect();
    if (width === 0 || height === 0) return;

    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (const node of nodes) {
      minX = Math.min(minX, node.x ?? 0);
      minY = Math.min(minY, node.y ?? 0);
      maxX = Math.max(maxX, node.x ?? 0);
      maxY = Math.max(maxY, node.y ?? 0);
    }
    const pad = 60;
    const k = Math.min(
      2,
      Math.min(
        width / (maxX - minX + pad * 2),
        height / (maxY - minY + pad * 2)
      )
    );
    setTransform({
      x: width / 2 - ((minX + maxX) / 2) * k,
      y: height / 2 - ((minY + maxY) / 2) * k,
      k,
    });
  }, [nodes]);

  // --- Pointer interactions: background pan + node drag ------------------
  const dragState = useRef<
    | { mode: "pan"; startX: number; startY: number; origin: Transform }
    | { mode: "node"; node: SimNode }
    | null
  >(null);

  const toWorld = useCallback(
    (clientX: number, clientY: number) => {
      const rect = containerRef.current!.getBoundingClientRect();
      return {
        x: (clientX - rect.left - transform.x) / transform.k,
        y: (clientY - rect.top - transform.y) / transform.k,
      };
    },
    [transform]
  );

  const onBackgroundPointerDown = (event: React.PointerEvent) => {
    (event.target as Element).setPointerCapture(event.pointerId);
    dragState.current = {
      mode: "pan",
      startX: event.clientX,
      startY: event.clientY,
      origin: transform,
    };
  };

  const onNodePointerDown = (event: React.PointerEvent, node: SimNode) => {
    event.stopPropagation();
    (event.currentTarget as Element).setPointerCapture(event.pointerId);
    dragState.current = { mode: "node", node };
    node.fx = node.x;
    node.fy = node.y;
    simulation.alphaTarget(0.25).restart();
  };

  const onPointerMove = (event: React.PointerEvent) => {
    const state = dragState.current;
    if (!state) return;
    if (state.mode === "pan") {
      setTransform({
        ...state.origin,
        x: state.origin.x + (event.clientX - state.startX),
        y: state.origin.y + (event.clientY - state.startY),
      });
    } else {
      const point = toWorld(event.clientX, event.clientY);
      state.node.fx = point.x;
      state.node.fy = point.y;
    }
  };

  const onPointerUp = () => {
    const state = dragState.current;
    if (state?.mode === "node") {
      state.node.fx = null;
      state.node.fy = null;
      simulation.alphaTarget(0);
    }
    dragState.current = null;
  };

  const zoomBy = useCallback(
    (factor: number, centerX?: number, centerY?: number) => {
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const cx = centerX ?? rect.width / 2;
      const cy = centerY ?? rect.height / 2;
      setTransform((current) => {
        const k = Math.min(4, Math.max(0.2, current.k * factor));
        const scale = k / current.k;
        return {
          k,
          x: cx - (cx - current.x) * scale,
          y: cy - (cy - current.y) * scale,
        };
      });
    },
    []
  );

  const onWheel = (event: React.WheelEvent) => {
    const rect = containerRef.current!.getBoundingClientRect();
    zoomBy(
      Math.exp(-event.deltaY * 0.0015),
      event.clientX - rect.left,
      event.clientY - rect.top
    );
  };

  // --- Highlight / search dimming ----------------------------------------
  const trimmedQuery = query.trim().toLowerCase();
  const neighborIds = hoveredId ? adjacency.get(hoveredId) : undefined;

  // An active search wins over hover: the pointer often rests on some node
  // while typing, and hover dimming would mask the search results.
  const nodeOpacity = (node: SimNode): number => {
    if (trimmedQuery) {
      return node.label.toLowerCase().includes(trimmedQuery) ? 1 : 0.12;
    }
    if (hoveredId) {
      return node.id === hoveredId || neighborIds?.has(node.id) ? 1 : 0.15;
    }
    return 1;
  };

  const linkOpacity = (link: SimLink): number => {
    const source = link.source as SimNode;
    const target = link.target as SimNode;
    if (trimmedQuery) {
      return target.label.toLowerCase().includes(trimmedQuery) ? 0.7 : 0.05;
    }
    if (hoveredId) {
      return source.id === hoveredId || target.id === hoveredId ? 0.9 : 0.05;
    }
    return 0.35;
  };

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full touch-none overflow-hidden"
    >
      <svg
        className="h-full w-full cursor-grab active:cursor-grabbing"
        role="img"
        aria-label="Graph of AI tools connected to their categories"
        onPointerDown={onBackgroundPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
      >
        <g
          transform={`translate(${transform.x}, ${transform.y}) scale(${transform.k})`}
        >
          {links.map((link, index) => {
            const source = link.source as SimNode;
            const target = link.target as SimNode;
            return (
              <line
                key={index}
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                stroke={nodeAccent(source)}
                strokeWidth={1 / transform.k}
                opacity={linkOpacity(link)}
              />
            );
          })}
          {nodes.map((node) => {
            const radius = nodeRadius(node);
            const accent = nodeAccent(node);
            const isCategory = node.type === "category";
            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                opacity={nodeOpacity(node)}
                className={isCategory ? "cursor-grab" : "cursor-pointer"}
                onPointerDown={(event) => onNodePointerDown(event, node)}
                onPointerEnter={() => setHoveredId(node.id)}
                onPointerLeave={() => setHoveredId(null)}
                onClick={() => {
                  if (!isCategory) onSelectTool(node);
                }}
              >
                <circle
                  r={radius}
                  fill={isCategory ? accent : `${accent}33`}
                  stroke={accent}
                  strokeWidth={isCategory ? 0 : 1.5}
                />
                {node.categoryIds.length > 1 && (
                  <circle
                    r={radius + 3.5}
                    fill="none"
                    stroke="#e4e4e7"
                    strokeWidth={1}
                    strokeDasharray="2.5 2.5"
                  />
                )}
                <text
                  y={radius + 12}
                  textAnchor="middle"
                  fill={isCategory ? "#fafafa" : "#a1a1aa"}
                  fontSize={isCategory ? 12 : 8.5}
                  fontWeight={isCategory ? 600 : 400}
                  style={{ pointerEvents: "none", userSelect: "none" }}
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </g>
      </svg>

      <div className="absolute bottom-4 right-4 flex flex-col gap-1">
        <button
          type="button"
          aria-label="Zoom in"
          onClick={() => zoomBy(1.4)}
          className="h-9 w-9 rounded-lg border border-zinc-700 bg-zinc-900 text-lg text-zinc-300 transition-colors hover:border-zinc-500 hover:text-white"
        >
          +
        </button>
        <button
          type="button"
          aria-label="Zoom out"
          onClick={() => zoomBy(1 / 1.4)}
          className="h-9 w-9 rounded-lg border border-zinc-700 bg-zinc-900 text-lg text-zinc-300 transition-colors hover:border-zinc-500 hover:text-white"
        >
          −
        </button>
      </div>

      <p className="pointer-events-none absolute bottom-4 left-4 text-xs text-zinc-600">
        Drag nodes · drag background to pan · scroll to zoom · click a tool for
        details
      </p>
    </div>
  );
}
