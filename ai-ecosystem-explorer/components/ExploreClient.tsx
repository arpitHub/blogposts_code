"use client";

import { useMemo, useState } from "react";
import type { GraphData, GraphNode } from "@/lib/graph";
import type { ToolEntry } from "@/lib/types";
import ExploreGraph from "./ExploreGraph";
import ToolModal from "./ToolModal";

interface ExploreClientProps {
  graph: GraphData;
}

export default function ExploreClient({ graph }: ExploreClientProps) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<ToolEntry | null>(null);

  const categoryNames = useMemo(() => {
    const map = new Map<string, string>();
    for (const node of graph.nodes) {
      if (node.type === "category") map.set(node.categoryIds[0], node.label);
    }
    return map;
  }, [graph]);

  const onSelectTool = (node: GraphNode) => {
    if (!node.tool) return;
    const categoryId = node.categoryIds[0];
    setSelected({
      tool: node.tool,
      categoryId,
      categoryName: categoryNames.get(categoryId) ?? categoryId,
    });
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="mx-auto w-full max-w-7xl px-4 pb-3 sm:px-6">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Highlight tools by name…"
          aria-label="Highlight tools"
          className="w-full max-w-sm rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-600"
        />
      </div>
      <div className="min-h-0 flex-1">
        <ExploreGraph graph={graph} query={query} onSelectTool={onSelectTool} />
      </div>
      {selected && (
        <ToolModal entry={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}
