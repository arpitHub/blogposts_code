"use client";

import type { Tool } from "@/lib/types";

interface ToolCardProps {
  tool: Tool;
  accent: string;
  onSelect: () => void;
}

export default function ToolCard({ tool, accent, onSelect }: ToolCardProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className="group flex items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3 text-left transition-colors hover:border-zinc-600 hover:bg-zinc-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-500"
    >
      {tool.logo ? (
        // eslint-disable-next-line @next/next/no-img-element -- external logos, unknown hosts
        <img
          src={tool.logo}
          alt=""
          className="h-9 w-9 shrink-0 rounded-lg object-contain"
        />
      ) : (
        <span
          aria-hidden="true"
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-sm font-bold"
          style={{ backgroundColor: `${accent}22`, color: accent }}
        >
          {tool.name.charAt(0).toUpperCase()}
        </span>
      )}
      <span className="min-w-0">
        <span className="block truncate text-sm font-medium text-zinc-100 group-hover:text-white">
          {tool.name}
        </span>
        <span className="block truncate text-xs text-zinc-500">
          {tool.description}
        </span>
      </span>
    </button>
  );
}
