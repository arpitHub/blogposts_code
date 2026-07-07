"use client";

import { useState } from "react";
import type { Category, Tool } from "@/lib/types";
import { getCategoryAccent } from "@/lib/data";
import ToolCard from "./ToolCard";

interface CategorySectionProps {
  category: Category;
  onSelectTool: (tool: Tool) => void;
}

export default function CategorySection({
  category,
  onSelectTool,
}: CategorySectionProps) {
  const [collapsed, setCollapsed] = useState(false);
  const accent = getCategoryAccent(category.id);

  return (
    <section className="pt-10 first:pt-8" aria-label={category.name}>
      <button
        type="button"
        onClick={() => setCollapsed((value) => !value)}
        aria-expanded={!collapsed}
        className="group flex w-full items-center gap-3 text-left"
      >
        <span
          aria-hidden="true"
          className="h-5 w-1 rounded-full"
          style={{ backgroundColor: accent }}
        />
        <h2 className="text-lg font-semibold tracking-tight text-zinc-100">
          {category.name}
        </h2>
        <span className="text-xs text-zinc-500">
          {category.tools.length}
        </span>
        <svg
          aria-hidden="true"
          viewBox="0 0 20 20"
          fill="currentColor"
          className={`ml-auto h-4 w-4 text-zinc-600 transition-transform group-hover:text-zinc-400 ${
            collapsed ? "-rotate-90" : ""
          }`}
        >
          <path
            fillRule="evenodd"
            d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      {!collapsed && (
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {category.tools.map((tool) => (
            <ToolCard
              key={tool.id}
              tool={tool}
              accent={accent}
              onSelect={() => onSelectTool(tool)}
            />
          ))}
        </div>
      )}
    </section>
  );
}
