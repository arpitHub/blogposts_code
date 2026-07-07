"use client";

import { useMemo, useState } from "react";
import Fuse from "fuse.js";
import type { Category, ToolEntry } from "@/lib/types";
import SearchBar from "./SearchBar";
import CategorySection from "./CategorySection";
import ToolModal from "./ToolModal";

interface DirectoryProps {
  categories: Category[];
  unverifiedCount: number;
}

export default function Directory({
  categories,
  unverifiedCount,
}: DirectoryProps) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<ToolEntry | null>(null);

  const entries = useMemo<ToolEntry[]>(
    () =>
      categories.flatMap((category) =>
        category.tools.map((tool) => ({
          tool,
          categoryId: category.id,
          categoryName: category.name,
        }))
      ),
    [categories]
  );

  const fuse = useMemo(
    () =>
      new Fuse(entries, {
        keys: [
          { name: "tool.name", weight: 2 },
          { name: "tool.description", weight: 1 },
        ],
        threshold: 0.35,
        ignoreLocation: true,
      }),
    [entries]
  );

  // Categories with only the tools matching the query; empty categories drop out.
  const visibleCategories = useMemo(() => {
    const trimmed = query.trim();
    if (!trimmed) return categories;

    const matchedIds = new Set(
      fuse.search(trimmed).map((result) => result.item.tool.id)
    );
    return categories
      .map((category) => ({
        ...category,
        tools: category.tools.filter((tool) => matchedIds.has(tool.id)),
      }))
      .filter((category) => category.tools.length > 0);
  }, [categories, fuse, query]);

  const totalVisible = visibleCategories.reduce(
    (sum, category) => sum + category.tools.length,
    0
  );

  return (
    <>
      <SearchBar
        query={query}
        onQueryChange={setQuery}
        resultCount={query.trim() ? totalVisible : null}
      />

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 pb-16 sm:px-6">
        {visibleCategories.length === 0 ? (
          <div className="py-24 text-center">
            <p className="text-lg font-medium text-zinc-300">
              No tools match &ldquo;{query}&rdquo;
            </p>
            <p className="mt-2 text-sm text-zinc-500">
              Try a different name or keyword, or clear the search.
            </p>
            <button
              type="button"
              onClick={() => setQuery("")}
              className="mt-6 rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-300 transition-colors hover:border-zinc-500 hover:text-white"
            >
              Clear search
            </button>
          </div>
        ) : (
          visibleCategories.map((category) => (
            <CategorySection
              key={category.id}
              category={category}
              onSelectTool={(tool) =>
                setSelected({
                  tool,
                  categoryId: category.id,
                  categoryName: category.name,
                })
              }
            />
          ))
        )}

        {unverifiedCount > 0 && (
          <p className="mt-12 border-t border-zinc-800 pt-6 text-xs text-zinc-600">
            Note: {unverifiedCount} tool name
            {unverifiedCount === 1 ? " was" : "s were"} unclear in the source
            infographic and {unverifiedCount === 1 ? "is" : "are"} marked
            &ldquo;(verify name)&rdquo; pending correction.
          </p>
        )}
      </main>

      {selected && (
        <ToolModal entry={selected} onClose={() => setSelected(null)} />
      )}
    </>
  );
}
