import rawData from "@/data/tools-data.json";
import type { Category, ToolEntry, ToolsData } from "./types";

const data = rawData as ToolsData;

export function getCategories(): Category[] {
  return data.categories;
}

export function getAllToolEntries(): ToolEntry[] {
  return data.categories.flatMap((category) =>
    category.tools.map((tool) => ({
      tool,
      categoryId: category.id,
      categoryName: category.name,
    }))
  );
}

// Tools whose names were cut off or unclear in the source infographic.
export function getUnverifiedTools(): ToolEntry[] {
  return getAllToolEntries().filter(
    ({ tool }) =>
      tool.id.endsWith("-unverified") || tool.name.includes("(verify name")
  );
}

// One accent per category, used for headers and card hover states.
// Falls back to the last entry for categories added later.
const CATEGORY_ACCENTS: Record<string, string> = {
  llm: "#f97316", // orange
  "agentic-ai": "#a78bfa", // violet
  rag: "#34d399", // emerald
  embedding: "#f472b6", // pink
  mcp: "#38bdf8", // sky
  "ai-security": "#f87171", // red
  observability: "#facc15", // yellow
  memory: "#4ade80", // green
  "ai-agent": "#c084fc", // purple
  automation: "#fb923c", // light orange
  "vector-db": "#22d3ee", // cyan
};

const DEFAULT_ACCENT = "#94a3b8"; // slate

export function getCategoryAccent(categoryId: string): string {
  return CATEGORY_ACCENTS[categoryId] ?? DEFAULT_ACCENT;
}
