export interface Tool {
  id: string;
  name: string;
  description: string;
  website: string;
  logo?: string;
  tags?: string[];
}

export interface Category {
  id: string;
  name: string;
  tools: Tool[];
}

export interface ToolsData {
  categories: Category[];
}

// Tool paired with its category — the flat shape used by search and the
// modal, and the shape Phase 2's graph view will consume as nodes.
export interface ToolEntry {
  tool: Tool;
  categoryId: string;
  categoryName: string;
}
