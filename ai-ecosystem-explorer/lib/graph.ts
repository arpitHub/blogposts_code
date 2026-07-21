import type { Tool } from "./types";
import { getCategories } from "./data";

export interface GraphNode {
  id: string;
  type: "category" | "tool";
  label: string;
  /** Category ids this node belongs to; a category node lists itself. */
  categoryIds: string[];
  /** Present on tool nodes only. */
  tool?: Tool;
}

export interface GraphLink {
  source: string;
  target: string;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

// Tools that appear in several categories (e.g. Redis under both Memory and
// Vector Database) share a name but have distinct ids in the seed data.
// Merge them into one node with an edge to each category.
function toolKey(tool: Tool): string {
  return tool.name.trim().toLowerCase();
}

export function buildGraph(): GraphData {
  const categories = getCategories();
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  const toolNodesByKey = new Map<string, GraphNode>();

  for (const category of categories) {
    nodes.push({
      id: `category:${category.id}`,
      type: "category",
      label: category.name,
      categoryIds: [category.id],
    });
  }

  for (const category of categories) {
    for (const tool of category.tools) {
      const key = toolKey(tool);
      let node = toolNodesByKey.get(key);
      if (!node) {
        node = {
          id: `tool:${tool.id}`,
          type: "tool",
          label: tool.name,
          categoryIds: [],
          tool,
        };
        toolNodesByKey.set(key, node);
        nodes.push(node);
      }
      node.categoryIds.push(category.id);
      links.push({ source: `category:${category.id}`, target: node.id });
    }
  }

  return { nodes, links };
}
