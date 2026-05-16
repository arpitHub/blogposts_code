export type PostStatus = "draft" | "published";

export interface Post {
  id: number;
  title: string;
  body: string;
  tags: string;
  status: PostStatus;
  created_at: string;
  updated_at: string;
}

export interface PostInput {
  title: string;
  body: string;
  tags: string;
  status: PostStatus;
}

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  listPosts: () => request<Post[]>("/posts"),
  getPost: (id: number) => request<Post>(`/posts/${id}`),
  createPost: (data: PostInput) =>
    request<Post>("/posts", { method: "POST", body: JSON.stringify(data) }),
  updatePost: (id: number, data: PostInput) =>
    request<Post>(`/posts/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),
  deletePost: (id: number) =>
    request<void>(`/posts/${id}`, { method: "DELETE" }),
};

export async function streamSuggestion(
  body: string,
  onChunk: (text: string) => void,
  signal?: AbortSignal,
) {
  const response = await fetch(`${BASE}/ai/suggest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ body }),
    signal,
  });
  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => "");
    throw new Error(`Suggest failed: ${response.status} ${text}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    if (chunk) onChunk(chunk);
  }
  const tail = decoder.decode();
  if (tail) onChunk(tail);
}
