import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Editor } from "@tiptap/react";
import { ArrowLeft, Loader2, Save, Sparkles } from "lucide-react";

import {
  api,
  type Post,
  type PostInput,
  type PostStatus,
  streamSuggestion,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { RichEditor } from "@/components/RichEditor";

const EMPTY: PostInput = {
  title: "",
  body: "",
  tags: "",
  status: "draft",
};

export function PostEditor() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const isNew = !id || id === "new";
  const postId = isNew ? null : Number(id);

  const [form, setForm] = useState<PostInput>(EMPTY);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [suggestError, setSuggestError] = useState<string | null>(null);
  const editorRef = useRef<Editor | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const { data: post, isLoading } = useQuery<Post>({
    queryKey: ["post", postId],
    queryFn: () => api.getPost(postId as number),
    enabled: postId !== null,
  });

  useEffect(() => {
    if (post) {
      setForm({
        title: post.title,
        body: post.body,
        tags: post.tags,
        status: post.status,
      });
    } else if (isNew) {
      setForm(EMPTY);
    }
  }, [post, isNew]);

  const saveMutation = useMutation({
    mutationFn: async (input: PostInput) => {
      if (postId === null) return api.createPost(input);
      return api.updatePost(postId, input);
    },
    onSuccess: (saved) => {
      queryClient.invalidateQueries({ queryKey: ["posts"] });
      queryClient.setQueryData(["post", saved.id], saved);
      if (isNew) navigate(`/posts/${saved.id}`, { replace: true });
    },
  });

  async function handleSuggest() {
    const editor = editorRef.current;
    if (!editor) return;
    setSuggestError(null);
    setIsSuggesting(true);
    const controller = new AbortController();
    abortRef.current = controller;

    const plainText = editor.getText();
    let inserted = false;

    try {
      await streamSuggestion(
        plainText,
        (chunk) => {
          if (!inserted) {
            editor.chain().focus("end").insertContent("<p></p>").run();
            inserted = true;
          }
          editor.chain().focus("end").insertContent(chunk).run();
        },
        controller.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setSuggestError((err as Error).message);
      }
    } finally {
      setIsSuggesting(false);
      abortRef.current = null;
    }
  }

  function handleStopSuggest() {
    abortRef.current?.abort();
  }

  function handleSave() {
    saveMutation.mutate(form);
  }

  if (!isNew && isLoading) {
    return <p className="text-sm text-muted-foreground">Loading post…</p>;
  }

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate("/posts")}
            aria-label="Back to posts"
          >
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h1 className="text-2xl font-semibold text-slate-900">
              {isNew ? "New post" : "Edit post"}
            </h1>
            <p className="text-sm text-muted-foreground">
              {isNew
                ? "Draft a new post, then publish when you're ready."
                : "Update content, tags, or status."}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusToggle
            value={form.status}
            onChange={(status) => setForm((f) => ({ ...f, status }))}
          />
          <Button onClick={handleSave} disabled={saveMutation.isPending}>
            {saveMutation.isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Save className="mr-2 h-4 w-4" />
            )}
            Save
          </Button>
        </div>
      </header>

      <Card>
        <CardContent className="space-y-5 pt-6">
          <div>
            <label className="mb-1.5 block text-sm font-medium text-slate-700">
              Title
            </label>
            <Input
              value={form.title}
              onChange={(e) =>
                setForm((f) => ({ ...f, title: e.target.value }))
              }
              placeholder="A great post needs a great title"
              className="text-lg"
            />
          </div>

          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <label className="text-sm font-medium text-slate-700">Body</label>
              {isSuggesting ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleStopSuggest}
                >
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Stop
                </Button>
              ) : (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleSuggest}
                >
                  <Sparkles className="mr-2 h-4 w-4 text-amber-500" />
                  Suggest continuation
                </Button>
              )}
            </div>
            <RichEditor
              value={form.body}
              onChange={(html) => setForm((f) => ({ ...f, body: html }))}
              onEditorReady={(editor) => {
                editorRef.current = editor;
              }}
            />
            {suggestError && (
              <p className="mt-2 text-xs text-destructive">{suggestError}</p>
            )}
          </div>

          <div>
            <label className="mb-1.5 block text-sm font-medium text-slate-700">
              Tags
            </label>
            <Input
              value={form.tags}
              onChange={(e) => setForm((f) => ({ ...f, tags: e.target.value }))}
              placeholder="comma, separated, tags"
            />
            <p className="mt-1 text-xs text-muted-foreground">
              Separate tags with commas.
            </p>
          </div>

          {saveMutation.isError && (
            <p className="text-sm text-destructive">
              Failed to save: {(saveMutation.error as Error).message}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function StatusToggle({
  value,
  onChange,
}: {
  value: PostStatus;
  onChange: (status: PostStatus) => void;
}) {
  const next: PostStatus = value === "draft" ? "published" : "draft";
  return (
    <button
      type="button"
      onClick={() => onChange(next)}
      className="inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 py-2 text-sm transition-colors hover:bg-accent"
      aria-label={`Toggle status (currently ${value})`}
    >
      <span className="text-muted-foreground">Status:</span>
      <Badge variant={value === "published" ? "success" : "muted"}>
        {value}
      </Badge>
    </button>
  );
}
