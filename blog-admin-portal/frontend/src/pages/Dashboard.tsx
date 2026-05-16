import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { FileText, FilePlus, FileCheck, Clock } from "lucide-react";

import { api, type Post } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { formatDate } from "@/lib/utils";

export function Dashboard() {
  const { data, isLoading } = useQuery({
    queryKey: ["posts"],
    queryFn: api.listPosts,
  });

  const posts = data ?? [];
  const total = posts.length;
  const published = posts.filter((p) => p.status === "published").length;
  const drafts = total - published;
  const recent = [...posts]
    .sort(
      (a, b) =>
        new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
    )
    .slice(0, 5);

  return (
    <div className="space-y-6">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            Overview of your blog content.
          </p>
        </div>
        <Button asChild>
          <Link to="/posts/new">New post</Link>
        </Button>
      </header>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard
          label="Total posts"
          value={isLoading ? "—" : total}
          icon={<FileText className="h-5 w-5" />}
          tint="bg-sky-100 text-sky-700"
        />
        <StatCard
          label="Published"
          value={isLoading ? "—" : published}
          icon={<FileCheck className="h-5 w-5" />}
          tint="bg-emerald-100 text-emerald-700"
        />
        <StatCard
          label="Drafts"
          value={isLoading ? "—" : drafts}
          icon={<FilePlus className="h-5 w-5" />}
          tint="bg-amber-100 text-amber-700"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            Recently updated
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : recent.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No posts yet. Create one to get started.
            </p>
          ) : (
            <ul className="divide-y">
              {recent.map((post) => (
                <RecentRow key={post.id} post={post} />
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function StatCard({
  label,
  value,
  icon,
  tint,
}: {
  label: string;
  value: number | string;
  icon: React.ReactNode;
  tint: string;
}) {
  return (
    <Card>
      <CardContent className="flex items-center justify-between pt-6">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {label}
          </p>
          <p className="mt-2 text-3xl font-semibold text-slate-900">{value}</p>
        </div>
        <div
          className={`flex h-10 w-10 items-center justify-center rounded-md ${tint}`}
        >
          {icon}
        </div>
      </CardContent>
    </Card>
  );
}

function RecentRow({ post }: { post: Post }) {
  return (
    <li className="flex items-center justify-between py-3">
      <div className="min-w-0 flex-1 pr-4">
        <Link
          to={`/posts/${post.id}`}
          className="block truncate text-sm font-medium text-slate-900 hover:text-primary"
        >
          {post.title || "(untitled)"}
        </Link>
        <p className="text-xs text-muted-foreground">
          Updated {formatDate(post.updated_at)}
        </p>
      </div>
      <Badge variant={post.status === "published" ? "success" : "muted"}>
        {post.status}
      </Badge>
    </li>
  );
}
