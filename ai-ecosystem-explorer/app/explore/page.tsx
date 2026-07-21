import type { Metadata } from "next";
import Link from "next/link";
import ExploreClient from "@/components/ExploreClient";
import { buildGraph } from "@/lib/graph";

export const metadata: Metadata = {
  title: "Explore — AI Ecosystem Explorer",
  description:
    "Interactive graph of AI tools and frameworks connected to their categories.",
};

export default function ExplorePage() {
  const graph = buildGraph();

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <header className="mx-auto flex w-full max-w-7xl items-baseline gap-4 px-4 pb-4 pt-8 sm:px-6">
        <h1 className="text-2xl font-bold tracking-tight text-zinc-50 sm:text-3xl">
          Explore
        </h1>
        <nav className="ml-auto flex gap-4 text-sm">
          <Link
            href="/"
            className="text-zinc-400 transition-colors hover:text-white"
          >
            Directory
          </Link>
          <span aria-current="page" className="font-medium text-zinc-100">
            Explore
          </span>
        </nav>
      </header>
      <ExploreClient graph={graph} />
    </div>
  );
}
