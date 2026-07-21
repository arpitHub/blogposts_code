import Link from "next/link";
import Directory from "@/components/Directory";
import { getCategories, getUnverifiedTools } from "@/lib/data";

export default function Home() {
  const categories = getCategories();
  const unverified = getUnverifiedTools();
  const totalTools = categories.reduce(
    (sum, category) => sum + category.tools.length,
    0
  );

  return (
    <div className="flex min-h-screen flex-col">
      <header className="mx-auto w-full max-w-7xl px-4 pb-6 pt-10 sm:px-6">
        <div className="flex items-baseline gap-4">
          <h1 className="text-2xl font-bold tracking-tight text-zinc-50 sm:text-3xl">
            AI Ecosystem Explorer
          </h1>
          <nav className="ml-auto flex gap-4 text-sm">
            <span aria-current="page" className="font-medium text-zinc-100">
              Directory
            </span>
            <Link
              href="/explore"
              className="text-zinc-400 transition-colors hover:text-white"
            >
              Explore
            </Link>
          </nav>
        </div>
        <p className="mt-2 max-w-2xl text-sm text-zinc-400">
          A directory of {totalTools} tools and frameworks across the modern AI
          stack — LLMs, agents, RAG, vector databases, and more. Search, browse
          by category, and click any card for details.
        </p>
      </header>
      <Directory categories={categories} unverifiedCount={unverified.length} />
    </div>
  );
}
