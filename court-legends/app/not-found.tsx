import Link from "next/link";

export default function NotFound() {
  return (
    <main id="main" tabIndex={-1} data-theme="home" className="theme-root grid min-h-screen place-items-center px-4 text-center">
      <div>
        <p className="board-text text-sm uppercase text-accent-ink">Out</p>
        <h1 className="display mt-2 text-5xl">That ball landed long</h1>
        <p className="mt-4 text-lg text-muted">There's no page at this address.</p>
        <p className="mt-6">
          <Link href="/" className="link">
            Back to all players
          </Link>
        </p>
      </div>
    </main>
  );
}
