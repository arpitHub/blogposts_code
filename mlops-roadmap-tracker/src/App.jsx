import roadmap from "./data/roadmap.js";
import useLocalStorage from "./hooks/useLocalStorage.js";
import ProgressBar from "./components/ProgressBar.jsx";
import PhaseCard from "./components/PhaseCard.jsx";

const STORAGE_KEY = "mlops-roadmap-progress";

export default function App() {
  // Persisted shape: { "<phaseId>:<itemId>": true, ... }
  const [progress, setProgress] = useLocalStorage(STORAGE_KEY, {});

  const toggleItem = (phaseId, itemId) => {
    const key = `${phaseId}:${itemId}`;
    setProgress((prev) => {
      const next = { ...prev };
      if (next[key]) {
        delete next[key];
      } else {
        next[key] = true;
      }
      return next;
    });
  };

  const checkedIdsFor = (phase) =>
    new Set(
      phase.items
        .filter((item) => progress[`${phase.id}:${item.id}`])
        .map((item) => item.id)
    );

  const totalItems = roadmap.reduce((sum, p) => sum + p.items.length, 0);
  const totalDone = roadmap.reduce(
    (sum, p) => sum + checkedIdsFor(p).size,
    0
  );

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6 sm:py-12">
      <header className="mb-8">
        <h1 className="text-2xl font-bold text-slate-50 sm:text-3xl">
          MLOps Roadmap Tracker
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          6 phases · 100 days · from Python basics to Kubernetes capstone
        </p>
        <div className="mt-5 flex items-center gap-3">
          <ProgressBar value={totalDone} max={totalItems} color="#60a5fa" />
          <span className="shrink-0 text-sm font-semibold tabular-nums text-slate-300">
            {totalDone}/{totalItems}
          </span>
        </div>
      </header>

      {/* Vertical dashed "road" connecting the phases (hidden on mobile). */}
      <main className="relative sm:pl-8">
        <div
          aria-hidden="true"
          className="absolute bottom-4 left-[7px] top-4 hidden w-0 border-l-2 border-dashed border-edge sm:block"
        />
        <div className="space-y-5">
          {roadmap.map((phase) => (
            <div key={phase.id} className="relative">
              <span
                aria-hidden="true"
                className="absolute -left-8 top-6 hidden h-4 w-4 rounded-full border-2 border-night sm:block"
                style={{ backgroundColor: phase.accent }}
              />
              <PhaseCard
                phase={phase}
                checkedIds={checkedIdsFor(phase)}
                onToggleItem={toggleItem}
              />
            </div>
          ))}
        </div>
      </main>

      <footer className="mt-10 text-center text-xs text-slate-600">
        Progress is saved locally in your browser.
      </footer>
    </div>
  );
}
