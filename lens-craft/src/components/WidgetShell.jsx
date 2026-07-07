// Standard two-panel widget layout extracted from the Exposure Triangle module:
// live preview on the left, controls stacked on the right.
export default function WidgetShell({ preview, previewFooter, controls }) {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.3fr_1fr]">
      <div className="overflow-hidden rounded-2xl border border-line bg-panel">
        <div className="aspect-[8/5] w-full">{preview}</div>
        {previewFooter && <div className="border-t border-line px-4 py-3">{previewFooter}</div>}
      </div>
      <div className="flex flex-col gap-6 rounded-2xl border border-line bg-panel p-5">{controls}</div>
    </div>
  );
}

export function Stat({ label, value }) {
  return (
    <div className="rounded-lg bg-panel-2 px-2 py-2 text-center">
      <div className="font-mono text-[10px] uppercase tracking-wide text-ink-3">{label}</div>
      <div className="mt-0.5 text-sm font-medium text-ink">{value}</div>
    </div>
  );
}
