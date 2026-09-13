import { useReveal } from '../hooks/useReveal';

// One link in the chain: the spine runs down the left, the dot marks this
// item's place in the case being built.
export function EvidenceItem({ item, isLast = false }) {
  const [ref, visible] = useReveal();

  return (
    <li ref={ref} className={`reveal relative pb-12 pl-7 last:pb-0 ${visible ? 'reveal-in' : ''}`}>
      {!isLast && (
        <span
          aria-hidden="true"
          className="absolute bottom-0 left-[3.5px] top-3 w-px bg-hairline"
        />
      )}
      <span
        aria-hidden="true"
        className="absolute left-0 top-[9px] h-2 w-2 rounded-full bg-accent"
      />
      <p className="text-[13px] text-accent">{item.label}</p>
      <h3 className="mt-2 font-display text-[19px] font-medium leading-snug text-text">
        {item.title}
      </h3>
      <p className="mt-2.5 text-[15px] leading-relaxed text-textDim">{item.body}</p>
      <div className="mt-4 flex items-baseline gap-3 border-t border-hairline pt-4">
        <span className="font-mono text-[20px] leading-none text-text">{item.stat}</span>
        <span className="text-[13px] leading-snug text-textDim">{item.statLabel}</span>
      </div>
    </li>
  );
}

export default EvidenceItem;
