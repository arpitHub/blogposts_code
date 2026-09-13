import { Link } from 'react-router-dom';
import { Check } from 'lucide-react';
import { useMythStore } from '../store/useMythStore';
import { useReveal } from '../hooks/useReveal';

export function MythCard({ myth }) {
  const guess = useMythStore((state) => state.guesses[myth.slug]);
  const [ref, visible] = useReveal();

  return (
    <li ref={ref} className={`reveal h-full ${visible ? 'reveal-in' : ''}`}>
      <Link
        to={`/myth/${myth.slug}`}
        className="flex h-full flex-col border border-hairline p-5 transition-colors hover:border-accent/45"
      >
        <div className="flex items-center justify-between gap-3">
          <span className="text-[13px] text-textDim">{myth.category}</span>
          {guess ? (
            <span
              className="flex items-center gap-1 text-[12px] text-textDim"
              title="You've already guessed this one"
            >
              <Check size={13} strokeWidth={2} aria-hidden="true" />
              guessed
            </span>
          ) : null}
        </div>
        <h2 className="mt-3 font-display text-[21px] font-medium leading-snug text-text">
          {myth.claim}
        </h2>
        <p className="mt-2.5 text-[14.5px] leading-relaxed text-textDim">{myth.hook}</p>
      </Link>
    </li>
  );
}

export default MythCard;
