import { useMythStore } from '../store/useMythStore';
import { useReveal } from '../hooks/useReveal';

const OPTIONS = [
  { value: 'myth', label: 'Myth' },
  { value: 'fact', label: 'Fact' },
];

export function GuessButtons({ slug }) {
  const guess = useMythStore((state) => state.guesses[slug]);
  const setGuess = useMythStore((state) => state.setGuess);
  const [ref, visible] = useReveal();

  return (
    <section ref={ref} className={`reveal ${visible ? 'reveal-in' : ''}`}>
      <h2 className="font-display text-[20px] font-medium text-text">
        Call it before you read on
      </h2>
      <p className="mt-2 text-[15px] leading-relaxed text-textDim">
        {guess
          ? 'Your call is saved. Change it any time before the verdict.'
          : 'Pick one. The verdict at the end will remember what you said.'}
      </p>
      <div className="mt-5 flex gap-3">
        {OPTIONS.map((option) => {
          const isSelected = guess === option.value;
          return (
            <button
              key={option.value}
              type="button"
              onClick={() => setGuess(slug, option.value)}
              aria-pressed={isSelected}
              className={`flex-1 border px-5 py-3 font-body text-[15px] transition-colors ${
                isSelected
                  ? 'border-accent text-accent'
                  : 'border-hairline text-text hover:border-textDim'
              }`}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </section>
  );
}

export default GuessButtons;
