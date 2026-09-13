import { VERDICTS } from '../data/myths';
import { useReveal } from '../hooks/useReveal';

// Tailwind needs literal class names, so the verdict colors are mapped, not built.
const STYLES = {
  busted: { word: 'text-busted', rule: 'border-busted/35' },
  confirmed: { word: 'text-confirmed', rule: 'border-confirmed/35' },
  mixed: { word: 'text-mixed', rule: 'border-mixed/35' },
};

// 'Confirmed' is the only verdict where "fact" is the right call.
function correctGuessFor(verdict) {
  return verdict === 'confirmed' ? 'fact' : 'myth';
}

export function VerdictStamp({ verdict, guess, correctReaction, incorrectReaction, closing }) {
  const [ref, visible] = useReveal({ threshold: 0.35 });
  const { word } = VERDICTS[verdict];
  const styles = STYLES[verdict];
  const reaction = guess
    ? guess === correctGuessFor(verdict)
      ? correctReaction
      : incorrectReaction
    : null;

  return (
    <section ref={ref} className="bg-surface px-5 py-14">
      <div className="mx-auto max-w-well">
        <p className="text-[13px] text-textDim">Verdict</p>
        <h2
          className={`mt-3 origin-left font-display text-[44px] font-semibold leading-[1.05] sm:text-[56px] ${
            styles.word
          } ${visible ? 'animate-stamp' : 'opacity-0'}`}
        >
          {word}
        </h2>
        {reaction ? (
          <p className={`mt-7 border-l-2 pl-4 text-[16px] leading-relaxed text-text ${styles.rule}`}>
            {reaction}
          </p>
        ) : (
          <p className="mt-7 text-[15px] leading-relaxed text-textDim">
            You scrolled past the guess on this one. Scroll back up and call it, the verdict will
            react.
          </p>
        )}
        <p className="mt-6 text-[15.5px] leading-relaxed text-textDim">{closing}</p>
      </div>
    </section>
  );
}

export default VerdictStamp;
