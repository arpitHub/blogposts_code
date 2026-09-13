import { useEffect } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import { getMyth, getRelatedMyths } from '../data/myths';
import { useMythStore } from '../store/useMythStore';
import { useReveal } from '../hooks/useReveal';
import GuessButtons from '../components/GuessButtons';
import EvidenceItem from '../components/EvidenceItem';
import VerdictStamp from '../components/VerdictStamp';

function Origin({ paragraphs }) {
  const [ref, visible] = useReveal();
  return (
    <section ref={ref} className={`reveal bg-surface px-5 py-12 ${visible ? 'reveal-in' : ''}`}>
      <div className="mx-auto max-w-well">
        <h2 className="font-display text-[20px] font-medium text-text">Where it came from</h2>
        {paragraphs.map((paragraph) => (
          <p key={paragraph.slice(0, 40)} className="mt-4 text-[15.5px] leading-relaxed text-textDim">
            {paragraph}
          </p>
        ))}
      </div>
    </section>
  );
}

export function MythDeepDive() {
  const { slug } = useParams();
  const myth = getMyth(slug);
  const guess = useMythStore((state) => state.guesses[slug]);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [slug]);

  if (!myth) return <Navigate to="/" replace />;

  const related = getRelatedMyths(slug);

  return (
    <main className="pt-14">
      <header className="mx-auto max-w-well px-5 pb-14 pt-12">
        <p className="text-[13px] text-textDim">{myth.category}</p>
        <h1 className="mt-4 font-display text-[34px] font-semibold leading-[1.12] tracking-tight text-text sm:text-[44px]">
          {myth.claim}
        </h1>
        <p className="mt-5 text-[17px] leading-relaxed text-textDim">{myth.hook}</p>
      </header>

      <div className="mx-auto max-w-well px-5 pb-14">
        <GuessButtons slug={myth.slug} />
      </div>

      <Origin paragraphs={myth.origin} />

      <section className="mx-auto max-w-well px-5 py-14">
        <h2 className="font-display text-[20px] font-medium text-text">The evidence</h2>
        <ol className="mt-8">
          {myth.evidence.map((item, index) => (
            <EvidenceItem
              key={item.title}
              item={item}
              isLast={index === myth.evidence.length - 1}
            />
          ))}
        </ol>
      </section>

      <VerdictStamp
        verdict={myth.verdict}
        guess={guess}
        correctReaction={myth.correctReaction}
        incorrectReaction={myth.incorrectReaction}
        closing={myth.closing}
      />

      <footer className="mx-auto max-w-well px-5 py-14">
        <h2 className="font-display text-[20px] font-medium text-text">Keep going</h2>
        <ul className="mt-5 border-t border-hairline">
          {related.map((other) => (
            <li key={other.slug} className="border-b border-hairline">
              <Link
                to={`/myth/${other.slug}`}
                className="block py-4 transition-colors hover:text-accent"
              >
                <span className="block text-[13px] text-textDim">{other.category}</span>
                <span className="mt-1 block font-display text-[18px] font-medium leading-snug">
                  {other.claim}
                </span>
              </Link>
            </li>
          ))}
        </ul>
        <Link to="/" className="mt-7 inline-block text-[14.5px] text-textDim hover:text-text">
          All ten myths
        </Link>
      </footer>
    </main>
  );
}

export default MythDeepDive;
