import { useMemo, useState } from 'react';
import { myths } from '../data/myths';
import CategoryFilter from '../components/CategoryFilter';
import MythCard from '../components/MythCard';

export function Hub() {
  const [category, setCategory] = useState('All');

  const visible = useMemo(
    () => (category === 'All' ? myths : myths.filter((m) => m.category === category)),
    [category]
  );

  return (
    <main className="mx-auto max-w-4xl px-5 pb-24 pt-24">
      <h1 className="font-display text-[40px] font-semibold leading-[1.05] tracking-tight text-text sm:text-[52px]">
        Mythcraft
      </h1>
      <p className="mt-4 max-w-well text-[17px] leading-relaxed text-textDim">
        Ten claims. Some hold up. Most don&apos;t. Guess first, then read the case: where each one
        came from, three pieces of evidence, and a verdict.
      </p>

      <div className="mt-10 border-b border-hairline pb-px">
        <CategoryFilter active={category} onChange={setCategory} />
      </div>

      <ul className="mt-8 grid grid-cols-1 gap-4 md:grid-cols-2">
        {visible.map((myth) => (
          <MythCard key={myth.slug} myth={myth} />
        ))}
      </ul>
    </main>
  );
}

export default Hub;
