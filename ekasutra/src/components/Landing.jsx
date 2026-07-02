import { useEffect, useState } from 'react';

export default function Landing({ onEnter }) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const t = requestAnimationFrame(() => setVisible(true));
    return () => cancelAnimationFrame(t);
  }, []);

  return (
    <section className={`landing ${visible ? 'is-visible' : ''}`}>
      <div className="landing-stars" aria-hidden="true" />
      <div className="landing-gradient" aria-hidden="true" />

      <div className="landing-content">
        <h1 className="landing-wordmark">Ekasutra</h1>
        <div className="landing-thread" aria-hidden="true" />
        <p className="landing-sanskrit">eka&nbsp;sūtra&nbsp;— one thread</p>
        <p className="landing-tagline">Two epics. One truth.</p>

        <button
          type="button"
          className="landing-cta"
          onClick={onEnter}
        >
          Explore the Threads
        </button>
      </div>
    </section>
  );
}
