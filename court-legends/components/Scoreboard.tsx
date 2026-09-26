"use client";

import { animate, useInView, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

/**
 * A number that counts up when it scrolls into view. The server renders the final value (good for
 * SEO, no-JS and no layout shift); the count only runs when motion is allowed.
 */
export function CountUp({ value, countTo }: { value: string; countTo?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "0px 0px -10% 0px" });
  const reduce = useReducedMotion();
  const [display, setDisplay] = useState(value);
  const [armed, setArmed] = useState(false);

  // Arm the count only for tiles that start off-screen, so nothing visible ever flashes to 0.
  useEffect(() => {
    if (countTo === undefined || reduce || !ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    if (rect.top > window.innerHeight) {
      setDisplay(format(0, value));
      setArmed(true);
    }
  }, [countTo, reduce, value]);

  useEffect(() => {
    if (!armed || !inView || countTo === undefined) return;
    const controls = animate(0, countTo, {
      duration: Math.min(1.4, 0.5 + countTo / 20),
      ease: [0.2, 0.7, 0.2, 1],
      onUpdate: (v) => setDisplay(format(Math.round(v), value)),
      onComplete: () => setDisplay(value),
    });
    return () => controls.stop();
  }, [armed, inView, countTo, value]);

  return (
    <span ref={ref}>
      <span className="sr-only">{value}</span>
      <span aria-hidden="true">{display}</span>
    </span>
  );
}

/** Keep any prefix/suffix around the number, e.g. "$14,000". */
function format(n: number, template: string) {
  const match = template.match(/^(\D*)[\d,]+(.*)$/);
  if (!match) return String(n);
  const withCommas = template.includes(",") ? n.toLocaleString("en-US") : String(n);
  return `${match[1]}${withCommas}${match[2]}`;
}
