"use client";

import { MotionConfig } from "framer-motion";
import { useEffect, useRef, type ReactNode } from "react";

/** Honour the visitor's prefers-reduced-motion setting everywhere framer-motion is used. */
export function MotionProvider({ children }: { children: ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}

/**
 * Fade and lift a block into place the first time it scrolls into view.
 * Content is visible by default: globals.css only hides [data-reveal] when JS is running and the
 * visitor hasn't asked for reduced motion, so no-JS and reduced-motion readers never wait on it.
 */
export function Reveal({ children, delay = 0, className = "" }: { children: ReactNode; delay?: number; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.dataset.shown = "";
          observer.disconnect();
        }
      },
      { rootMargin: "0px 0px -10% 0px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return (
    <div ref={ref} data-reveal="" className={className} style={delay ? { transitionDelay: `${delay}s` } : undefined}>
      {children}
    </div>
  );
}
