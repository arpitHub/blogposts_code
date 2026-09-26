"use client";

import { motion, useScroll, useSpring } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";

export type RailItem = { set: number; title: string; years: string; startYear: number };

/**
 * Wraps the server-rendered chapters with navigation that tracks the reader:
 * a sticky year rail on wide screens, and a sticky progress bar on phones.
 */
export function JourneyShell({ items, children }: { items: RailItem[]; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(items[0]?.set ?? 1);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 0.2", "end 0.8"] });
  const progress = useSpring(scrollYProgress, { stiffness: 140, damping: 30, restDelta: 0.001 });

  useEffect(() => {
    const sections = items
      .map((i) => document.getElementById(`set-${i.set}`))
      .filter((el): el is HTMLElement => Boolean(el));
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) setActive(Number(e.target.getAttribute("data-set")));
        }
      },
      { rootMargin: "-40% 0px -55% 0px" },
    );
    sections.forEach((s) => observer.observe(s));
    return () => observer.disconnect();
  }, [items]);

  const current = items.find((i) => i.set === active) ?? items[0];

  return (
    <div ref={ref} className="relative lg:grid lg:grid-cols-[200px_1fr] lg:gap-14">
      {/* Phones and tablets: a thin scoreboard strip pinned under the top of the screen */}
      <div className="sticky top-0 z-20 -mx-4 mb-8 border-b-2 border-board bg-board px-4 py-2 text-board-ink sm:-mx-8 sm:px-8 lg:hidden">
        <p className="board-text flex items-baseline justify-between text-sm uppercase">
          <span>
            Set {current.set} · {current.title}
          </span>
          <span className="text-board-accent">{current.years}</span>
        </p>
        <motion.span
          aria-hidden="true"
          className="absolute inset-x-0 bottom-0 h-[3px] origin-left bg-board-accent"
          style={{ scaleX: progress }}
        />
      </div>

      {/* Wide screens: the year rail, drawn as a sideline with the sets marked along it */}
      <nav aria-label="Journey chapters" className="hidden lg:block">
        <div className="sticky top-8">
          <p className="board-text mb-4 text-xs uppercase text-muted">Career</p>
          <div className="relative pl-6">
            <span aria-hidden="true" className="absolute bottom-1 left-[7px] top-1 w-[3px] bg-ink/15" />
            <motion.span
              aria-hidden="true"
              className="absolute left-[7px] top-1 w-[3px] origin-top bg-accent"
              style={{ scaleY: progress, bottom: "0.25rem" }}
            />
            <ol className="space-y-1">
              {items.map((i) => {
                const isActive = i.set === active;
                return (
                  <li key={i.set} className="relative">
                    <span
                      aria-hidden="true"
                      className={`absolute -left-[23px] top-[0.55rem] h-3 w-3 rounded-full border-2 transition-colors ${
                        isActive ? "border-ink bg-[#d9e24a]" : "border-ink/30 bg-paper"
                      }`}
                    />
                    <a
                      href={`#set-${i.set}`}
                      aria-current={isActive ? "location" : undefined}
                      className={`block rounded px-2 py-1 transition-colors hover:bg-paper-2 ${isActive ? "bg-paper-2" : ""}`}
                    >
                      <span className="board-text block text-base leading-tight text-ink">{i.startYear}</span>
                      <span className={`block text-xs leading-snug ${isActive ? "text-ink" : "text-muted"}`}>
                        Set {i.set} · {i.title}
                      </span>
                    </a>
                  </li>
                );
              })}
            </ol>
          </div>
        </div>
      </nav>

      <div className="min-w-0">{children}</div>
    </div>
  );
}
