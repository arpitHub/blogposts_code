import { Fragment } from "react";

// A set score such as "22–20" or "7–6(3)"
const SET_SCORE = /(\d+–\d+(?:\(\d+\))?)/g;

/**
 * Renders running text so set scores never wrap at the dash ("22–" / "20" on a phone).
 * Each set stays whole; lines can still break between sets.
 */
export function T({ children }: { children: string }) {
  const parts = children.split(SET_SCORE);
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <span key={i} className="whitespace-nowrap">
            {part}
          </span>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </>
  );
}
