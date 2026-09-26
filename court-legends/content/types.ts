/**
 * Data model for a Court Legends profile.
 *
 * Every player lives in one file under content/players/<slug>.ts and default-exports a `Player`.
 * Only use facts that appear in the player's research sheet (content/research/<slug>.md), and give
 * every chapter, record and note its sources.
 */

export type Source = {
  /** Short human label, e.g. "ITHF: Rod Laver" */
  label: string;
  url: string;
};

/** Visual theme key. Each one needs a matching `[data-theme="…"]` block in app/themes.css. */
export type ThemeName = "laver" | "ashe";

export type Match = {
  year: number;
  event: string;
  round: string;
  /** Opponent's name as it should be shown */
  opponent: string;
  /** Leave undefined when no source gives the score. The UI then says so instead of guessing */
  score?: string;
  result: "W" | "L";
  /** One sentence on why this match mattered */
  why: string;
  sources: Source[];
};

/** A scoreboard tile: one number or moment */
export type Tile = {
  label: string;
  value: string;
  caption?: string;
};

export type Chapter = {
  /** "Set" number in the match metaphor, starting at 1 */
  set: number;
  /** Short chapter name, e.g. "Roots" */
  title: string;
  years: [start: number, end: number | "present"];
  headline: string;
  /** 2–3 paragraphs of narrative */
  paragraphs: string[];
  /** Only matches with a source */
  matches?: Match[];
  tile: Tile;
  /** A short note tying the chapter to what was happening in tennis or the world */
  eraNote: { text: string; sources: Source[] };
  sources: Source[];
};

export type RecordStatus =
  /** Still unmatched today */
  | "still-stands"
  /** Was a record at the time and has since been surpassed */
  | "since-broken"
  /** A "first" that by definition can't be broken */
  | "first"
  /** Sources count it differently, so we don't call it a record */
  | "sources-differ"
  /** A career number, not a record claim */
  | "stat";

export type RecordStat = {
  label: string;
  value: string;
  /** Numeric portion to animate, if any, e.g. 11 in "11" */
  countTo?: number;
  detail: string;
  status: RecordStatus;
  sources: Source[];
};

export type NoteItem = {
  title: string;
  body: string;
  sources: Source[];
};

export type Portrait = {
  /** Big typographic fallback, e.g. "RL" */
  monogram: string;
  /** Base alt text; the fetched image's year is appended when known */
  alt: string;
  /** CSS object-position for cropping the photo into the 4:5 frame, e.g. "40% 15%". Default "50% 20%" */
  focus?: string;
  /** Wikimedia Commons lookup used by scripts/fetch-images.mjs */
  commons: {
    /** Files to try first, in order ("File:…") */
    files: string[];
    /** A category to search if none of the files qualify ("Category:…") */
    category?: string;
    /** Skip category members whose title matches this (case-insensitive) */
    exclude?: string;
    /** Photos dated inside this range are preferred, to match the theme's era */
    era: [number, number];
  };
};

export type Player = {
  slug: string;
  name: string;
  /** Surname or familiar name used in running text */
  shortName: string;
  born: { date: string; place: string };
  died?: { date: string; place: string };
  country: string;
  /** e.g. "1938 –" or "1943 – 1993" */
  lifespan: string;
  /** One line under the name */
  identity: string;
  theme: ThemeName;
  /** Era label for cards, e.g. "The 1960s" */
  eraLabel: string;
  heroFacts: { label: string; value: string }[];
  portrait: Portrait;
  journey: Chapter[];
  records: RecordStat[];
  era: { onCourt: NoteItem[]; inTheWorld: NoteItem[] };
  beyond: NoteItem[];
  legacy: string[];
  /** Things the sources disagree on, stated plainly on the page */
  sourcesDiffer: NoteItem[];
  /** General references that aren't tied to a single chapter */
  references: Source[];
};

/** Written by scripts/fetch-images.mjs into content/images.json */
export type ImageCredit = {
  src: string;
  width: number;
  height: number;
  title: string;
  author: string;
  license: string;
  licenseUrl?: string;
  sourceUrl: string;
  date?: string;
  description?: string;
};
