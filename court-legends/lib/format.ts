import type { Chapter, ImageCredit, Player, Source } from "@/content/types";

export function yearRange([start, end]: Chapter["years"]): string {
  if (end === start) return String(start);
  return `${start}–${end === "present" ? "today" : end}`;
}

export function photoYear(credit?: ImageCredit): number | undefined {
  const m = credit?.date?.match(/\b(18|19|20)\d{2}\b/);
  return m ? Number(m[0]) : undefined;
}

export function portraitAlt(player: Player, credit?: ImageCredit): string {
  const year = photoYear(credit);
  return year ? `${player.portrait.alt}, ${year}` : player.portrait.alt;
}

/** Every source cited anywhere on a player's page, de-duplicated and sorted for the bibliography. */
export function allSources(player: Player): Source[] {
  const seen = new Map<string, Source>();
  const add = (list: Source[]) => list.forEach((s) => seen.has(s.url) || seen.set(s.url, s));
  player.journey.forEach((c) => {
    add(c.sources);
    add(c.eraNote.sources);
    c.matches?.forEach((m) => add(m.sources));
  });
  player.records.forEach((r) => add(r.sources));
  [...player.era.onCourt, ...player.era.inTheWorld, ...player.beyond, ...player.sourcesDiffer].forEach((n) => add(n.sources));
  add(player.references);
  return [...seen.values()].sort((a, b) => a.label.localeCompare(b.label));
}

export const siteUrl = process.env.VERCEL_PROJECT_PRODUCTION_URL
  ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
  : "https://court-legends.vercel.app";
