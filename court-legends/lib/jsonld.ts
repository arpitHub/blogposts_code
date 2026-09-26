import type { Player } from "@/content/types";
import { siteUrl } from "./format";

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/** "9 August 1938" → "1938-08-09" */
function isoDate(date: string): string | undefined {
  const m = date.match(/^(\d{1,2}) (\w+) (\d{4})$/);
  if (!m) return undefined;
  const month = MONTHS.indexOf(m[2]) + 1;
  return month ? `${m[3]}-${String(month).padStart(2, "0")}-${m[1].padStart(2, "0")}` : undefined;
}

/** schema.org Person data for search engines, built only from fields already on the page. */
export function personJsonLd(player: Player) {
  return {
    "@context": "https://schema.org",
    "@type": "ProfilePage",
    url: `${siteUrl}/players/${player.slug}`,
    mainEntity: {
      "@type": "Person",
      name: player.name,
      description: player.identity,
      birthDate: isoDate(player.born.date),
      birthPlace: player.born.place,
      deathDate: player.died ? isoDate(player.died.date) : undefined,
      deathPlace: player.died?.place,
      nationality: player.country,
      jobTitle: "Tennis player",
      sameAs: player.references.map((r) => r.url),
    },
  };
}
