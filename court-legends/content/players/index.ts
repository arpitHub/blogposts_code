import type { ImageCredit, Player } from "../types";
import images from "../images.json";
import ashe from "./ashe";
import laver from "./laver";

/**
 * Registered players, in the order the home page lists them.
 * To add a player: create content/players/<slug>.ts, add a theme in app/themes.css,
 * add the slug to ThemeName in content/types.ts, and import it here.
 */
export const players: Player[] = [laver, ashe];

export function getPlayer(slug: string): Player | undefined {
  return players.find((p) => p.slug === slug);
}

/** Portrait fetched from Wikimedia Commons by scripts/fetch-images.mjs, if one qualified */
export function getPortrait(slug: string): ImageCredit | undefined {
  return (images as Record<string, ImageCredit>)[slug];
}
