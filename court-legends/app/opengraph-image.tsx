import { players } from "@/content/players";
import { ogImage, ogSize } from "@/lib/og";

export const size = ogSize;
export const contentType = "image/png";
export const alt = "Court Legends: sourced profiles of tennis greats";

export default function Image() {
  return ogImage({
    theme: "home",
    kicker: "Profiles of the game's greats",
    title: "Court Legends",
    subtitle: "One full-page, sourced profile per tennis great, told set by set.",
    footer: players.map((p) => p.name).join(" · "),
  });
}
