import { getPlayer, players } from "@/content/players";
import { ogImage, ogSize } from "@/lib/og";

export const size = ogSize;
export const contentType = "image/png";
export const alt = "Court Legends player profile";

export function generateStaticParams() {
  return players.map((p) => ({ slug: p.slug }));
}

export default async function Image({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const p = getPlayer(slug)!;
  return ogImage({ theme: p.theme, kicker: `${p.country} · ${p.lifespan}`, title: p.name, subtitle: p.identity, footer: `${p.journey.length} sets, fully sourced` });
}
