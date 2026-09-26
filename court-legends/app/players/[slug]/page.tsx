import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { NetDivider } from "@/components/CourtLines";
import { Hero } from "@/components/Hero";
import { Journey } from "@/components/Journey";
import { BeyondTheCourt, EraContext, LegacyAndSources, Records } from "@/components/Sections";
import { SiteFooter, SiteHeader, SkipLink } from "@/components/SiteChrome";
import { NextPlayer, PageNav } from "@/components/PageNav";
import { personJsonLd } from "@/lib/jsonld";
import { getPlayer, getPortrait, players } from "@/content/players";

export const dynamicParams = false;

export function generateStaticParams() {
  return players.map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({ params }: PageProps<"/players/[slug]">): Promise<Metadata> {
  const { slug } = await params;
  const player = getPlayer(slug);
  if (!player) return {};
  const description = `${player.name} (${player.lifespan}), ${player.country}. ${player.identity}`;
  return {
    title: player.name,
    description,
    alternates: { canonical: `/players/${player.slug}` },
    openGraph: { title: `${player.name} · Court Legends`, description, type: "profile", url: `/players/${player.slug}` },
    twitter: { card: "summary_large_image", title: `${player.name} · Court Legends`, description },
  };
}

export default async function PlayerPage({ params }: PageProps<"/players/[slug]">) {
  const { slug } = await params;
  const player = getPlayer(slug);
  if (!player) notFound();
  const credit = getPortrait(player.slug);
  const next = players[(players.indexOf(player) + 1) % players.length];

  return (
    <div data-theme={player.theme} className="theme-root min-h-screen">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(personJsonLd(player)).replace(/</g, "\\u003c") }}
      />
      <SkipLink />
      <SiteHeader current={player.slug} />
      <main id="main" tabIndex={-1} className="focus:outline-none">
        <Hero player={player} credit={credit} />
        <PageNav />
        <NetDivider />
        <Journey player={player} />
        <NetDivider />
        <Records player={player} />
        <EraContext player={player} />
        <BeyondTheCourt player={player} />
        <NetDivider />
        <LegacyAndSources player={player} credit={credit} />
        {next !== player ? <NextPlayer next={next} /> : null}
      </main>
      <SiteFooter />
    </div>
  );
}
