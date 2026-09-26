import type { MetadataRoute } from "next";
import { players } from "@/content/players";
import { siteUrl } from "@/lib/format";

export default function sitemap(): MetadataRoute.Sitemap {
  return [{ url: siteUrl }, ...players.map((p) => ({ url: `${siteUrl}/players/${p.slug}` }))];
}
