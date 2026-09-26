#!/usr/bin/env node
/**
 * Downloads each player's portrait from Wikimedia Commons into public/images/players/ and writes
 * the credit (author, licence, source URL) to content/images.json.
 *
 * Only freely licensed files qualify: public domain, CC0, CC BY or CC BY-SA. Anything else,
 * including fair-use or unknown licences, is skipped. Candidates come from the player's data file
 * (`portrait.commons`): the pinned `files` first, then members of `category`. Photos dated inside
 * the theme's `era` are preferred.
 *
 *   npm run images              fetch or refresh every player (fails loudly on network errors)
 *   npm run images -- laver     just one player
 *   --soft                      never fail the build; keep whatever is already in images.json
 *   --force                     re-download even when a portrait is already recorded
 *
 * Runs automatically before `next build` in soft mode, so a Vercel build fills in any portrait that
 * hasn't been committed yet. Commit the downloaded files and images.json to make them permanent.
 */
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PLAYERS_DIR = path.join(ROOT, "content/players");
const MANIFEST = path.join(ROOT, "content/images.json");
const OUT_DIR = path.join(ROOT, "public/images/players");
const API = "https://commons.wikimedia.org/w/api.php";
const USER_AGENT = "CourtLegends/0.1 (https://github.com/arpitHub/blogposts_code; educational tennis profiles)";
const WIDTH = 900;

const args = process.argv.slice(2);
const soft = args.includes("--soft");
const force = args.includes("--force");
const only = args.filter((a) => !a.startsWith("--"));

/** Accept public domain, CC0, CC BY and CC BY-SA, in any version or jurisdiction port. */
function isFreeLicence(short = "", url = "") {
  const s = short.trim().toLowerCase();
  if (!s) return false;
  if (/(fair use|non-free|copyrighted|all rights reserved|\bnc\b|\bnd\b|noncommercial|noderiv)/.test(s)) return false;
  if (/^public domain$|^pd\b|^pd-|^cc0|^cc-zero/.test(s)) return true;
  if (/^cc[ -]by(-sa)?[ -]\d/.test(s)) return true;
  return /creativecommons\.org\/(licenses\/by(-sa)?\/|publicdomain\/)/.test(url);
}

const stripHtml = (html = "") =>
  html.replace(/<[^>]*>/g, " ").replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#0?39;/g, "'").replace(/&nbsp;/g, " ").replace(/\s+/g, " ").trim();

async function api(params) {
  const url = `${API}?${new URLSearchParams({ format: "json", formatversion: "2", origin: "*", ...params })}`;
  const res = await fetch(url, { headers: { "User-Agent": USER_AGENT } });
  if (!res.ok) throw new Error(`Commons API ${res.status} for ${url}`);
  return res.json();
}

async function categoryFiles(category) {
  const data = await api({ action: "query", list: "categorymembers", cmtitle: category, cmtype: "file", cmlimit: "100" });
  return (data.query?.categorymembers ?? []).map((m) => m.title);
}

async function imageInfo(titles) {
  const out = [];
  for (let i = 0; i < titles.length; i += 40) {
    const data = await api({
      action: "query",
      prop: "imageinfo",
      titles: titles.slice(i, i + 40).join("|"),
      iiprop: "url|size|mime|extmetadata",
      iiurlwidth: String(WIDTH),
    });
    for (const page of data.query?.pages ?? []) {
      const info = page.imageinfo?.[0];
      if (info) out.push({ title: page.title, ...info });
    }
  }
  return out;
}

function describe(info) {
  const m = info.extmetadata ?? {};
  const v = (k) => m[k]?.value ?? "";
  const date = stripHtml(v("DateTimeOriginal")) || stripHtml(v("DateTime"));
  const year = Number((date.match(/\b(18|19|20)\d{2}\b/) ?? [])[0]) || undefined;
  return {
    licence: stripHtml(v("LicenseShortName")),
    licenceUrl: stripHtml(v("LicenseUrl")) || undefined,
    author: stripHtml(v("Artist")) || stripHtml(v("Credit")) || "Unknown author",
    description: stripHtml(v("ImageDescription")).slice(0, 300) || undefined,
    nonFree: /true/i.test(v("NonFree")),
    date: date || undefined,
    year,
  };
}

async function pickPortrait(player) {
  const { files = [], category, exclude, era } = player.portrait.commons;
  const pinned = [...files];
  let titles = [...pinned];
  if (category) {
    const excludeRe = exclude ? new RegExp(exclude, "i") : null;
    const members = (await categoryFiles(category)).filter(
      (t) => /\.(jpe?g|png)$/i.test(t) && !(excludeRe && excludeRe.test(t)) && !titles.includes(t),
    );
    titles = titles.concat(members);
  }
  if (!titles.length) return { reason: "no candidate files" };

  const infos = await imageInfo(titles);
  const scored = [];
  for (const info of infos) {
    if (!/^image\/(jpeg|png)$/.test(info.mime)) continue;
    const meta = describe(info);
    if (meta.nonFree || !isFreeLicence(meta.licence, meta.licenceUrl)) {
      console.log(`  skip (licence "${meta.licence || "unknown"}"): ${info.title}`);
      continue;
    }
    let score = 0;
    const pinIndex = pinned.indexOf(info.title);
    if (pinIndex >= 0) score += 1000 - pinIndex;
    if (meta.year && era && meta.year >= era[0] && meta.year <= era[1]) score += 50;
    if (new RegExp(player.shortName, "i").test(info.title + " " + (meta.description ?? ""))) score += 10;
    if (info.height >= info.width) score += 5;
    if (info.width >= 600) score += 2;
    scored.push({ info, meta, score });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored[0] ?? { reason: "no freely licensed candidates" };
}

async function download(url, dest) {
  const res = await fetch(url, { headers: { "User-Agent": USER_AGENT } });
  if (!res.ok) throw new Error(`Download ${res.status}: ${url}`);
  await writeFile(dest, Buffer.from(await res.arrayBuffer()));
}

async function loadPlayers() {
  const files = (await readdir(PLAYERS_DIR)).filter((f) => f.endsWith(".ts") && f !== "index.ts");
  const players = [];
  for (const f of files) {
    const mod = await import(pathToFileURL(path.join(PLAYERS_DIR, f)).href);
    players.push(mod.default);
  }
  return players.filter((p) => !only.length || only.includes(p.slug));
}

async function main() {
  const manifest = JSON.parse(await readFile(MANIFEST, "utf8").catch(() => "{}"));
  await mkdir(OUT_DIR, { recursive: true });
  const players = await loadPlayers();
  let failures = 0;

  for (const player of players) {
    if (manifest[player.slug] && !force) {
      console.log(`${player.slug}: already have ${manifest[player.slug].title} (use --force to refresh)`);
      continue;
    }
    console.log(`${player.slug}: searching Wikimedia Commons…`);
    try {
      const pick = await pickPortrait(player);
      if (!pick.info) {
        console.log(`${player.slug}: ${pick.reason}; the page will use its typographic portrait.`);
        continue;
      }
      const { info, meta } = pick;
      const ext = /png$/i.test(info.mime) ? "png" : "jpg";
      const file = `${player.slug}.${ext}`;
      await download(info.thumburl ?? info.url, path.join(OUT_DIR, file));
      manifest[player.slug] = {
        src: `/images/players/${file}`,
        width: info.thumbwidth ?? info.width,
        height: info.thumbheight ?? info.height,
        title: info.title.replace(/^File:/, ""),
        author: meta.author,
        license: meta.licence,
        licenseUrl: meta.licenceUrl,
        sourceUrl: info.descriptionurl,
        date: meta.date,
        description: meta.description,
      };
      console.log(`${player.slug}: saved ${file} from ${info.title} (${meta.licence}, ${meta.author})`);
    } catch (err) {
      failures++;
      console.warn(`${player.slug}: could not reach Wikimedia Commons (${err.message})`);
    }
  }

  await writeFile(MANIFEST, JSON.stringify(manifest, null, 2) + "\n");
  if (failures && !soft) process.exit(1);
}

main().catch((err) => {
  console.warn(`fetch-images: ${err.message}`);
  process.exit(soft ? 0 : 1);
});
