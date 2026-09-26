import { ImageResponse } from "next/og";

export const ogSize = { width: 1200, height: 630 };

type OgTheme = { paper: string; ink: string; accent: string; court: string; line: string; board: string; boardAccent: string };

export const ogThemes: Record<string, OgTheme> = {
  home: { paper: "#f6f4ee", ink: "#16202a", accent: "#234d71", court: "#2c5f8a", line: "#fbfaf6", board: "#16202a", boardAccent: "#e8c547" },
  laver: { paper: "#f3eedf", ink: "#1b261d", accent: "#24482a", court: "#3a6b3f", line: "#fbfaf4", board: "#1f3a26", boardAccent: "#e2d08a" },
  ashe: { paper: "#f2e3c6", ink: "#2a1a10", accent: "#9a3b0e", court: "#2f6a6e", line: "#fdf4e1", board: "#2a1a10", boardAccent: "#f0a830" },
};

/** Shared OG card: title and tagline on the left, a court drawn from above on the right. */
export function ogImage({ theme, kicker, title, subtitle, footer }: { theme: string; kicker: string; title: string; subtitle: string; footer: string }) {
  const t = ogThemes[theme] ?? ogThemes.home;
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", background: t.paper, color: t.ink, padding: 56 }}>
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", flex: 1, paddingRight: 40 }}>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ fontSize: 26, letterSpacing: 4, textTransform: "uppercase", color: t.accent }}>{kicker}</div>
            <div style={{ fontSize: 104, fontWeight: 800, lineHeight: 1, marginTop: 18 }}>{title}</div>
            <div style={{ fontSize: 34, lineHeight: 1.25, marginTop: 26, maxWidth: 640 }}>{subtitle}</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", fontSize: 24 }}>
            <div style={{ display: "flex", background: t.board, color: t.boardAccent, padding: "8px 16px", letterSpacing: 3, textTransform: "uppercase" }}>
              Court Legends
            </div>
            <div style={{ marginLeft: 18, color: t.accent }}>{footer}</div>
          </div>
        </div>
        <div style={{ display: "flex", width: 330, background: t.court, padding: 18 }}>
          <div style={{ display: "flex", flex: 1, border: `5px solid ${t.line}`, position: "relative" }}>
            <div style={{ position: "absolute", top: 0, bottom: 0, left: 26, width: 3, background: t.line }} />
            <div style={{ position: "absolute", top: 0, bottom: 0, right: 26, width: 3, background: t.line }} />
            <div style={{ position: "absolute", left: 0, right: 0, top: "50%", height: 8, background: t.line }} />
            <div style={{ position: "absolute", left: 26, right: 26, top: "27%", height: 3, background: t.line }} />
            <div style={{ position: "absolute", left: 26, right: 26, top: "73%", height: 3, background: t.line }} />
            <div style={{ position: "absolute", left: "50%", top: "27%", height: "46%", width: 3, background: t.line }} />
            <div style={{ position: "absolute", left: 180, top: 120, width: 34, height: 34, borderRadius: 34, background: "#d9e24a" }} />
          </div>
        </div>
      </div>
    ),
    ogSize,
  );
}
