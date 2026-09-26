import { Fraunces, Oswald, Playfair_Display, Source_Serif_4, Work_Sans } from "next/font/google";

// All faces are exposed as CSS variables on <html>; each theme in app/themes.css picks its own
// display and body faces from them. Oswald is shared (it's the scoreboard face) so it's the only one
// preloaded; the theme faces load on demand with size-adjusted fallbacks, so there's no layout shift
// and no page pays for another player's fonts up front.

const playfair = Playfair_Display({ subsets: ["latin"], variable: "--font-playfair", display: "swap", preload: false });
const sourceSerif = Source_Serif_4({ subsets: ["latin"], variable: "--font-source-serif", display: "swap", preload: false });
const fraunces = Fraunces({ subsets: ["latin"], variable: "--font-fraunces", display: "swap", preload: false });
const workSans = Work_Sans({ subsets: ["latin"], variable: "--font-work-sans", display: "swap", preload: false });
const oswald = Oswald({ subsets: ["latin"], variable: "--font-oswald", display: "swap" });

export const fontVariables = [playfair, sourceSerif, fraunces, workSans, oswald].map((f) => f.variable).join(" ");
