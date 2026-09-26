import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { MotionProvider } from "@/components/Motion";
import { fontVariables } from "@/lib/fonts";
import { siteUrl } from "@/lib/format";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: { default: "Court Legends", template: "%s · Court Legends" },
  description: "One full-page, sourced profile per tennis great, starting with Rod Laver and Arthur Ashe.",
  openGraph: { siteName: "Court Legends", type: "website" },
  twitter: { card: "summary_large_image" },
};

export const viewport: Viewport = {
  themeColor: "#16202a",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={fontVariables} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: "document.documentElement.classList.add('js')" }} />
      </head>
      <body className="min-h-screen antialiased">
        <MotionProvider>{children}</MotionProvider>
      </body>
    </html>
  );
}
