"use client";

import { useEffect, useRef } from "react";
import type { ToolEntry } from "@/lib/types";
import { getCategoryAccent } from "@/lib/data";

interface ToolModalProps {
  entry: ToolEntry;
  onClose: () => void;
}

export default function ToolModal({ entry, onClose }: ToolModalProps) {
  const { tool, categoryId, categoryName } = entry;
  const accent = getCategoryAccent(categoryId);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeButtonRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  const website = tool.website.trim();

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="tool-modal-title"
      className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 p-0 backdrop-blur-sm sm:items-center sm:p-6"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-t-2xl border border-zinc-800 bg-zinc-900 p-6 shadow-2xl sm:rounded-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start gap-4">
          {tool.logo ? (
            // eslint-disable-next-line @next/next/no-img-element -- external logos, unknown hosts
            <img
              src={tool.logo}
              alt=""
              className="h-12 w-12 shrink-0 rounded-xl object-contain"
            />
          ) : (
            <span
              aria-hidden="true"
              className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl text-lg font-bold"
              style={{ backgroundColor: `${accent}22`, color: accent }}
            >
              {tool.name.charAt(0).toUpperCase()}
            </span>
          )}
          <div className="min-w-0 flex-1">
            <h3
              id="tool-modal-title"
              className="text-lg font-semibold text-zinc-50"
            >
              {tool.name}
            </h3>
            <span
              className="mt-1 inline-block rounded-full px-2 py-0.5 text-xs font-medium"
              style={{ backgroundColor: `${accent}22`, color: accent }}
            >
              {categoryName}
            </span>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="-m-1 rounded-lg p-1 text-zinc-500 transition-colors hover:text-zinc-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-500"
          >
            <svg
              aria-hidden="true"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="h-5 w-5"
            >
              <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
            </svg>
          </button>
        </div>

        <p className="mt-4 text-sm leading-relaxed text-zinc-300">
          {tool.description}
        </p>

        {tool.tags && tool.tags.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-1.5">
            {tool.tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full border border-zinc-700 px-2 py-0.5 text-xs text-zinc-400"
              >
                {tag}
              </span>
            ))}
          </div>
        )}

        {website && (
          <a
            href={website}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-6 inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium text-zinc-950 transition-opacity hover:opacity-90"
            style={{ backgroundColor: accent }}
          >
            Visit site
            <svg
              aria-hidden="true"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="h-3.5 w-3.5"
            >
              <path
                fillRule="evenodd"
                d="M4.25 5.5a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 0 0 .75-.75v-4a.75.75 0 0 1 1.5 0v4A2.25 2.25 0 0 1 12.75 17h-8.5A2.25 2.25 0 0 1 2 14.75v-8.5A2.25 2.25 0 0 1 4.25 4h5a.75.75 0 0 1 0 1.5h-5Z"
                clipRule="evenodd"
              />
              <path
                fillRule="evenodd"
                d="M6.194 12.753a.75.75 0 0 0 1.06.053L16.5 4.44v2.81a.75.75 0 0 0 1.5 0v-4.5a.75.75 0 0 0-.75-.75h-4.5a.75.75 0 0 0 0 1.5h2.553l-9.056 8.194a.75.75 0 0 0-.053 1.06Z"
                clipRule="evenodd"
              />
            </svg>
          </a>
        )}
      </div>
    </div>
  );
}
