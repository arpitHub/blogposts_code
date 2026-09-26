import Image from "next/image";
import type { ImageCredit, Player } from "@/content/types";
import { portraitAlt } from "@/lib/format";

/**
 * The player's photo from Wikimedia Commons, or, if no freely licensed photo has been fetched, a
 * typographic portrait. Both fill the same 4:5 box, so the layout never shifts.
 */
export function Portrait({
  player,
  credit,
  priority = false,
  sizes = "(min-width: 1024px) 420px, 90vw",
}: {
  player: Player;
  credit?: ImageCredit;
  priority?: boolean;
  sizes?: string;
}) {
  if (credit) {
    return (
      <Image
        src={credit.src}
        width={credit.width}
        height={credit.height}
        alt={portraitAlt(player, credit)}
        sizes={sizes}
        priority={priority}
        className="block aspect-[4/5] h-auto w-full object-cover"
        style={{ filter: "var(--photo-filter)", objectPosition: player.portrait.focus ?? "50% 20%" }}
      />
    );
  }
  return (
    <div
      role="img"
      aria-label={`${player.name}: typographic portrait`}
      className="relative flex aspect-[4/5] w-full flex-col items-center justify-center overflow-hidden bg-paper-2 text-ink [container-type:inline-size]"
    >
      <span aria-hidden="true" className="display select-none text-[42cqw] leading-none">
        {player.portrait.monogram}
      </span>
      <span aria-hidden="true" className="board-text mt-[4cqw] text-[max(0.75rem,5cqw)] uppercase text-muted">
        {player.lifespan}
      </span>
      {/* a ball, for the tennis-minded */}
      <span aria-hidden="true" className="absolute right-[16%] top-[10%] h-[7cqw] w-[7cqw] min-h-3 min-w-3 rounded-full bg-[#d9e24a] shadow-[inset_-3px_-3px_0_rgb(0_0_0/0.15)]" />
    </div>
  );
}

/** One line of credit under the portrait and in the page footer. */
export function PortraitCredit({ player, credit }: { player: Player; credit?: ImageCredit }) {
  if (!credit) {
    return (
      <p className="text-xs text-muted">
        Portrait: typographic illustration. No freely licensed photo of {player.name} from Wikimedia Commons has been
        added yet.
      </p>
    );
  }
  return (
    <p className="text-xs text-muted">
      Photo:{" "}
      <a href={credit.sourceUrl} className="link" target="_blank" rel="noopener noreferrer">
        {credit.title}
      </a>{" "}
      by {credit.author}, via Wikimedia Commons,{" "}
      {credit.licenseUrl ? (
        <a href={credit.licenseUrl} className="link" target="_blank" rel="noopener noreferrer">
          {credit.license}
        </a>
      ) : (
        credit.license
      )}
      . Cropped and toned for this page.
    </p>
  );
}
