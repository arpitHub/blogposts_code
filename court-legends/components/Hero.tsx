import type { ImageCredit, Player } from "@/content/types";
import { CourtFrame } from "./CourtLines";
import { Portrait, PortraitCredit } from "./Portrait";

export function Hero({ player, credit }: { player: Player; credit?: ImageCredit }) {
  return (
    <section aria-labelledby="player-name" className="mx-auto grid w-full max-w-6xl items-center gap-10 px-4 pb-14 pt-6 sm:px-8 lg:grid-cols-[1.25fr_1fr] lg:gap-16 lg:pb-20 lg:pt-10">
      <div>
        <p className="board-text text-sm uppercase text-accent-ink">
          {player.country} · {player.eraLabel}
        </p>
        <h1 id="player-name" className="display mt-3 text-[clamp(3.2rem,11vw,7.5rem)] leading-[0.92]">
          {player.name}
        </h1>
        <p className="board-text mt-4 text-lg uppercase text-muted">
          {player.lifespan}
          <span className="sr-only">.</span>
        </p>
        <p className="mt-6 max-w-xl text-xl leading-snug sm:text-2xl">{player.identity}</p>

        <dl className="mt-8 grid max-w-xl grid-cols-2 gap-px overflow-hidden rounded-[var(--radius)] border-2 border-board bg-board-ink/20 text-board-ink sm:grid-cols-4">
          {player.heroFacts.map((f) => (
            <div key={f.label} className="bg-board px-4 py-3">
              <dt className="board-text text-[0.7rem] uppercase text-board-ink/75">{f.label}</dt>
              <dd className="board-text mt-1 text-lg leading-tight text-board-accent">{f.value}</dd>
            </div>
          ))}
        </dl>

        <p className="mt-6 text-sm text-muted">
          Born {player.born.date}, {player.born.place}
          {player.died ? ` · Died ${player.died.date}, ${player.died.place}` : ""}
        </p>
      </div>

      <div className="mx-auto w-full max-w-[300px] sm:max-w-[420px]">
        <CourtFrame>
          <Portrait player={player} credit={credit} priority />
        </CourtFrame>
        <div className="mt-3">
          <PortraitCredit player={player} credit={credit} />
        </div>
      </div>
    </section>
  );
}
