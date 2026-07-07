import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import PhaseExplorer from '../../components/widgets/PhaseExplorer'
import VariantCompare from '../../components/widgets/VariantCompare'
import { BACKHAND_1H, BACKHAND_2H, BACKHAND_DIFFERENCES } from '../../data/strokes/backhand'

export default function Backhand() {
  return (
    <div>
      <PageIntro moduleId="backhand" kicker="Stroke technique">
        <p>
          The backhand comes in two flavors — the flowing one-hander and the compact
          two-hander — and neither is “better”. They’re different tools with different
          trade-offs. This page shows you both, side by side, so you can pick yours with
          open eyes (or finally understand the one you already have).
        </p>
      </PageIntro>

      <WidgetFrame wide title="Both backhands, one slider" hint="play both — watch where the swings diverge">
        <VariantCompare
          left={{ title: 'One-handed', phases: BACKHAND_1H }}
          right={{ title: 'Two-handed', phases: BACKHAND_2H }}
        />
      </WidgetFrame>

      <Prose title="The key mechanical differences">
        <p>
          Scrub the comparison above to the <strong>Contact</strong> phase and look closely:
          the one-hander meets the ball a full step further in front with a straight arm,
          while the two-hander keeps contact closer and drives through with the second hand.
          That one difference explains almost everything else about how the two shots behave:
        </p>
      </Prose>

      <section className="mx-auto max-w-3xl px-6 py-2">
        <div className="overflow-x-auto rounded-2xl border border-line bg-white shadow-sm">
          <table className="w-full min-w-130 text-sm">
            <thead>
              <tr className="border-b border-line bg-court-50/60 text-left">
                <th className="px-4 py-3 font-semibold text-court-900"></th>
                <th className="px-4 py-3 font-semibold text-clay-700">One-handed</th>
                <th className="px-4 py-3 font-semibold text-court-700">Two-handed</th>
              </tr>
            </thead>
            <tbody>
              {BACKHAND_DIFFERENCES.map((d) => (
                <tr key={d.dim} className="border-b border-line last:border-0 align-top">
                  <td className="px-4 py-3 font-medium text-court-900">{d.dim}</td>
                  <td className="px-4 py-3 leading-relaxed text-court-700">{d.oneH}</td>
                  <td className="px-4 py-3 leading-relaxed text-court-700">{d.twoH}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <Prose>
        <Aside title="Which should a beginner choose?">
          If in doubt, start two-handed: it’s stable, forgiving, and quicker to trust in
          rallies. You can always develop the one-handed slice alongside it — every player
          needs that shot eventually, whichever backhand they drive with.
        </Aside>
        <p>
          Want to drill deeper into either motion phase by phase? Use the explorer below and
          toggle between the variants — same six checkpoints as the forehand page.
        </p>
      </Prose>

      <WidgetFrame wide title="Phase-by-phase breakdown" hint="toggle the variant, then scrub or play">
        <PhaseExplorer
          variants={[
            { id: '1h', label: 'One-handed', phases: BACKHAND_1H },
            { id: '2h', label: 'Two-handed', phases: BACKHAND_2H },
          ]}
        />
      </WidgetFrame>

      <TryThis>
        <p>
          <strong>The towel-under-the-arm drill (two-hander):</strong> tuck a towel under
          your back armpit and rally — if it drops, your arms are flying away from your body
          instead of turning with it. <strong>For one-handers:</strong> after each finish,
          freeze and check your free arm — if it’s not stretched behind you, your shoulders
          opened early. Twenty balls each, cross-court only.
        </p>
      </TryThis>

      <NextUp moduleId="backhand" />
    </div>
  )
}
