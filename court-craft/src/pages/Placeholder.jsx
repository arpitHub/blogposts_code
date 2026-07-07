import { useParams } from 'react-router-dom'
import { byId } from '../data/modules'
import { PageIntro, NextUp } from '../components/ui'

/** Temporary page for modules not yet built. */
export default function Placeholder({ moduleId }) {
  const params = useParams()
  const id = moduleId ?? params.moduleId
  const mod = byId(id)
  if (!mod) return null
  return (
    <div>
      <PageIntro moduleId={id} kicker="Coming soon">
        <p>{mod.blurb}</p>
      </PageIntro>
      <div className="mx-auto max-w-3xl px-6">
        <div className="rounded-2xl border-2 border-dashed border-court-200 bg-white/60 px-8 py-16 text-center">
          <div className="text-4xl">{mod.icon}</div>
          <p className="mt-4 font-medium text-court-700">This module is under construction.</p>
          <p className="mt-1 text-sm text-court-500">
            Its interactive widgets are on the way — check back soon.
          </p>
        </div>
      </div>
      <NextUp moduleId={id} />
    </div>
  )
}
