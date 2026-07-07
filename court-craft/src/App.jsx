import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Landing from './pages/Landing'
import Placeholder from './pages/Placeholder'
import { MODULES } from './data/modules'

// Built module pages register themselves here; anything absent falls back to
// the Placeholder page.
import Spin from './pages/modules/Spin'
import Scoring from './pages/modules/Scoring'
import Grips from './pages/modules/Grips'
import Court from './pages/modules/Court'
import Serve from './pages/modules/Serve'
import Forehand from './pages/modules/Forehand'
import Backhand from './pages/modules/Backhand'
import Volley from './pages/modules/Volley'
import Overhead from './pages/modules/Overhead'
import Footwork from './pages/modules/Footwork'
import Positioning from './pages/modules/Positioning'
import Strategy from './pages/modules/Strategy'
import Equipment from './pages/modules/Equipment'
import Fitness from './pages/modules/Fitness'
import Roadmap from './pages/modules/Roadmap'
import Drills from './pages/modules/Drills'

const PAGES = {
  spin: Spin,
  scoring: Scoring,
  grips: Grips,
  court: Court,
  serve: Serve,
  forehand: Forehand,
  backhand: Backhand,
  volley: Volley,
  overhead: Overhead,
  footwork: Footwork,
  positioning: Positioning,
  strategy: Strategy,
  equipment: Equipment,
  fitness: Fitness,
  roadmap: Roadmap,
  drills: Drills,
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Landing />} />
        {MODULES.map((m) => {
          const Page = PAGES[m.id]
          return (
            <Route
              key={m.id}
              path={m.path}
              element={Page ? <Page /> : <Placeholder moduleId={m.id} />}
            />
          )
        })}
        <Route path="*" element={<Landing />} />
      </Route>
    </Routes>
  )
}
