import Scene from './components/Scene/Scene'
import ControlPanel from './components/UI/ControlPanel'
import Legend from './components/UI/Legend'
import InfoPanel from './components/UI/InfoPanel'
import TourNarration from './components/UI/TourNarration'

/**
 * Top-level layout: full-screen 3D scene with UI overlays.
 */
export default function App() {
  return (
    <div className="relative h-full w-full select-none">
      <Scene />
      <ControlPanel />
      <Legend />
      <InfoPanel />
      <TourNarration />
    </div>
  )
}
