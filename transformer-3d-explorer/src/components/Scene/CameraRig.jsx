import { useEffect, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import useAppStore from '../../store/useAppStore'
import { getTourScript } from '../../data/tourScripts'

/**
 * Camera behavior for both modes:
 *  - Free Explore: OrbitControls (drag to rotate, scroll/pinch to zoom, pan)
 *  - Guided Tour: user input disabled; camera + target smoothly lerp toward
 *    the current tour step's framing.
 */
export default function CameraRig() {
  const controlsRef = useRef()
  const exploreMode = useAppStore((s) => s.exploreMode)
  const architectureMode = useAppStore((s) => s.architectureMode)
  const currentTourStep = useAppStore((s) => s.currentTourStep)

  const targetPos = useRef(new THREE.Vector3(0, 4, 26))
  const targetLook = useRef(new THREE.Vector3(0, 4, 0))

  const isTour = exploreMode === 'tour'

  useEffect(() => {
    if (!isTour) return
    const step = getTourScript(architectureMode)[currentTourStep]
    if (step) {
      targetPos.current.set(...step.cameraPosition)
      targetLook.current.set(...step.cameraTarget)
    }
  }, [isTour, architectureMode, currentTourStep])

  // Reframe the whole structure when the architecture changes in free mode
  useEffect(() => {
    if (isTour) return
    const wide =
      architectureMode === 'encoder-decoder'
        ? { pos: [0, 6, 30], look: [0, 5, 0] }
        : { pos: [0, 3, 26], look: [0, 2, 0] }
    targetPos.current.set(...wide.pos)
    targetLook.current.set(...wide.look)
  }, [isTour, architectureMode])

  useFrame(({ camera }) => {
    const controls = controlsRef.current
    if (!controls) return
    if (isTour) {
      camera.position.lerp(targetPos.current, 0.04)
      controls.target.lerp(targetLook.current, 0.04)
    }
    controls.update()
  })

  return (
    <OrbitControls
      ref={controlsRef}
      enabled={!isTour}
      enableDamping
      dampingFactor={0.08}
      minDistance={4}
      maxDistance={60}
      makeDefault
    />
  )
}
