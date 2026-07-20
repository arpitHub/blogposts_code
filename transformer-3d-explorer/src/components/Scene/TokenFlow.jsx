import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { SAMPLE_TOKENS } from '../../data/architectureData'

/**
 * Animated token flow: labeled tiles at the input, glowing orbs that travel
 * along the pipeline path(s), and a sinusoidal ripple ring at the positional
 * encoding block. Purely illustrative motion.
 */

const FLOW_Z = 1.1 // orbs travel slightly in front of the blocks
const FLOW_SPEED = 0.035

function pathThrough(blocks) {
  const pts = blocks.map(
    (b) => new THREE.Vector3(b.position[0], b.position[1], FLOW_Z)
  )
  return pts.length >= 2 ? new THREE.CatmullRomCurve3(pts) : null
}

function FlowOrbs({ curve, color = '#93c5fd' }) {
  const refs = useRef([])
  useFrame(({ clock }) => {
    refs.current.forEach((mesh, i) => {
      if (!mesh || !curve) return
      const t =
        (clock.elapsedTime * FLOW_SPEED + i / SAMPLE_TOKENS.length) % 1
      const p = curve.getPointAt(t)
      mesh.position.copy(p)
      // fade in at the start and out at the end of the path
      const fade = Math.min(1, Math.min(t, 1 - t) * 8)
      mesh.material.opacity = 0.25 + 0.75 * fade
    })
  })
  if (!curve) return null
  return (
    <group>
      {SAMPLE_TOKENS.map((_, i) => (
        <mesh key={i} ref={(el) => (refs.current[i] = el)}>
          <sphereGeometry args={[0.16, 16, 16]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={1.4}
            transparent
          />
        </mesh>
      ))}
    </group>
  )
}

function PositionalRipple({ position }) {
  const ref = useRef()
  useFrame(({ clock }) => {
    if (!ref.current) return
    const t = (clock.elapsedTime % 2) / 2
    const s = 0.6 + t * 1.6
    ref.current.scale.setScalar(s)
    ref.current.material.opacity = 0.6 * (1 - t)
  })
  return (
    <mesh ref={ref} position={position} rotation={[Math.PI / 2, 0, 0]}>
      <torusGeometry args={[0.8, 0.03, 8, 48]} />
      <meshBasicMaterial color="#60a5fa" transparent opacity={0.5} />
    </mesh>
  )
}

export default function TokenFlow({ blocks }) {
  const shared = useMemo(
    () =>
      blocks
        .filter((b) => b.section === 'shared')
        .sort((a, b) => a.position[1] - b.position[1]),
    [blocks]
  )

  const paths = useMemo(() => {
    const bySection = (name) =>
      blocks
        .filter((b) => b.section === name)
        .sort((a, b) => a.position[1] - b.position[1])
    const output = blocks.filter((b) => b.section === 'output')
    const encoder = bySection('encoder')
    const decoder = bySection('decoder')
    const stack = bySection('stack')

    const result = []
    if (encoder.length) {
      // encoder branch ends at the encoder top; decoder branch reaches output
      result.push({ curve: pathThrough([...shared, ...encoder]), color: '#93c5fd' })
      result.push({
        curve: pathThrough([...shared, ...decoder, ...output]),
        color: '#f0abfc'
      })
    } else {
      result.push({
        curve: pathThrough([...shared, ...stack, ...output]),
        color: '#93c5fd'
      })
    }
    return result.filter((p) => p.curve)
  }, [blocks, shared])

  const inputBlock = shared[0]
  const positionalBlock = shared[2]

  return (
    <group>
      {paths.map((p, i) => (
        <FlowOrbs key={i} curve={p.curve} color={p.color} />
      ))}

      {/* token tiles at the input */}
      {inputBlock && (
        <Html
          center
          position={[
            inputBlock.position[0],
            inputBlock.position[1] - 1.2,
            0
          ]}
          distanceFactor={14}
          style={{ pointerEvents: 'none' }}
        >
          <div className="flex gap-1">
            {SAMPLE_TOKENS.map((tok, i) => (
              <span
                key={i}
                className="rounded bg-blue-500/80 px-1.5 py-0.5 text-[10px] font-bold text-white shadow"
              >
                {tok}
              </span>
            ))}
          </div>
        </Html>
      )}

      {positionalBlock && (
        <PositionalRipple position={positionalBlock.position} />
      )}
    </group>
  )
}
