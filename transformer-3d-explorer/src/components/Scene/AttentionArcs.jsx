import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import { SAMPLE_TOKENS } from '../../data/architectureData'

/**
 * Stylized attention visualization: a row of small token orbs sits on top of
 * every attention block, connected by animated bezier arcs whose opacity
 * pulses to suggest (illustrative, not computed) attention strengths.
 *
 *  - self-attention: arcs between all token pairs, split into colored "heads"
 *  - masked-attention: arcs only reach backward; future tokens are grayed out
 *  - cross-attention: arcs reach across to the encoder stack's output
 */

const TOKEN_COUNT = SAMPLE_TOKENS.length
const TOKEN_SPACING = 0.65
const FOCUS_TOKEN = 2 // tokens after this index are "the future" when masked

// Slight hue variations of the block color = the different heads
function headColors(baseHex, heads = 2) {
  const base = new THREE.Color(baseHex)
  return Array.from({ length: heads }, (_, i) => {
    const c = base.clone()
    const hsl = {}
    c.getHSL(hsl)
    c.setHSL((hsl.h + i * 0.06) % 1, hsl.s, Math.min(hsl.l + i * 0.08, 0.85))
    return c
  })
}

function tokenPositions(blockPos) {
  const [bx, by, bz] = blockPos
  const startX = bx - ((TOKEN_COUNT - 1) * TOKEN_SPACING) / 2
  return Array.from({ length: TOKEN_COUNT }, (_, i) => [
    startX + i * TOKEN_SPACING,
    by + 0.65,
    bz
  ])
}

function arcPoints(from, to, height = 0.7) {
  const mid = [
    (from[0] + to[0]) / 2,
    Math.max(from[1], to[1]) + height,
    (from[2] + to[2]) / 2
  ]
  const curve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(...from),
    new THREE.Vector3(...mid),
    new THREE.Vector3(...to)
  )
  return curve.getPoints(20)
}

function PulsingArc({ points, color, phase, lineWidth = 1.5 }) {
  const ref = useRef()
  useFrame(({ clock }) => {
    if (ref.current?.material) {
      ref.current.material.opacity =
        0.25 + 0.45 * (0.5 + 0.5 * Math.sin(clock.elapsedTime * 2 + phase))
    }
  })
  return (
    <Line
      ref={ref}
      points={points}
      color={color}
      lineWidth={lineWidth}
      transparent
      opacity={0.4}
    />
  )
}

function ArcSet({ block, crossTarget }) {
  const isMasked = block.type === 'masked-attention'
  const isCross = block.type === 'cross-attention'
  const tokens = useMemo(() => tokenPositions(block.position), [block.position])
  const colors = useMemo(() => headColors(block.color, isCross ? 1 : 2), [
    block.color,
    isCross
  ])

  const arcs = useMemo(() => {
    const list = []
    if (isCross && crossTarget) {
      // decoder tokens → encoder output tokens
      const targets = tokenPositions(crossTarget)
      tokens.forEach((from, i) => {
        targets.forEach((to, j) => {
          if (Math.abs(i - j) <= 1) {
            list.push({
              points: arcPoints(from, to, 1.6),
              color: colors[0],
              phase: i + j
            })
          }
        })
      })
      return list
    }
    tokens.forEach((from, i) => {
      tokens.forEach((to, j) => {
        if (i === j) return
        if (isMasked && j > i) return // never attend to the future
        if (isMasked && i > FOCUS_TOKEN) return // future tokens are inert
        colors.forEach((color, h) => {
          list.push({
            points: arcPoints(from, to, 0.5 + h * 0.35),
            color,
            phase: i * 2 + j + h * 3
          })
        })
      })
    })
    return list
  }, [tokens, colors, isMasked, isCross, crossTarget])

  return (
    <group>
      {/* token orbs riding on top of the block */}
      {tokens.map((p, i) => {
        const isFuture = isMasked && i > FOCUS_TOKEN
        return (
          <mesh key={i} position={p}>
            <sphereGeometry args={[0.13, 16, 16]} />
            <meshStandardMaterial
              color={isFuture ? '#475569' : block.color}
              emissive={isFuture ? '#334155' : block.color}
              emissiveIntensity={isFuture ? 0.1 : 0.8}
              transparent
              opacity={isFuture ? 0.45 : 1}
            />
          </mesh>
        )
      })}
      {arcs.map((a, i) => (
        <PulsingArc key={i} {...a} />
      ))}
    </group>
  )
}

/**
 * Renders arc sets for every attention block in the active architecture.
 * `blocks` is the full block list; cross-attention arcs target the top of
 * the encoder stack when one exists.
 */
export default function AttentionArcs({ blocks }) {
  const encoderTop = useMemo(() => {
    const enc = blocks.filter((b) => b.section === 'encoder')
    return enc.length ? enc[enc.length - 1].position : null
  }, [blocks])

  const attentionBlocks = blocks.filter((b) =>
    ['self-attention', 'masked-attention', 'cross-attention'].includes(b.type)
  )

  return (
    <group>
      {attentionBlocks.map((b) => (
        <ArcSet key={b.id} block={b} crossTarget={encoderTop} />
      ))}
    </group>
  )
}
