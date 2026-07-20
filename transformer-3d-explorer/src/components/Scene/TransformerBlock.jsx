import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import useAppStore from '../../store/useAppStore'
import { getTourScript } from '../../data/tourScripts'

/**
 * Generic renderer for one architecture block. Shape is chosen from the
 * block's type; color/position/label come straight from architectureData.
 * Handles hover (tooltip) and click (info panel) in Free Explore, and
 * highlight/dim states during the Guided Tour.
 */

function geometryFor(type) {
  switch (type) {
    case 'embedding':
      return <sphereGeometry args={[0.55, 32, 32]} />
    case 'self-attention':
    case 'masked-attention':
    case 'cross-attention':
      return <boxGeometry args={[2.6, 0.8, 1.6]} />
    case 'add-norm':
      return <cylinderGeometry args={[1.1, 1.1, 0.3, 32]} />
    case 'feed-forward':
      // hourglass/funnel: narrow waist, wide mouths (expand → contract)
      return <cylinderGeometry args={[1.2, 0.5, 1.2, 24]} />
    case 'output':
      return <boxGeometry args={[3.4, 0.5, 2.2]} />
    default:
      return <boxGeometry args={[1.5, 0.8, 1.5]} />
  }
}

export default function TransformerBlock({ block }) {
  const meshRef = useRef()
  const matRef = useRef()

  const architectureMode = useAppStore((s) => s.architectureMode)
  const exploreMode = useAppStore((s) => s.exploreMode)
  const currentTourStep = useAppStore((s) => s.currentTourStep)
  const selectedBlockId = useAppStore((s) => s.selectedBlockId)
  const hoveredBlockId = useAppStore((s) => s.hoveredBlockId)
  const selectBlock = useAppStore((s) => s.selectBlock)
  const setHoveredBlock = useAppStore((s) => s.setHoveredBlock)

  const isFree = exploreMode === 'free'
  const tourBlockId = !isFree
    ? getTourScript(architectureMode)[currentTourStep]?.blockId
    : null

  const isFocused =
    block.id === tourBlockId ||
    block.id === selectedBlockId ||
    block.id === hoveredBlockId
  const isDimmed = !isFree && tourBlockId && block.id !== tourBlockId

  const baseColor = useMemo(() => new THREE.Color(block.color), [block.color])

  useFrame((state) => {
    const mat = matRef.current
    const mesh = meshRef.current
    if (!mat || !mesh) return

    // Smoothly approach target glow / dim / scale each frame
    const targetEmissive = isFocused ? 1.1 : 0.25
    const targetOpacity = isDimmed ? 0.28 : 1
    const targetScale = isFocused ? 1.12 : 1

    mat.emissiveIntensity = THREE.MathUtils.lerp(
      mat.emissiveIntensity,
      targetEmissive,
      0.08
    )
    mat.opacity = THREE.MathUtils.lerp(mat.opacity, targetOpacity, 0.08)
    const s = THREE.MathUtils.lerp(mesh.scale.x, targetScale, 0.08)
    mesh.scale.setScalar(s)

    // Gentle "settle" pulse on Add & Norm blocks
    if (block.type === 'add-norm' && !isDimmed) {
      mat.emissiveIntensity +=
        0.15 * (1 + Math.sin(state.clock.elapsedTime * 2.5)) * 0.5
    }
  })

  const handleOver = (e) => {
    if (!isFree) return
    e.stopPropagation()
    setHoveredBlock(block.id)
    document.body.style.cursor = 'pointer'
  }
  const handleOut = () => {
    if (!isFree) return
    setHoveredBlock(null)
    document.body.style.cursor = 'auto'
  }
  const handleClick = (e) => {
    if (!isFree) return
    e.stopPropagation()
    selectBlock(block.id)
  }

  return (
    <group position={block.position}>
      <mesh
        ref={meshRef}
        onPointerOver={handleOver}
        onPointerOut={handleOut}
        onClick={handleClick}
      >
        {geometryFor(block.type)}
        <meshStandardMaterial
          ref={matRef}
          color={baseColor}
          emissive={baseColor}
          emissiveIntensity={0.25}
          transparent
          opacity={1}
          roughness={0.35}
          metalness={0.2}
        />
      </mesh>

      {/* second inverted funnel half for feed-forward (expand then contract) */}
      {block.type === 'feed-forward' && (
        <mesh position={[0, -1.05, 0]} scale={[1, -1, 1]}>
          <cylinderGeometry args={[1.2, 0.5, 1.2, 24]} />
          <meshStandardMaterial
            color={baseColor}
            emissive={baseColor}
            emissiveIntensity={0.25}
            transparent
            opacity={isDimmed ? 0.28 : 0.85}
            roughness={0.35}
            metalness={0.2}
          />
        </mesh>
      )}

      {/* translucent mask sheet over the "future" side of masked attention */}
      {block.type === 'masked-attention' && (
        <mesh position={[0.85, 0.55, 0]}>
          <boxGeometry args={[1.1, 0.25, 1.7]} />
          <meshStandardMaterial
            color="#64748b"
            transparent
            opacity={isDimmed ? 0.15 : 0.45}
            roughness={0.9}
          />
        </mesh>
      )}

      {/* label — always for focused block, on hover otherwise */}
      {(isFocused || isFree) && !isDimmed && (
        <Html
          center
          position={[0, 1.15, 0]}
          distanceFactor={14}
          style={{ pointerEvents: 'none' }}
        >
          <div
            className={`whitespace-nowrap rounded px-2 py-0.5 text-[11px] font-semibold ${
              isFocused
                ? 'bg-slate-900/90 text-white ring-1 ring-white/30'
                : 'bg-slate-900/60 text-slate-300'
            }`}
          >
            {block.label}
          </div>
        </Html>
      )}
    </group>
  )
}
