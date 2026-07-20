import { Canvas } from '@react-three/fiber'
import useAppStore from '../../store/useAppStore'
import { getBlocks } from '../../data/architectureData'
import TransformerBlock from './TransformerBlock'
import AttentionArcs from './AttentionArcs'
import TokenFlow from './TokenFlow'
import CameraRig from './CameraRig'

/**
 * Canvas root: lighting, fog and background, composed with the block
 * meshes, attention arcs, token flow and the camera rig. Which blocks are
 * shown comes straight from the store's architecture mode.
 */
export default function Scene() {
  const architectureMode = useAppStore((s) => s.architectureMode)
  const clearSelection = useAppStore((s) => s.clearSelection)
  const blocks = getBlocks(architectureMode)

  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: [0, 6, 30], fov: 45, near: 0.1, far: 200 }}
      onPointerMissed={() => clearSelection()}
    >
      <color attach="background" args={['#0f172a']} />
      <fog attach="fog" args={['#0f172a', 35, 90]} />

      <ambientLight intensity={0.5} />
      <directionalLight position={[8, 14, 10]} intensity={1.1} />
      <pointLight position={[-10, 6, -6]} intensity={0.4} color="#818cf8" />

      {/* subtle ground plane for depth perception */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -11, 0]}>
        <planeGeometry args={[120, 120]} />
        <meshStandardMaterial color="#111c33" roughness={1} metalness={0} />
      </mesh>

      <group key={architectureMode}>
        {blocks.map((block) => (
          <TransformerBlock key={block.id} block={block} />
        ))}
        <AttentionArcs blocks={blocks} />
        <TokenFlow blocks={blocks} />
      </group>

      <CameraRig />
    </Canvas>
  )
}
