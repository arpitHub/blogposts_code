import { getBlocks } from './architectureData'

/**
 * Guided-tour scripts, one ordered array per architecture.
 * Each step: { blockId, cameraPosition: [x,y,z], cameraTarget: [x,y,z], narration }
 *
 * Steps are generated from the block graph so camera framing always matches
 * block positions; narration is authored per step below. Adding a language
 * or a new architecture means editing only this file + architectureData.js.
 */

// Camera sits in front of and slightly above the focused block
function cameraFor(block, { dist = 7, lift = 1.5, side = 0 } = {}) {
  const [x, y, z] = block.position
  return {
    cameraPosition: [x + side, y + lift, z + dist],
    cameraTarget: [x, y, z]
  }
}

const NARRATION = {
  'input-tokens':
    'Welcome! Everything starts with text. The sentence is split into tokens — the small tiles you see here. The model never sees letters, only these token pieces.',
  embedding:
    'Each token tile now becomes a glowing sphere: an embedding vector. Think of it as the token\'s coordinates in a huge "meaning space" — similar words live near each other.',
  positional:
    'Watch the ripple: a sinusoidal positional signal is blended into every sphere. Without it, the model could not tell what order the words came in.',
  'self-attention':
    'This is the famous part. Arcs connect every token to every other token — each colored group of arcs is one attention head, looking for a different kind of relationship. Thicker, brighter arcs mean "pay more attention here".',
  'masked-attention':
    'Attention again, but masked: arcs only reach backward. The grayed-out tokens on the right are the future — the model must predict the next token without peeking at it.',
  'cross-attention':
    'Now the decoder consults the encoder. These darker indigo arcs reach across from the decoder tokens to the encoder\'s finished output — this is how the output side reads the input side.',
  'add-norm':
    'See the bypass beam? The block\'s input skips over and is added back to its output (a residual connection), then everything is normalized to keep the numbers well-behaved.',
  'feed-forward':
    'Each sphere now flies through this funnel on its own: expand, activate, compress. No token-to-token talk here — just individual processing that applies the model\'s learned knowledge.',
  output:
    'The finale: vectors funnel onto the vocabulary plane, where a linear layer and softmax light up a probability for every possible next token. The brightest column is the model\'s prediction.',
  'stack-repeat':
    'One layer is done — and the whole stack simply repeats, ×N times. Each pass refines the representations further. Real models stack dozens of these layers.'
}

function narrationFor(block) {
  // Shared-pipeline blocks carry their original flavor via id suffix
  if (block.id.endsWith('input-tokens')) return NARRATION['input-tokens']
  if (block.id.endsWith('positional')) return NARRATION.positional
  return NARRATION[block.type]
}

function buildTour(architectureMode) {
  const blocks = getBlocks(architectureMode)
  const steps = []
  const seenSections = new Set()

  for (const b of blocks) {
    // Narrate the shared pipeline, every layer-0 block, and the output.
    // Layer 1+ collapses into a single "the stack repeats" step.
    if (b.layerIndex !== null && b.layerIndex > 0) {
      const key = `${b.section}-repeat`
      if (!seenSections.has(key)) {
        seenSections.add(key)
        steps.push({
          blockId: b.id,
          ...cameraFor(b, { dist: 11, lift: 2.5 }),
          narration: NARRATION['stack-repeat']
        })
      }
      continue
    }

    const isCross = b.type === 'cross-attention'
    steps.push({
      blockId: b.id,
      // Pull back a bit on cross-attention so both stacks are in frame
      ...cameraFor(b, isCross ? { dist: 12, side: -3 } : {}),
      narration: narrationFor(b)
    })
  }
  return steps
}

export const TOUR_SCRIPTS = {
  'encoder-decoder': buildTour('encoder-decoder'),
  'decoder-only': buildTour('decoder-only')
}

export function getTourScript(architectureMode) {
  return TOUR_SCRIPTS[architectureMode] ?? []
}

// Seconds each step stays on screen while the tour is playing
export const TOUR_STEP_SECONDS = 9
