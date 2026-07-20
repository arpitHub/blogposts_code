/**
 * Block graphs for both architectures.
 *
 * Every block: { id, type, label, description, position: [x, y, z], color,
 *               section, layerIndex }
 *
 * All visuals are conceptual/illustrative — nothing here is computed from
 * real embeddings or attention weights. New architectures or richer
 * explanations can be added by editing only this file and tourScripts.js.
 */

// Color legend — keep in sync with tailwind.config.js and Legend.jsx
export const COLORS = {
  embedding: '#3b82f6', // blue — embedding / input processing
  'self-attention': '#c084fc', // light purple
  'masked-attention': '#e879f9', // magenta (causal)
  'cross-attention': '#4c1d95', // dark purple/indigo (encoder-decoder only)
  'add-norm': '#22c55e', // green — residual + normalization
  'feed-forward': '#f97316', // orange
  output: '#eab308' // gold — linear + softmax
}

// Sample sentence used by the token-flow animation (illustrative only)
export const SAMPLE_TOKENS = ['The', 'cat', 'sat', 'down']

// How many times each stack visually repeats (shown as ×N on labels)
export const STACK_REPEATS = 2

const DESCRIPTIONS = {
  'input-tokens':
    'The raw text is chopped into tokens — small pieces like words or word fragments. Each tile you see is one token from the sample sentence.',
  embedding:
    'Each token is converted into a long list of numbers called an embedding vector. Tokens with similar meanings end up as nearby points in this space.',
  positional:
    'Attention alone has no sense of word order, so a wave-like positional signal is added to every embedding. This lets the model tell "cat sat" apart from "sat cat".',
  'self-attention':
    'Every token looks at every other token and decides which ones matter for understanding it. Several attention "heads" (the colored arc groups) each learn to focus on different kinds of relationships.',
  'masked-attention':
    'Same idea as self-attention, but with a causal mask: each token may only look at itself and earlier tokens. The grayed-out future tokens show what the model is forbidden to peek at — essential for predicting the next word fairly.',
  'cross-attention':
    'Here the decoder tokens attend to the encoder\'s finished output instead of to themselves. This is how the generated sequence consults the encoded input — the heart of translation-style models.',
  'add-norm':
    'A residual "bypass beam" adds the block\'s input straight back onto its output, then layer normalization settles the values into a stable range. This keeps deep stacks trainable.',
  'feed-forward':
    'Each token vector passes independently through a small two-layer network that first expands it, applies a non-linearity, then compresses it back. This is where much of the model\'s stored knowledge is applied.',
  output:
    'A final linear layer projects each vector onto the whole vocabulary, and softmax turns those scores into probabilities. The brightest bar is the model\'s predicted next token.'
}

const LABELS = {
  'input-tokens': 'Input Tokens',
  embedding: 'Token Embedding',
  positional: 'Positional Encoding',
  'self-attention': 'Multi-Head Self-Attention',
  'masked-attention': 'Masked Self-Attention',
  'cross-attention': 'Cross-Attention',
  'add-norm': 'Add & Norm',
  'feed-forward': 'Feed-Forward Network',
  output: 'Linear + Softmax'
}

// Positional/input blocks reuse the embedding color (blue = input processing)
const typeColor = (type) =>
  COLORS[type] ?? COLORS.embedding

function block(id, type, position, { section, layerIndex = null, label } = {}) {
  return {
    id,
    type: type === 'input-tokens' || type === 'positional' ? 'embedding' : type,
    label:
      (label ?? LABELS[type]) +
      (layerIndex !== null ? `  (Layer ${layerIndex + 1})` : ''),
    description: DESCRIPTIONS[type],
    position,
    color: typeColor(
      type === 'input-tokens' || type === 'positional' ? 'embedding' : type
    ),
    section,
    layerIndex
  }
}

const SPACING = 1.9

// Builds one repeated stack of blocks rising from startY at a given x
function buildStack(prefix, x, startY, types, section) {
  const blocks = []
  let y = startY
  for (let layer = 0; layer < STACK_REPEATS; layer++) {
    for (const type of types) {
      blocks.push(
        block(`${prefix}-l${layer}-${type}-${blocks.length}`, type, [x, y, 0], {
          section,
          layerIndex: layer
        })
      )
      y += SPACING
    }
  }
  return blocks
}

function sharedPipeline(prefix, x) {
  return [
    block(`${prefix}-input-tokens`, 'input-tokens', [x, -9, 0], {
      section: 'shared',
      label: LABELS['input-tokens']
    }),
    block(`${prefix}-embedding`, 'embedding', [x, -6.6, 0], {
      section: 'shared'
    }),
    block(`${prefix}-positional`, 'positional', [x, -4.2, 0], {
      section: 'shared',
      label: LABELS.positional
    })
  ]
}

// ---------- Original Transformer (encoder-decoder) ----------

const ENCODER_X = -4.5
const DECODER_X = 4.5

const encoderStack = buildStack(
  'enc',
  ENCODER_X,
  -1,
  ['self-attention', 'add-norm', 'feed-forward', 'add-norm'],
  'encoder'
)

const decoderStack = buildStack(
  'dec',
  DECODER_X,
  -1,
  [
    'masked-attention',
    'add-norm',
    'cross-attention',
    'add-norm',
    'feed-forward',
    'add-norm'
  ],
  'decoder'
)

const encoderDecoderBlocks = [
  ...sharedPipeline('ed', 0),
  ...encoderStack,
  ...decoderStack,
  block('ed-output', 'output', [DECODER_X, decoderStack.at(-1).position[1] + 2.8, 0], {
    section: 'output'
  })
]

// ---------- GPT-style (decoder-only) ----------

const gptStack = buildStack(
  'gpt',
  0,
  -1,
  ['masked-attention', 'add-norm', 'feed-forward', 'add-norm'],
  'stack'
)

const decoderOnlyBlocks = [
  ...sharedPipeline('do', 0),
  ...gptStack,
  block('do-output', 'output', [0, gptStack.at(-1).position[1] + 2.8, 0], {
    section: 'output'
  })
]

export const ARCHITECTURES = {
  'encoder-decoder': {
    id: 'encoder-decoder',
    name: 'Original Transformer',
    subtitle: 'Encoder–Decoder ("Attention Is All You Need")',
    blocks: encoderDecoderBlocks
  },
  'decoder-only': {
    id: 'decoder-only',
    name: 'GPT-style',
    subtitle: 'Decoder-only',
    blocks: decoderOnlyBlocks
  }
}

export function getBlocks(architectureMode) {
  return ARCHITECTURES[architectureMode]?.blocks ?? []
}

export function getBlockById(architectureMode, blockId) {
  return getBlocks(architectureMode).find((b) => b.id === blockId) ?? null
}

// Human-readable "where am I in the pipeline" string for the info panel
export function pipelinePositionOf(architectureMode, blockId) {
  const blocks = getBlocks(architectureMode)
  const idx = blocks.findIndex((b) => b.id === blockId)
  if (idx === -1) return ''
  const b = blocks[idx]
  const sectionName = {
    shared: 'Shared input pipeline',
    encoder: 'Encoder stack (×N)',
    decoder: 'Decoder stack (×N)',
    stack: 'Decoder stack (×N)',
    output: 'Output head'
  }[b.section]
  return `Step ${idx + 1} of ${blocks.length} — ${sectionName}`
}

// Legend entries shown in the UI, filtered by architecture
export function legendFor(architectureMode) {
  const entries = [
    { type: 'embedding', label: 'Embedding / input', color: COLORS.embedding },
    {
      type: 'self-attention',
      label: 'Self-attention',
      color: COLORS['self-attention']
    },
    {
      type: 'masked-attention',
      label: 'Masked self-attention',
      color: COLORS['masked-attention']
    },
    {
      type: 'cross-attention',
      label: 'Cross-attention',
      color: COLORS['cross-attention']
    },
    { type: 'add-norm', label: 'Add & Norm', color: COLORS['add-norm'] },
    { type: 'feed-forward', label: 'Feed-forward', color: COLORS['feed-forward'] },
    { type: 'output', label: 'Output / softmax', color: COLORS.output }
  ]
  return architectureMode === 'decoder-only'
    ? entries.filter(
        (e) => e.type !== 'cross-attention' && e.type !== 'self-attention'
      )
    : entries
}
