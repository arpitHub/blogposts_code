import { useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import Section from '../components/Section'
import { Depth, useDepth } from '../context/DepthContext'
import { tokenize, VOCAB_SIZE } from '../lib/tokenize'
import { tokenColor } from '../lib/palette'

const SAMPLES = [
  'Generative AI models are surprisingly powerful',
  'The tokenizer happily deconstructs unbelievably long words',
  'She sells seashells by the seashore',
]

export default function Tokens() {
  const [text, setText] = useState(SAMPLES[0])
  const { depth } = useDepth()
  const tokens = useMemo(() => tokenize(text), [text])

  return (
    <Section
      id="tokens"
      kicker="Step 1 · Tokens"
      title="First, your words get chopped up"
      lede={
        <Depth
          b="A model can't read letters or whole sentences. Before anything else happens, your text is snipped into bite-sized pieces called tokens. Common words survive in one piece; longer or rarer words get split at familiar seams. Try typing below — watch where the cuts land."
          t={
            <>
              Text is segmented by a subword tokenizer (BPE-style) into a fixed vocabulary of{' '}
              <code className="rounded bg-surface px-1 py-0.5 text-sm">{VOCAB_SIZE.toLocaleString()}</code>{' '}
              entries. Frequent strings become single tokens; rare words decompose into subword pieces.
              Each piece maps to an integer ID — the only thing the model ever sees. (This demo uses a
              toy morpheme splitter, not a trained BPE merge table.)
            </>
          }
        />
      }
    >
      <div className="rounded-2xl border border-hairline bg-surface p-5 sm:p-6">
        <label htmlFor="token-input" className="mb-2 block text-xs font-medium uppercase tracking-wider text-ink-3">
          Type anything
        </label>
        <input
          id="token-input"
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          spellCheck={false}
          className="w-full rounded-lg border border-hairline bg-page px-4 py-3 text-base text-ink outline-none transition-colors focus:border-tok-blue"
          placeholder="Type a sentence…"
        />
        <div className="mt-2 flex flex-wrap gap-2">
          {SAMPLES.map((s) => (
            <button
              key={s}
              onClick={() => setText(s)}
              className="rounded-full border border-hairline px-3 py-1 text-xs text-ink-3 transition-colors hover:border-tok-blue hover:text-ink-2"
            >
              “{s.length > 34 ? s.slice(0, 34) + '…' : s}”
            </button>
          ))}
        </div>

        <div className="mt-8 flex min-h-24 flex-wrap items-start gap-y-4" aria-live="polite">
          <AnimatePresence mode="popLayout">
            {tokens.map((tok, i) => (
              <motion.span
                key={`${tok.text}-${i}`}
                layout
                initial={{ opacity: 0, scale: 0.6, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.6 }}
                transition={{ type: 'spring', stiffness: 500, damping: 32 }}
                className="flex flex-col items-center"
                style={{ marginLeft: tok.pieceIndex === 0 ? 12 : 2 }}
              >
                <span
                  className="rounded-md px-2 py-1 font-mono text-sm text-white"
                  style={{
                    background: `color-mix(in srgb, ${tokenColor(tok.wordIndex)} 28%, transparent)`,
                    boxShadow: `inset 0 0 0 1px ${tokenColor(tok.wordIndex)}`,
                  }}
                >
                  {tok.text}
                </span>
                {depth === 'technical' && (
                  <span className="mt-1 font-mono text-[10px] text-ink-3">{tok.id}</span>
                )}
              </motion.span>
            ))}
          </AnimatePresence>
        </div>

        <p className="mt-6 text-sm text-ink-3">
          <span className="font-semibold text-ink-2">{tokens.length}</span>{' '}
          {tokens.length === 1 ? 'token' : 'tokens'}
          <Depth
            b=". Pieces from the same word share a color — notice how a word like “surprisingly” becomes several chips."
            t={
              <>
                {' '}· the numbers under each chip are token IDs, indices into the{' '}
                {VOCAB_SIZE.toLocaleString()}-entry vocabulary. Context windows, pricing, and “max
                tokens” all count these pieces, not words.
              </>
            }
          />
        </p>
      </div>
    </Section>
  )
}
