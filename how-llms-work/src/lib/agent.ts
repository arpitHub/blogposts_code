// Scripted agent run for the Agents section: one query, three tool-using
// iterations (search → calculator → code executor), then a final answer.
// Everything is mocked, but the loop structure is the real ReAct pattern.

export type ToolName = 'search' | 'calculator' | 'code'

export const TOOLS: Record<ToolName, { label: string; icon: string }> = {
  search: { label: 'search', icon: '🔎' },
  calculator: { label: 'calculator', icon: '🧮' },
  code: { label: 'code executor', icon: '💻' },
}

export interface AgentStep {
  plan: string // beginner-friendly plan text
  thought: string // ReAct trace: Thought
  tool: ToolName
  action: string // ReAct trace: Action
  observation: string // ReAct trace: Observation
  decide: string // why it loops again (or stops)
  loops: boolean // true → back to plan, false → answer
}

export const AGENT_QUERY =
  'What is the combined population of the three Nordic capitals, rounded to the nearest 100,000?'

export const AGENT_STEPS: AgentStep[] = [
  {
    plan: 'I don’t know these numbers offhand — better look them up.',
    thought:
      'I need current population figures for Stockholm, Oslo and Copenhagen. I should search rather than rely on memory.',
    tool: 'search',
    action: 'search("population Stockholm Oslo Copenhagen city")',
    observation: 'Stockholm ≈ 984,000 · Oslo ≈ 717,000 · Copenhagen ≈ 660,000',
    decide: 'Got the numbers — but I still need to add them. Loop again.',
    loops: true,
  },
  {
    plan: 'Now add the three numbers — math is safer in a calculator than in my head.',
    thought:
      'LLMs are unreliable at arithmetic; delegate the addition to the calculator tool.',
    tool: 'calculator',
    action: 'calculator("984000 + 717000 + 660000")',
    observation: '2,361,000',
    decide: 'Have the sum, but the question asks for rounding. One more step.',
    loops: true,
  },
  {
    plan: 'Round it to the nearest 100,000 — quick bit of code makes it exact.',
    thought:
      'Rounding to the nearest 100k is round(x, -5). Run it to be certain rather than eyeballing.',
    tool: 'code',
    action: 'python: round(2_361_000, -5)',
    observation: '2,400,000',
    decide: 'That fully answers the question. Stop looping and reply.',
    loops: false,
  },
]

export const AGENT_ANSWER =
  'About 2.4 million people — Stockholm (984k) + Oslo (717k) + Copenhagen (660k) = 2,361,000, which rounds to 2,400,000.'

// One agent iteration passes through these stages in order; 'answer' is terminal.
export type Stage = 'plan' | 'act' | 'observe' | 'decide'
export const STAGES: Stage[] = ['plan', 'act', 'observe', 'decide']
