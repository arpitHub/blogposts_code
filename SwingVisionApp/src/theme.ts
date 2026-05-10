export const COLORS = {
  bg: '#0a0a0f',
  surface: '#111118',
  card: '#16161f',
  border: '#1e1e2e',
  accent: '#c8f53a',
  accentDim: '#9dba2a',
  accentGlow: 'rgba(200,245,58,0.15)',
  red: '#ff4d6d',
  blue: '#4da6ff',
  orange: '#ff8c42',
  purple: '#b57bee',
  text: '#f0f0f8',
  textMuted: '#6b6b85',
  textSub: '#9898b0',
  courtGreen: '#2d5a27',
} as const;

export const FONTS = {
  mono: 'DMMono_500Medium',
  monoRegular: 'DMMono_400Regular',
  body: 'System',
} as const;

export const RADII = {
  sm: 8,
  md: 12,
  lg: 18,
  pill: 999,
} as const;

export const SPACING = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
} as const;

export type ColorKey = keyof typeof COLORS;
