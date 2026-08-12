export type ShotType = 'serve' | 'winner' | 'error';

export interface Shot {
  x: number;
  y: number;
  type: ShotType;
}

export interface RecentMatch {
  id: string;
  result: 'W' | 'L';
  opponent: string;
  date: string;
  duration: string;
  score: string;
}

export interface QuickAction {
  id: string;
  label: string;
  sublabel: string;
  emoji: string;
  target: '/(tabs)/record' | '/(tabs)/stats' | '/(tabs)/highlights' | '/(tabs)/index';
  highlight?: boolean;
}

export interface Highlight {
  id: string;
  title: string;
  duration: string;
  type: 'serve' | 'winner' | 'rally' | 'error';
  emoji: string;
  views: number;
  tags: string[];
}

export interface Achievement {
  id: string;
  emoji: string;
  label: string;
  earned: boolean;
}

export interface SettingRow {
  id: string;
  emoji: string;
  label: string;
  sublabel: string;
}

export interface GoalCard {
  id: string;
  label: string;
  current: number;
  target: number;
  unit?: string;
  color: 'accent' | 'blue' | 'orange' | 'purple';
}

export interface ShotDistribution {
  id: string;
  label: string;
  count: number;
  percent: number;
  color: 'accent' | 'blue' | 'orange' | 'purple';
}

export interface RallySample {
  shots: number;
  wonByPlayer: 0 | 1;
}

export interface SeasonStat {
  id: string;
  label: string;
  value: string;
}

export const HOME_SUMMARY = {
  weekLabel: 'This Week',
  shots: 847,
  goal: 1000,
  winRate: '68%',
  avgServe: '167 km/h',
  sessions: 4,
  greeting: 'SwingVision',
  title: 'Welcome back, Arpit',
};

export const QUICK_ACTIONS: QuickAction[] = [
  {
    id: 'record',
    label: 'Record Match',
    sublabel: 'Start a new session',
    emoji: '◉',
    target: '/(tabs)/record',
    highlight: true,
  },
  {
    id: 'stats',
    label: 'View Stats',
    sublabel: 'Match analytics',
    emoji: '📊',
    target: '/(tabs)/stats',
  },
  {
    id: 'highlights',
    label: 'Highlights',
    sublabel: 'Best moments',
    emoji: '🎬',
    target: '/(tabs)/highlights',
  },
  {
    id: 'line',
    label: 'Line Challenge',
    sublabel: 'Coming soon',
    emoji: '🎯',
    target: '/(tabs)/index',
  },
];

export const RECENT_MATCHES: RecentMatch[] = [
  {
    id: 'm1',
    result: 'W',
    opponent: 'Carlos M.',
    date: 'May 8',
    duration: '1h 42m',
    score: '6-4 6-3',
  },
  {
    id: 'm2',
    result: 'L',
    opponent: 'Jordan K.',
    date: 'May 5',
    duration: '2h 11m',
    score: '4-6 7-5 3-6',
  },
  {
    id: 'm3',
    result: 'W',
    opponent: 'Devon R.',
    date: 'May 2',
    duration: '1h 18m',
    score: '6-2 6-1',
  },
];

export const RECORD_SETUP = [
  { id: 'mode', label: 'Mode', value: 'Singles Match' },
  { id: 'court', label: 'Court', value: 'Hard Court' },
  { id: 'camera', label: 'Camera', value: 'Baseline View' },
  { id: 'ai', label: 'AI Line Calls', value: 'On' },
];

export const RECORD_LIVE_BASE = {
  lastServe: 189,
  rallyAvg: 6.3,
  winners: 4,
};

export const SHOT_HEATMAP: Shot[] = [
  { x: 24, y: 14, type: 'serve' },
  { x: 76, y: 12, type: 'serve' },
  { x: 18, y: 52, type: 'winner' },
  { x: 82, y: 56, type: 'winner' },
  { x: 50, y: 8, type: 'serve' },
  { x: 38, y: 60, type: 'error' },
  { x: 62, y: 60, type: 'winner' },
  { x: 50, y: 50, type: 'error' },
];

export const HEAD_TO_HEAD = {
  title: 'YOU vs CARLOS',
  rows: [
    { label: 'Winners', val1: 18, val2: 12, color1: 'accent' as const, color2: 'blue' as const },
    { label: 'Unf. Errors', val1: 9, val2: 14, color1: 'red' as const, color2: 'orange' as const },
    { label: '1st Serve %', val1: 72, val2: 64, color1: 'accent' as const, color2: 'blue' as const },
    { label: 'Break Pts Won', val1: 3, val2: 1, color1: 'accent' as const, color2: 'blue' as const },
  ],
};

export const SPEED_GAUGES = [
  { id: 's1', label: '1st Serve', speed: 189, max: 250 },
  { id: 's2', label: '2nd Serve', speed: 162, max: 250 },
  { id: 'fh', label: 'Forehand', speed: 134, max: 200 },
];

export const SHOT_DISTRIBUTION: ShotDistribution[] = [
  { id: 'fh', label: 'Forehand', count: 84, percent: 38, color: 'accent' },
  { id: 'bh', label: 'Backhand', count: 71, percent: 32, color: 'blue' },
  { id: 'sv', label: 'Serve', count: 46, percent: 21, color: 'orange' },
  { id: 'vl', label: 'Volley', count: 20, percent: 9, color: 'purple' },
];

export const SHOT_LEGEND = [
  { id: 'serve', label: 'Serve', color: 'accent' as const },
  { id: 'winner', label: 'Winner', color: 'blue' as const },
  { id: 'error', label: 'Error', color: 'red' as const },
];

export const SEASON_STATS: SeasonStat[] = [
  { id: 'wr', label: 'Win Rate', value: '68%' },
  { id: 'avg', label: 'Avg Serve', value: '171 km/h' },
  { id: 'shots', label: 'Total Shots', value: '12,847' },
  { id: 'rally', label: 'Longest Rally', value: '47 shots' },
];

export const RALLIES: RallySample[] = [
  { shots: 4, wonByPlayer: 0 },
  { shots: 9, wonByPlayer: 1 },
  { shots: 6, wonByPlayer: 0 },
  { shots: 12, wonByPlayer: 1 },
  { shots: 3, wonByPlayer: 0 },
  { shots: 7, wonByPlayer: 1 },
  { shots: 18, wonByPlayer: 0 },
  { shots: 5, wonByPlayer: 1 },
  { shots: 11, wonByPlayer: 0 },
  { shots: 8, wonByPlayer: 1 },
];

export const GOALS: GoalCard[] = [
  { id: 'g1', label: 'Hit 1,000 shots this week', current: 847, target: 1000, color: 'accent' },
  { id: 'g2', label: 'Win 5 matches this month', current: 3, target: 5, color: 'blue' },
  { id: 'g3', label: 'Improve serve to 180 km/h', current: 171, target: 180, unit: 'km/h', color: 'orange' },
  { id: 'g4', label: 'Play 20 sessions this month', current: 12, target: 20, color: 'purple' },
];

export const HIGHLIGHTS: Highlight[] = [
  { id: 'h1', title: 'Ace Down the T', duration: '0:08', type: 'serve', emoji: '🎾', views: 24, tags: ['Serve', 'Ace'] },
  { id: 'h2', title: 'Cross-Court Winner', duration: '0:12', type: 'winner', emoji: '⚡', views: 41, tags: ['Forehand', 'Winner'] },
  { id: 'h3', title: '18-Shot Epic Rally', duration: '0:34', type: 'rally', emoji: '🔥', views: 67, tags: ['Rally', 'Long'] },
  { id: 'h4', title: 'Backhand Down the Line', duration: '0:09', type: 'winner', emoji: '💥', views: 18, tags: ['Backhand', 'Winner'] },
  { id: 'h5', title: 'Smash at the Net', duration: '0:06', type: 'winner', emoji: '🎯', views: 33, tags: ['Volley', 'Winner'] },
];

export const HIGHLIGHT_FILTERS = ['All', 'Serves', 'Winners', 'Rallies', 'Errors'] as const;

export const PROFILE = {
  name: 'Arpit',
  initial: 'A',
  subtitle: 'NTRP 4.0 · Texas',
  plan: 'PRO PLAN',
};

export const ACHIEVEMENTS: Achievement[] = [
  { id: 'a1', emoji: '🏆', label: '10 Wins', earned: true },
  { id: 'a2', emoji: '⚡', label: '200km/h', earned: true },
  { id: 'a3', emoji: '🔥', label: '50 Aces', earned: true },
  { id: 'a4', emoji: '🎯', label: '1k Shots', earned: false },
  { id: 'a5', emoji: '👑', label: 'Monthly MVP', earned: false },
];

export const SUBSCRIPTION = {
  title: 'SwingVision Pro',
  description: '20 recording hours/month · AI Stats · Highlights',
  usedLabel: '18h 24m used',
  totalLabel: '20h total',
  fillPercent: 92,
};

export const SETTINGS_ROWS: SettingRow[] = [
  { id: 's1', emoji: '🎾', label: 'My Equipment', sublabel: 'Wilson Pro Staff 97' },
  { id: 's2', emoji: '📊', label: 'Coaching Insights', sublabel: 'Weekly report available' },
  { id: 's3', emoji: '☁️', label: 'Cloud Storage', sublabel: '12.4 GB of 50 GB used' },
  { id: 's4', emoji: '⌚', label: 'Apple Watch', sublabel: 'Connected' },
  { id: 's5', emoji: '⚙️', label: 'Settings', sublabel: 'Preferences and account' },
];

export const STATS_TABS = ['Match', 'Season', 'Goals'] as const;
export type StatsTab = (typeof STATS_TABS)[number];
