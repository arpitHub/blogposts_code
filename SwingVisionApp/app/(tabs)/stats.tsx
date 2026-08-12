import React, { useState } from 'react';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { COLORS, FONTS } from '../../src/theme';
import { Card } from '../../src/components/ui/Card';
import { CourtHeatmap } from '../../src/components/ui/CourtHeatmap';
import { SpeedGauge } from '../../src/components/ui/SpeedGauge';
import { StatBar } from '../../src/components/ui/StatBar';
import { RallyBar } from '../../src/components/ui/RallyBar';
import {
  GOALS,
  HEAD_TO_HEAD,
  RALLIES,
  SEASON_STATS,
  SHOT_DISTRIBUTION,
  SHOT_HEATMAP,
  SHOT_LEGEND,
  SPEED_GAUGES,
  STATS_TABS,
  type GoalCard,
  type ShotDistribution,
  type StatsTab,
} from '../../src/data/mockData';

const COLOR_MAP: Record<'accent' | 'blue' | 'orange' | 'purple', string> = {
  accent: COLORS.accent,
  blue: COLORS.blue,
  orange: COLORS.orange,
  purple: COLORS.purple,
};

export default function StatsScreen(): React.ReactElement {
  const [tab, setTab] = useState<StatsTab>('Match');

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>Stats</Text>
        <Text style={styles.subtitle}>Performance breakdown</Text>
      </View>

      <View style={styles.segment}>
        {STATS_TABS.map(t => {
          const active = tab === t;
          return (
            <Pressable
              key={t}
              onPress={() => setTab(t)}
              style={[styles.segmentBtn, active && styles.segmentBtnActive]}
            >
              <Text style={[styles.segmentText, active && styles.segmentTextActive]}>{t}</Text>
            </Pressable>
          );
        })}
      </View>

      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        {tab === 'Match' ? <MatchView /> : null}
        {tab === 'Season' ? <SeasonView /> : null}
        {tab === 'Goals' ? <GoalsView /> : null}
      </ScrollView>
    </SafeAreaView>
  );
}

function MatchView(): React.ReactElement {
  return (
    <View style={{ gap: 14 }}>
      <Card>
        <Text style={styles.cardTitle}>Speed</Text>
        <View style={styles.gaugeRow}>
          {SPEED_GAUGES.map(g => (
            <SpeedGauge key={g.id} speed={g.speed} label={g.label} max={g.max} />
          ))}
        </View>
      </Card>

      <Card>
        <Text style={styles.cardTitle}>{HEAD_TO_HEAD.title}</Text>
        <View style={{ marginTop: 12 }}>
          {HEAD_TO_HEAD.rows.map(r => (
            <StatBar
              key={r.label}
              label={r.label}
              val1={r.val1}
              val2={r.val2}
              color1={r.color1}
              color2={r.color2}
            />
          ))}
        </View>
      </Card>

      <Card>
        <Text style={styles.cardTitle}>Shot Placement</Text>
        <View style={{ marginTop: 12 }}>
          <CourtHeatmap shots={SHOT_HEATMAP} />
        </View>
        <View style={styles.legendRow}>
          {SHOT_LEGEND.map(l => (
            <View key={l.id} style={styles.legendItem}>
              <View style={[styles.legendDot, { backgroundColor: COLORS[l.color] }]} />
              <Text style={styles.legendText}>{l.label}</Text>
            </View>
          ))}
        </View>
      </Card>

      <Card>
        <Text style={styles.cardTitle}>Shot Distribution</Text>
        <View style={{ marginTop: 12, gap: 12 }}>
          {SHOT_DISTRIBUTION.map(d => (
            <DistributionRow key={d.id} item={d} />
          ))}
        </View>
      </Card>
    </View>
  );
}

function DistributionRow({ item }: { item: ShotDistribution }): React.ReactElement {
  const color = COLOR_MAP[item.color];
  return (
    <View>
      <View style={styles.distHeader}>
        <Text style={styles.distLabel}>{item.label}</Text>
        <Text style={styles.distMeta}>
          <Text style={[styles.distMono, { color }]}>{item.count}</Text>
          <Text style={styles.distMuted}>  shots · </Text>
          <Text style={[styles.distMono, { color }]}>{item.percent}%</Text>
        </Text>
      </View>
      <View style={styles.distTrack}>
        <View style={[styles.distFill, { width: `${item.percent}%`, backgroundColor: color }]} />
      </View>
    </View>
  );
}

function SeasonView(): React.ReactElement {
  return (
    <View style={{ gap: 14 }}>
      <View style={styles.statGrid}>
        {SEASON_STATS.map(s => (
          <Card key={s.id} style={styles.statCell}>
            <Text style={styles.statLabel}>{s.label}</Text>
            <Text style={styles.statValue}>{s.value}</Text>
          </Card>
        ))}
      </View>
      <Card>
        <Text style={styles.cardTitle}>Rally Lengths</Text>
        <Text style={styles.cardSub}>Last {RALLIES.length} rallies</Text>
        <View style={{ marginTop: 12 }}>
          <RallyBar rallies={RALLIES} />
        </View>
      </Card>
    </View>
  );
}

function GoalsView(): React.ReactElement {
  return (
    <View style={{ gap: 12 }}>
      {GOALS.map(g => (
        <GoalRow key={g.id} goal={g} />
      ))}
    </View>
  );
}

function GoalRow({ goal }: { goal: GoalCard }): React.ReactElement {
  const color = COLOR_MAP[goal.color];
  const pct = Math.min((goal.current / goal.target) * 100, 100);
  return (
    <Card>
      <View style={styles.goalHeader}>
        <Text style={styles.goalLabel}>{goal.label}</Text>
        <Text style={[styles.goalNumeric, { color }]}>
          {goal.current.toLocaleString()}/{goal.target.toLocaleString()}
          {goal.unit ? ` ${goal.unit}` : ''}
        </Text>
      </View>
      <View style={styles.goalTrack}>
        <View style={[styles.goalFill, { width: `${pct}%`, backgroundColor: color }]} />
      </View>
      <Text style={styles.goalPct}>{Math.round(pct)}% complete</Text>
    </Card>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: COLORS.bg,
  },
  scroll: {
    paddingHorizontal: 16,
    paddingBottom: 100,
  },
  headerRow: {
    paddingHorizontal: 16,
    paddingTop: 12,
  },
  title: {
    color: COLORS.text,
    fontSize: 22,
    fontWeight: '800',
  },
  subtitle: {
    color: COLORS.textMuted,
    fontSize: 13,
    marginTop: 2,
  },
  segment: {
    flexDirection: 'row',
    backgroundColor: COLORS.card,
    borderColor: COLORS.border,
    borderWidth: 1,
    borderRadius: 12,
    padding: 4,
    marginHorizontal: 16,
    marginVertical: 14,
  },
  segmentBtn: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 8,
    alignItems: 'center',
  },
  segmentBtnActive: {
    backgroundColor: COLORS.accent,
  },
  segmentText: {
    color: COLORS.textMuted,
    fontSize: 13,
    fontWeight: '700',
  },
  segmentTextActive: {
    color: COLORS.bg,
  },
  cardTitle: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: '700',
  },
  cardSub: {
    color: COLORS.textMuted,
    fontSize: 12,
    marginTop: 2,
  },
  gaugeRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 8,
  },
  legendRow: {
    flexDirection: 'row',
    gap: 14,
    justifyContent: 'center',
    marginTop: 12,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  legendText: {
    color: COLORS.textSub,
    fontSize: 11,
  },
  distHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  distLabel: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: '700',
  },
  distMeta: {
    fontSize: 12,
  },
  distMono: {
    fontFamily: FONTS.mono,
    fontSize: 12,
  },
  distMuted: {
    color: COLORS.textMuted,
    fontSize: 11,
  },
  distTrack: {
    height: 6,
    backgroundColor: COLORS.border,
    borderRadius: 3,
    overflow: 'hidden',
  },
  distFill: {
    height: '100%',
    borderRadius: 3,
  },
  statGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  statCell: {
    width: '48.5%',
  },
  statLabel: {
    color: COLORS.textMuted,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
  },
  statValue: {
    color: COLORS.accent,
    fontFamily: FONTS.mono,
    fontSize: 22,
    marginTop: 6,
  },
  goalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  goalLabel: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: '700',
    flex: 1,
    paddingRight: 8,
  },
  goalNumeric: {
    fontFamily: FONTS.mono,
    fontSize: 13,
  },
  goalTrack: {
    height: 6,
    backgroundColor: COLORS.border,
    borderRadius: 3,
    overflow: 'hidden',
  },
  goalFill: {
    height: '100%',
    borderRadius: 3,
  },
  goalPct: {
    color: COLORS.textMuted,
    fontSize: 11,
    marginTop: 6,
  },
});
