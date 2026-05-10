import React from 'react';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { COLORS, FONTS, RADII } from '../../src/theme';
import { Card } from '../../src/components/ui/Card';
import { Icon } from '../../src/components/ui/Icon';
import {
  HOME_SUMMARY,
  QUICK_ACTIONS,
  RECENT_MATCHES,
  type QuickAction,
  type RecentMatch,
} from '../../src/data/mockData';

export default function HomeScreen(): React.ReactElement {
  const router = useRouter();
  const progress = Math.min((HOME_SUMMARY.shots / HOME_SUMMARY.goal) * 100, 100);

  const handleAction = (action: QuickAction): void => {
    router.push(action.target);
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.header}>
          <View>
            <Text style={styles.greeting}>{HOME_SUMMARY.greeting}</Text>
            <Text style={styles.title}>{HOME_SUMMARY.title}</Text>
          </View>
          <View style={styles.headerRight}>
            <Pressable style={styles.iconBtn}>
              <Icon name="bell" size={20} color={COLORS.text} />
            </Pressable>
            <LinearGradient
              colors={[COLORS.accent, COLORS.blue]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={styles.avatar}
            >
              <Text style={styles.avatarText}>A</Text>
            </LinearGradient>
          </View>
        </View>

        <LinearGradient
          colors={['#1a1a28', '#0f1f0a']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.hero}
        >
          <Text style={styles.heroLabel}>{HOME_SUMMARY.weekLabel}</Text>
          <Text style={styles.heroNumber}>{HOME_SUMMARY.shots}</Text>
          <Text style={styles.heroSub}>shots logged</Text>

          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: `${progress}%` }]} />
          </View>
          <Text style={styles.progressLabel}>
            {HOME_SUMMARY.shots} / {HOME_SUMMARY.goal} weekly goal
          </Text>

          <View style={styles.heroStats}>
            <HeroStat label="Win Rate" value={HOME_SUMMARY.winRate} />
            <View style={styles.divider} />
            <HeroStat label="Avg Serve" value={HOME_SUMMARY.avgServe} />
            <View style={styles.divider} />
            <HeroStat label="Sessions" value={String(HOME_SUMMARY.sessions)} />
          </View>
        </LinearGradient>

        <View style={styles.grid}>
          {QUICK_ACTIONS.map(action => (
            <Pressable
              key={action.id}
              onPress={() => handleAction(action)}
              style={({ pressed }) => [
                styles.actionWrap,
                pressed && { opacity: 0.7 },
              ]}
            >
              <Card
                style={[
                  styles.actionCard,
                  action.highlight && styles.actionHighlight,
                ]}
              >
                <Text
                  style={[
                    styles.actionEmoji,
                    action.highlight && { color: COLORS.bg },
                  ]}
                >
                  {action.emoji}
                </Text>
                <Text
                  style={[
                    styles.actionLabel,
                    action.highlight && { color: COLORS.bg },
                  ]}
                >
                  {action.label}
                </Text>
                <Text
                  style={[
                    styles.actionSub,
                    action.highlight && { color: 'rgba(10,10,15,0.7)' },
                  ]}
                >
                  {action.sublabel}
                </Text>
              </Card>
            </Pressable>
          ))}
        </View>

        <Text style={styles.sectionHeading}>Recent Matches</Text>
        {RECENT_MATCHES.map(m => (
          <MatchRow key={m.id} match={m} />
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

function HeroStat({ label, value }: { label: string; value: string }): React.ReactElement {
  return (
    <View style={styles.heroStatItem}>
      <Text style={styles.heroStatValue}>{value}</Text>
      <Text style={styles.heroStatLabel}>{label}</Text>
    </View>
  );
}

function MatchRow({ match }: { match: RecentMatch }): React.ReactElement {
  const isWin = match.result === 'W';
  return (
    <Card style={styles.matchRow}>
      <View
        style={[
          styles.resultBadge,
          { backgroundColor: isWin ? COLORS.accent : COLORS.red },
        ]}
      >
        <Text
          style={[
            styles.resultText,
            { color: isWin ? COLORS.bg : COLORS.text },
          ]}
        >
          {match.result}
        </Text>
      </View>
      <View style={styles.matchInfo}>
        <Text style={styles.matchOpponent}>{match.opponent}</Text>
        <Text style={styles.matchSub}>
          {match.date} · {match.duration}
        </Text>
      </View>
      <Text style={styles.matchScore}>{match.score}</Text>
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
    paddingTop: 12,
    paddingBottom: 100,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 18,
  },
  greeting: {
    color: COLORS.textMuted,
    fontSize: 13,
    letterSpacing: 1,
    fontWeight: '600',
  },
  title: {
    color: COLORS.text,
    fontSize: 22,
    fontWeight: '800',
    marginTop: 2,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  iconBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: COLORS.card,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: COLORS.bg,
    fontWeight: '900',
    fontSize: 16,
  },
  hero: {
    borderRadius: RADII.lg,
    padding: 20,
    borderWidth: 1,
    borderColor: COLORS.border,
    marginBottom: 18,
  },
  heroLabel: {
    color: COLORS.textMuted,
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
  heroNumber: {
    color: COLORS.accent,
    fontFamily: FONTS.mono,
    fontSize: 56,
    marginTop: 4,
  },
  heroSub: {
    color: COLORS.textSub,
    fontSize: 13,
    marginTop: -4,
  },
  progressTrack: {
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.border,
    overflow: 'hidden',
    marginTop: 14,
  },
  progressFill: {
    height: '100%',
    backgroundColor: COLORS.accent,
  },
  progressLabel: {
    color: COLORS.textSub,
    fontSize: 11,
    marginTop: 6,
  },
  heroStats: {
    flexDirection: 'row',
    marginTop: 16,
    alignItems: 'center',
  },
  heroStatItem: {
    flex: 1,
    alignItems: 'center',
  },
  heroStatValue: {
    color: COLORS.text,
    fontFamily: FONTS.mono,
    fontSize: 16,
  },
  heroStatLabel: {
    color: COLORS.textMuted,
    fontSize: 10,
    marginTop: 2,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  divider: {
    width: 1,
    height: 28,
    backgroundColor: COLORS.border,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 22,
  },
  actionWrap: {
    width: '48.5%',
  },
  actionCard: {
    minHeight: 100,
    justifyContent: 'space-between',
  },
  actionHighlight: {
    backgroundColor: COLORS.accent,
    borderColor: COLORS.accent,
  },
  actionEmoji: {
    color: COLORS.accent,
    fontSize: 24,
  },
  actionLabel: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: '700',
    marginTop: 8,
  },
  actionSub: {
    color: COLORS.textMuted,
    fontSize: 11,
    marginTop: 2,
  },
  sectionHeading: {
    color: COLORS.text,
    fontSize: 17,
    fontWeight: '700',
    marginBottom: 10,
  },
  matchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    marginBottom: 8,
    gap: 12,
  },
  resultBadge: {
    width: 32,
    height: 32,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  resultText: {
    fontWeight: '900',
    fontSize: 14,
  },
  matchInfo: {
    flex: 1,
  },
  matchOpponent: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: '700',
  },
  matchSub: {
    color: COLORS.textMuted,
    fontSize: 11,
    marginTop: 2,
  },
  matchScore: {
    color: COLORS.accent,
    fontFamily: FONTS.mono,
    fontSize: 14,
  },
});
