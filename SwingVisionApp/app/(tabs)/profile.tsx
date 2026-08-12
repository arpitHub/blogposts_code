import React from 'react';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { COLORS, FONTS, RADII } from '../../src/theme';
import { Card } from '../../src/components/ui/Card';
import { Chip } from '../../src/components/ui/Chip';
import { Icon } from '../../src/components/ui/Icon';
import {
  ACHIEVEMENTS,
  PROFILE,
  SETTINGS_ROWS,
  SUBSCRIPTION,
  type Achievement,
  type SettingRow,
} from '../../src/data/mockData';

export default function ProfileScreen(): React.ReactElement {
  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.avatarSection}>
          <LinearGradient
            colors={[COLORS.accent, COLORS.blue]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.avatar}
          >
            <Text style={styles.avatarText}>{PROFILE.initial}</Text>
          </LinearGradient>
          <Text style={styles.name}>{PROFILE.name}</Text>
          <Text style={styles.subtitle}>{PROFILE.subtitle}</Text>
          <Chip label={PROFILE.plan} variant="outline" style={{ marginTop: 10 }} />
        </View>

        <Text style={styles.sectionTitle}>Achievements</Text>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.achievementRow}
        >
          {ACHIEVEMENTS.map(a => (
            <AchievementCard key={a.id} achievement={a} />
          ))}
        </ScrollView>

        <LinearGradient
          colors={['#1a2a18', '#0a1208']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.subCard}
        >
          <Text style={styles.subTitle}>{SUBSCRIPTION.title}</Text>
          <Text style={styles.subDesc}>{SUBSCRIPTION.description}</Text>
          <View style={styles.subTrack}>
            <View style={[styles.subFill, { width: `${SUBSCRIPTION.fillPercent}%` }]} />
          </View>
          <View style={styles.subFooter}>
            <Text style={styles.subFooterText}>{SUBSCRIPTION.usedLabel}</Text>
            <Text style={styles.subFooterText}>{SUBSCRIPTION.totalLabel}</Text>
          </View>
        </LinearGradient>

        <Text style={styles.sectionTitle}>Settings</Text>
        <View style={styles.settingsList}>
          {SETTINGS_ROWS.map(s => (
            <SettingRowItem key={s.id} row={s} />
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function AchievementCard({ achievement }: { achievement: Achievement }): React.ReactElement {
  return (
    <View
      style={[
        styles.achievementCard,
        { opacity: achievement.earned ? 1 : 0.4 },
      ]}
    >
      <Text style={styles.achievementEmoji}>{achievement.emoji}</Text>
      <Text style={styles.achievementLabel}>{achievement.label}</Text>
      {achievement.earned ? (
        <View style={styles.earnedBadge}>
          <Icon name="check" size={10} color={COLORS.bg} />
        </View>
      ) : null}
    </View>
  );
}

function SettingRowItem({ row }: { row: SettingRow }): React.ReactElement {
  return (
    <Pressable>
      {({ pressed }) => (
        <Card
          style={[
            styles.settingsRow,
            pressed && { backgroundColor: COLORS.surface },
          ]}
        >
          <Text style={styles.settingEmoji}>{row.emoji}</Text>
          <View style={styles.settingTextWrap}>
            <Text style={styles.settingLabel}>{row.label}</Text>
            <Text style={styles.settingSub}>{row.sublabel}</Text>
          </View>
          <Icon name="chevron-right" size={18} color={COLORS.textMuted} />
        </Card>
      )}
    </Pressable>
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
  avatarSection: {
    alignItems: 'center',
    marginBottom: 20,
    marginTop: 6,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: COLORS.bg,
    fontWeight: '900',
    fontSize: 32,
  },
  name: {
    color: COLORS.text,
    fontSize: 22,
    fontWeight: '800',
    marginTop: 12,
  },
  subtitle: {
    color: COLORS.textMuted,
    fontSize: 13,
    marginTop: 2,
  },
  sectionTitle: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: '700',
    marginTop: 8,
    marginBottom: 12,
  },
  achievementRow: {
    paddingRight: 16,
    gap: 10,
  },
  achievementCard: {
    width: 90,
    height: 100,
    backgroundColor: COLORS.card,
    borderColor: COLORS.border,
    borderWidth: 1,
    borderRadius: RADII.md,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 10,
    position: 'relative',
  },
  achievementEmoji: {
    fontSize: 28,
  },
  achievementLabel: {
    color: COLORS.text,
    fontSize: 11,
    fontWeight: '700',
    marginTop: 8,
    textAlign: 'center',
  },
  earnedBadge: {
    position: 'absolute',
    top: 6,
    right: 6,
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: COLORS.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  subCard: {
    marginTop: 22,
    padding: 18,
    borderRadius: RADII.lg,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  subTitle: {
    color: COLORS.text,
    fontSize: 16,
    fontWeight: '800',
  },
  subDesc: {
    color: COLORS.textSub,
    fontSize: 12,
    marginTop: 4,
  },
  subTrack: {
    height: 6,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 3,
    overflow: 'hidden',
    marginTop: 14,
  },
  subFill: {
    height: '100%',
    backgroundColor: COLORS.accent,
  },
  subFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  subFooterText: {
    color: COLORS.textMuted,
    fontSize: 11,
    fontFamily: FONTS.mono,
  },
  settingsList: {
    gap: 8,
  },
  settingsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    gap: 12,
  },
  settingEmoji: {
    fontSize: 22,
  },
  settingTextWrap: {
    flex: 1,
  },
  settingLabel: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: '700',
  },
  settingSub: {
    color: COLORS.textMuted,
    fontSize: 12,
    marginTop: 2,
  },
});
