import React, { useMemo, useState } from 'react';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import { COLORS, FONTS, RADII } from '../../src/theme';
import { Chip } from '../../src/components/ui/Chip';
import { Card } from '../../src/components/ui/Card';
import { Icon } from '../../src/components/ui/Icon';
import {
  HIGHLIGHTS,
  HIGHLIGHT_FILTERS,
  type Highlight,
} from '../../src/data/mockData';

type Filter = (typeof HIGHLIGHT_FILTERS)[number];

const TYPE_TO_FILTER: Record<Highlight['type'], Filter> = {
  serve: 'Serves',
  winner: 'Winners',
  rally: 'Rallies',
  error: 'Errors',
};

const TYPE_COLOR: Record<Highlight['type'], string> = {
  serve: COLORS.accent,
  winner: COLORS.blue,
  rally: COLORS.orange,
  error: COLORS.red,
};

export default function HighlightsScreen(): React.ReactElement {
  const [filter, setFilter] = useState<Filter>('All');

  const visible = useMemo(
    () =>
      filter === 'All'
        ? HIGHLIGHTS
        : HIGHLIGHTS.filter(h => TYPE_TO_FILTER[h.type] === filter),
    [filter],
  );

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>Clips</Text>
        <Text style={styles.subtitle}>Your best moments</Text>
      </View>

      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.filterRow}
      >
        {HIGHLIGHT_FILTERS.map(f => (
          <Chip
            key={f}
            label={f}
            active={filter === f}
            onPress={() => setFilter(f)}
            style={{ marginRight: 8 }}
          />
        ))}
      </ScrollView>

      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        {visible.map(h => (
          <HighlightCard key={h.id} highlight={h} />
        ))}
        {visible.length === 0 ? (
          <Card style={{ marginTop: 12 }}>
            <Text style={styles.emptyText}>No clips match this filter yet.</Text>
          </Card>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

function HighlightCard({ highlight }: { highlight: Highlight }): React.ReactElement {
  const [expanded, setExpanded] = useState(false);
  const height = useSharedValue(80);
  const progress = useSharedValue(0);
  const accentColor = TYPE_COLOR[highlight.type];

  const togglePress = (): void => {
    const next = !expanded;
    setExpanded(next);
    height.value = withTiming(next ? 140 : 80, { duration: 280 });
    progress.value = withTiming(next ? 0.62 : 0, { duration: 600 });
  };

  const thumbStyle = useAnimatedStyle(() => ({ height: height.value }));
  const progressStyle = useAnimatedStyle(() => ({
    width: `${progress.value * 100}%`,
  }));

  return (
    <Pressable onPress={togglePress} style={styles.cardWrap}>
      <Card padding={0} style={styles.cardOuter}>
        <Animated.View style={[styles.thumb, thumbStyle]}>
          <LinearGradient
            colors={['#1a1a28', '#0a0a0f']}
            style={StyleSheet.absoluteFill}
          />
          <Text style={styles.emoji}>{highlight.emoji}</Text>
          <View style={[styles.typeBadge, { borderColor: accentColor }]}>
            <Text style={[styles.typeBadgeText, { color: accentColor }]}>
              {highlight.type.toUpperCase()}
            </Text>
          </View>
          <View style={styles.durationBadge}>
            <Text style={styles.durationText}>{highlight.duration}</Text>
          </View>
          {expanded ? (
            <View style={styles.progressTrack}>
              <Animated.View
                style={[styles.progressFill, { backgroundColor: accentColor }, progressStyle]}
              />
            </View>
          ) : null}
        </Animated.View>
        <View style={styles.meta}>
          <View style={styles.metaLeft}>
            <Text style={styles.cardTitle}>{highlight.title}</Text>
            <View style={styles.tagRow}>
              {highlight.tags.map(t => (
                <View key={t} style={styles.tag}>
                  <Text style={styles.tagText}>{t}</Text>
                </View>
              ))}
            </View>
          </View>
          <View style={styles.metaRight}>
            <View style={[styles.playBtn, { borderColor: accentColor }]}>
              <Icon name="play-circle" size={16} color={accentColor} />
            </View>
            <Text style={styles.views}>{highlight.views} views</Text>
          </View>
        </View>
      </Card>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: COLORS.bg,
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
  filterRow: {
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  scroll: {
    paddingHorizontal: 16,
    paddingBottom: 100,
  },
  cardWrap: {
    marginBottom: 12,
  },
  cardOuter: {
    overflow: 'hidden',
  },
  thumb: {
    width: '100%',
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
  },
  emoji: {
    fontSize: 38,
  },
  typeBadge: {
    position: 'absolute',
    top: 10,
    left: 10,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderWidth: 1,
    borderRadius: 999,
    backgroundColor: 'rgba(10,10,15,0.6)',
  },
  typeBadgeText: {
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 1,
  },
  durationBadge: {
    position: 'absolute',
    bottom: 10,
    right: 10,
    backgroundColor: 'rgba(10,10,15,0.7)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  durationText: {
    color: COLORS.text,
    fontFamily: FONTS.mono,
    fontSize: 11,
  },
  progressTrack: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 3,
    backgroundColor: 'rgba(255,255,255,0.15)',
  },
  progressFill: {
    height: '100%',
  },
  meta: {
    flexDirection: 'row',
    padding: 12,
    alignItems: 'center',
  },
  metaLeft: {
    flex: 1,
  },
  cardTitle: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: '700',
  },
  tagRow: {
    flexDirection: 'row',
    gap: 6,
    marginTop: 6,
  },
  tag: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: RADII.pill,
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  tagText: {
    color: COLORS.textSub,
    fontSize: 10,
    fontWeight: '600',
  },
  metaRight: {
    alignItems: 'center',
    gap: 4,
  },
  playBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  views: {
    color: COLORS.textMuted,
    fontSize: 10,
  },
  emptyText: {
    color: COLORS.textMuted,
    fontSize: 13,
    textAlign: 'center',
  },
});
