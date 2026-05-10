import React, { useEffect, useRef, useState } from 'react';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import { COLORS, FONTS, RADII } from '../../src/theme';
import { Card } from '../../src/components/ui/Card';
import { CourtHeatmap } from '../../src/components/ui/CourtHeatmap';
import { LiveBadge } from '../../src/components/ui/LiveBadge';
import { Icon } from '../../src/components/ui/Icon';
import {
  RECORD_LIVE_BASE,
  RECORD_SETUP,
  SHOT_HEATMAP,
} from '../../src/data/mockData';

type RecordState = 'idle' | 'recording' | 'stopped';

interface LiveStats {
  lastServe: number;
  rallyAvg: number;
  winners: number;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0');
  const s = (seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function jitter(base: number, range: number): number {
  return base + (Math.random() * 2 - 1) * range;
}

export default function RecordScreen(): React.ReactElement {
  const router = useRouter();
  const [permission, requestPermission] = useCameraPermissions();
  const [state, setState] = useState<RecordState>('idle');
  const [elapsed, setElapsed] = useState(0);
  const [liveStats, setLiveStats] = useState<LiveStats>(RECORD_LIVE_BASE);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const statsRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (state === 'recording') {
      timerRef.current = setInterval(() => setElapsed(s => s + 1), 1000);
      statsRef.current = setInterval(() => {
        setLiveStats({
          lastServe: Math.round(jitter(RECORD_LIVE_BASE.lastServe, 8)),
          rallyAvg: Math.round(jitter(RECORD_LIVE_BASE.rallyAvg, 0.6) * 10) / 10,
          winners: Math.max(0, RECORD_LIVE_BASE.winners + Math.round(jitter(0, 1.5))),
        });
      }, 3000);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (statsRef.current) clearInterval(statsRef.current);
    };
  }, [state]);

  const handleStart = (): void => {
    setElapsed(0);
    setLiveStats(RECORD_LIVE_BASE);
    setState('recording');
  };

  const handleStop = (): void => {
    setState('stopped');
  };

  const handleReset = (): void => {
    setState('idle');
    setElapsed(0);
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.headerRow}>
          <Text style={styles.title}>Record</Text>
          <Text style={styles.subtitle}>
            {state === 'idle'
              ? 'Set up your match'
              : state === 'recording'
                ? 'Match in progress'
                : 'Session complete'}
          </Text>
        </View>

        <View style={styles.viewfinder}>
          {permission?.granted ? (
            <CameraView
              style={StyleSheet.absoluteFillObject}
              facing="back"
              mode="video"
            />
          ) : (
            <View style={styles.permissionFallback}>
              <Text style={styles.permissionEmoji}>📷</Text>
              <Text style={styles.permissionTitle}>Camera preview</Text>
              <Text style={styles.permissionSub}>
                {permission ? 'Permission needed to record' : 'Loading camera…'}
              </Text>
              {permission && !permission.granted ? (
                <Pressable
                  style={styles.permissionBtn}
                  onPress={() => {
                    void requestPermission();
                  }}
                >
                  <Text style={styles.permissionBtnText}>Enable Camera</Text>
                </Pressable>
              ) : null}
            </View>
          )}

          {state === 'recording' ? (
            <>
              <View style={styles.viewfinderTopLeft}>
                <View style={styles.recBadge}>
                  <LiveBadge label="REC" />
                  <Text style={styles.timer}>{formatTime(elapsed)}</Text>
                </View>
              </View>
              <View style={styles.viewfinderTopRight}>
                <View style={styles.aiBadge}>
                  <View style={styles.aiDot} />
                  <Text style={styles.aiText}>AI Active</Text>
                </View>
              </View>
            </>
          ) : null}
        </View>

        {state === 'recording' ? (
          <View style={styles.liveStats}>
            <LiveStatChip label="Last Serve" value={`${liveStats.lastServe} km/h`} />
            <LiveStatChip label="Rally Avg" value={`${liveStats.rallyAvg.toFixed(1)} shots`} />
            <LiveStatChip label="Winners" value={String(liveStats.winners)} />
          </View>
        ) : null}

        {state === 'idle' ? (
          <View style={styles.setupGrid}>
            {RECORD_SETUP.map(s => (
              <Card key={s.id} style={styles.setupCard}>
                <Text style={styles.setupLabel}>{s.label}</Text>
                <Text style={styles.setupValue}>{s.value}</Text>
              </Card>
            ))}
          </View>
        ) : null}

        {state === 'stopped' ? (
          <View>
            <Card style={{ marginTop: 16 }}>
              <Text style={styles.cardTitle}>Shot Placement</Text>
              <View style={{ marginTop: 10 }}>
                <CourtHeatmap shots={SHOT_HEATMAP} />
              </View>
              <Text style={styles.completeMsg}>Session complete</Text>
            </Card>
            <View style={styles.endButtons}>
              <Pressable
                style={[styles.actionBtn, styles.actionBtnPrimary]}
                onPress={() => router.push('/(tabs)/stats')}
              >
                <Text style={styles.actionBtnPrimaryText}>View Stats</Text>
              </Pressable>
              <Pressable
                style={[styles.actionBtn, styles.actionBtnGhost]}
                onPress={handleReset}
              >
                <Text style={styles.actionBtnGhostText}>Save & Exit</Text>
              </Pressable>
            </View>
          </View>
        ) : null}

        {state !== 'stopped' ? (
          <View style={styles.controlRow}>
            {state === 'idle' ? (
              <RecordButton color={COLORS.accent} onPress={handleStart}>
                <Icon name="circle-fill" size={36} color={COLORS.bg} />
              </RecordButton>
            ) : (
              <RecordButton color={COLORS.red} onPress={handleStop}>
                <Icon name="stop-square" size={28} color={COLORS.text} />
              </RecordButton>
            )}
          </View>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

interface RecordButtonProps {
  color: string;
  onPress: () => void;
  children: React.ReactNode;
}

function RecordButton({ color, onPress, children }: RecordButtonProps): React.ReactElement {
  const scale = useSharedValue(1);
  const animStyle = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }] }));

  return (
    <Pressable
      onPressIn={() => {
        scale.value = withTiming(0.95, { duration: 80 });
      }}
      onPressOut={() => {
        scale.value = withTiming(1, { duration: 120 });
      }}
      onPress={onPress}
    >
      <Animated.View
        style={[
          styles.recordBtn,
          { borderColor: color, shadowColor: color },
          animStyle,
        ]}
      >
        <View
          style={[
            styles.recordBtnInner,
            { backgroundColor: color === COLORS.accent ? COLORS.accent : COLORS.red },
          ]}
        >
          {children}
        </View>
      </Animated.View>
    </Pressable>
  );
}

function LiveStatChip({ label, value }: { label: string; value: string }): React.ReactElement {
  return (
    <View style={styles.liveChip}>
      <Text style={styles.liveChipLabel}>{label}</Text>
      <Text style={styles.liveChipValue}>{value}</Text>
    </View>
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
  headerRow: {
    marginBottom: 14,
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
  viewfinder: {
    aspectRatio: 16 / 9,
    width: '100%',
    backgroundColor: '#000',
    borderRadius: RADII.lg,
    borderWidth: 1,
    borderColor: COLORS.border,
    overflow: 'hidden',
    position: 'relative',
  },
  permissionFallback: {
    flex: 1,
    backgroundColor: COLORS.surface,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  permissionEmoji: {
    fontSize: 36,
  },
  permissionTitle: {
    color: COLORS.text,
    fontSize: 15,
    marginTop: 8,
    fontWeight: '700',
  },
  permissionSub: {
    color: COLORS.textMuted,
    fontSize: 12,
    marginTop: 4,
  },
  permissionBtn: {
    marginTop: 12,
    backgroundColor: COLORS.accent,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 999,
  },
  permissionBtnText: {
    color: COLORS.bg,
    fontWeight: '800',
    fontSize: 13,
  },
  viewfinderTopLeft: {
    position: 'absolute',
    top: 12,
    left: 12,
  },
  viewfinderTopRight: {
    position: 'absolute',
    top: 12,
    right: 12,
  },
  recBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  timer: {
    color: COLORS.text,
    fontFamily: FONTS.mono,
    fontSize: 14,
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  aiBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(200,245,58,0.2)',
    borderColor: COLORS.accent,
    borderWidth: 1,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 999,
    gap: 6,
  },
  aiDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.accent,
  },
  aiText: {
    color: COLORS.accent,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  liveStats: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 14,
  },
  liveChip: {
    flex: 1,
    backgroundColor: COLORS.card,
    borderColor: COLORS.border,
    borderWidth: 1,
    borderRadius: 12,
    padding: 12,
  },
  liveChipLabel: {
    color: COLORS.textMuted,
    fontSize: 10,
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    fontWeight: '700',
  },
  liveChipValue: {
    color: COLORS.text,
    fontFamily: FONTS.mono,
    fontSize: 16,
    marginTop: 4,
  },
  setupGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginTop: 14,
  },
  setupCard: {
    width: '48.5%',
  },
  setupLabel: {
    color: COLORS.textMuted,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
  },
  setupValue: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: '700',
    marginTop: 6,
  },
  controlRow: {
    alignItems: 'center',
    marginTop: 28,
  },
  recordBtn: {
    width: 84,
    height: 84,
    borderRadius: 42,
    borderWidth: 4,
    alignItems: 'center',
    justifyContent: 'center',
    shadowOpacity: 0.5,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 0 },
  },
  recordBtnInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardTitle: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: '700',
  },
  completeMsg: {
    color: COLORS.accent,
    fontSize: 13,
    fontWeight: '700',
    marginTop: 12,
    textAlign: 'center',
  },
  endButtons: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 14,
  },
  actionBtn: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  actionBtnPrimary: {
    backgroundColor: COLORS.accent,
  },
  actionBtnPrimaryText: {
    color: COLORS.bg,
    fontWeight: '800',
    fontSize: 14,
  },
  actionBtnGhost: {
    backgroundColor: COLORS.card,
    borderColor: COLORS.border,
    borderWidth: 1,
  },
  actionBtnGhostText: {
    color: COLORS.text,
    fontWeight: '700',
    fontSize: 14,
  },
});
