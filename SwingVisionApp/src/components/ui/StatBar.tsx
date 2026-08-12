import React, { useEffect } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Animated, { useAnimatedStyle, useSharedValue, withTiming } from 'react-native-reanimated';
import { COLORS, FONTS } from '../../theme';

type BarColor = 'accent' | 'blue' | 'red' | 'orange' | 'purple';

const COLOR_MAP: Record<BarColor, string> = {
  accent: COLORS.accent,
  blue: COLORS.blue,
  red: COLORS.red,
  orange: COLORS.orange,
  purple: COLORS.purple,
};

interface StatBarProps {
  label: string;
  val1: number;
  val2: number;
  color1?: BarColor;
  color2?: BarColor;
}

export function StatBar({
  label,
  val1,
  val2,
  color1 = 'accent',
  color2 = 'blue',
}: StatBarProps): React.ReactElement {
  const total = val1 + val2 || 1;
  const targetLeft = (val1 / total) * 100;

  const left = useSharedValue(0);

  useEffect(() => {
    left.value = withTiming(targetLeft, { duration: 600 });
  }, [targetLeft, left]);

  const leftStyle = useAnimatedStyle(() => ({
    width: `${left.value}%`,
  }));
  const rightStyle = useAnimatedStyle(() => ({
    width: `${100 - left.value}%`,
  }));

  return (
    <View style={styles.row}>
      <View style={styles.header}>
        <Text style={[styles.value, { color: COLOR_MAP[color1] }]}>{val1}</Text>
        <Text style={styles.label}>{label}</Text>
        <Text style={[styles.value, styles.valueRight, { color: COLOR_MAP[color2] }]}>{val2}</Text>
      </View>
      <View style={styles.barTrack}>
        <Animated.View style={[styles.barLeft, { backgroundColor: COLOR_MAP[color1] }, leftStyle]} />
        <Animated.View style={[styles.barRight, { backgroundColor: COLOR_MAP[color2] }, rightStyle]} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    marginBottom: 14,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  label: {
    flex: 1,
    textAlign: 'center',
    color: COLORS.textMuted,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  value: {
    fontFamily: FONTS.mono,
    fontSize: 18,
    minWidth: 40,
  },
  valueRight: {
    textAlign: 'right',
  },
  barTrack: {
    flexDirection: 'row',
    height: 4,
    borderRadius: 2,
    overflow: 'hidden',
    backgroundColor: COLORS.border,
  },
  barLeft: {
    height: '100%',
  },
  barRight: {
    height: '100%',
  },
});
