import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { COLORS, FONTS } from '../../theme';
import { RallySample } from '../../data/mockData';

interface RallyBarProps {
  rallies: RallySample[];
  height?: number;
}

export function RallyBar({ rallies, height = 140 }: RallyBarProps): React.ReactElement {
  const max = Math.max(...rallies.map(r => r.shots), 1);
  return (
    <View>
      <View style={[styles.row, { height }]}>
        {rallies.map((r, i) => {
          const h = Math.max((r.shots / max) * (height - 20), 6);
          const color = r.wonByPlayer === 0 ? COLORS.accent : COLORS.blue;
          return (
            <View key={i} style={styles.col}>
              <Text style={styles.label}>{r.shots}</Text>
              <View style={[styles.bar, { height: h, backgroundColor: color }]} />
            </View>
          );
        })}
      </View>
      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.dot, { backgroundColor: COLORS.accent }]} />
          <Text style={styles.legendText}>You won</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.dot, { backgroundColor: COLORS.blue }]} />
          <Text style={styles.legendText}>Opponent won</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    paddingHorizontal: 4,
  },
  col: {
    flex: 1,
    alignItems: 'center',
    marginHorizontal: 2,
  },
  label: {
    color: COLORS.textMuted,
    fontFamily: FONTS.mono,
    fontSize: 10,
    marginBottom: 4,
  },
  bar: {
    width: '100%',
    borderTopLeftRadius: 4,
    borderTopRightRadius: 4,
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 18,
    marginTop: 12,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  legendText: {
    color: COLORS.textSub,
    fontSize: 11,
  },
});
