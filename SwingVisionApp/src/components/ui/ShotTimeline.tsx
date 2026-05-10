import React from 'react';
import { StyleSheet, View } from 'react-native';
import { COLORS } from '../../theme';
import { ShotType } from '../../data/mockData';

interface ShotTimelineProps {
  shots: ShotType[];
}

const COLOR: Record<ShotType, string> = {
  serve: COLORS.accent,
  winner: COLORS.blue,
  error: COLORS.red,
};

export function ShotTimeline({ shots }: ShotTimelineProps): React.ReactElement {
  return (
    <View style={styles.row}>
      {shots.map((s, i) => (
        <View key={i} style={[styles.dot, { backgroundColor: COLOR[s] }]} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
});
