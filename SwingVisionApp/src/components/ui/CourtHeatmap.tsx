import React from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Circle, Line, Rect } from 'react-native-svg';
import { COLORS } from '../../theme';
import { Shot, ShotType } from '../../data/mockData';

interface CourtHeatmapProps {
  shots: Shot[];
  height?: number;
}

const TYPE_COLOR: Record<ShotType, string> = {
  serve: COLORS.accent,
  winner: COLORS.blue,
  error: COLORS.red,
};

export function CourtHeatmap({ shots, height = 200 }: CourtHeatmapProps): React.ReactElement {
  return (
    <View style={[styles.wrap, { height }]}>
      <Svg viewBox="0 0 100 70" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
        <Rect x={0} y={0} width={100} height={70} rx={4} fill={COLORS.courtGreen} />
        <Rect
          x={6}
          y={4}
          width={88}
          height={62}
          rx={1}
          stroke="#ffffff"
          strokeWidth={0.7}
          fill="none"
        />
        <Line x1={6} y1={35} x2={94} y2={35} stroke="#ffffff" strokeWidth={1.5} opacity={0.9} />
        <Rect
          x={20}
          y={4}
          width={60}
          height={31}
          stroke="#ffffff"
          strokeWidth={0.6}
          fill="none"
        />
        <Rect
          x={20}
          y={35}
          width={60}
          height={31}
          stroke="#ffffff"
          strokeWidth={0.6}
          fill="none"
        />
        <Line x1={50} y1={4} x2={50} y2={35} stroke="#ffffff" strokeWidth={0.6} />
        <Line x1={50} y1={35} x2={50} y2={66} stroke="#ffffff" strokeWidth={0.6} />
        <Line x1={20} y1={4} x2={20} y2={66} stroke="#ffffff" strokeWidth={0.6} opacity={0.8} />
        <Line x1={80} y1={4} x2={80} y2={66} stroke="#ffffff" strokeWidth={0.6} opacity={0.8} />
        {shots.map((shot, i) => (
          <Circle
            key={i}
            cx={shot.x}
            cy={shot.y}
            r={2.5}
            fill={TYPE_COLOR[shot.type]}
            stroke={COLORS.bg}
            strokeWidth={0.4}
          />
        ))}
      </Svg>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    width: '100%',
    overflow: 'hidden',
    borderRadius: 8,
  },
});
