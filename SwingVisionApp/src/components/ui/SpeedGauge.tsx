import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Svg, { Circle, Line, Path } from 'react-native-svg';
import { COLORS, FONTS } from '../../theme';

interface SpeedGaugeProps {
  speed: number;
  label: string;
  max?: number;
}

const START_DEG = -220;
const END_DEG = 40;
const SWEEP = END_DEG - START_DEG;
const CENTER_X = 50;
const CENTER_Y = 50;
const RADIUS = 38;

function polar(cx: number, cy: number, r: number, angleDeg: number): { x: number; y: number } {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function arcPath(startDeg: number, endDeg: number): string {
  const start = polar(CENTER_X, CENTER_Y, RADIUS, startDeg);
  const end = polar(CENTER_X, CENTER_Y, RADIUS, endDeg);
  const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
  const sweepFlag = endDeg > startDeg ? 1 : 0;
  return `M ${start.x} ${start.y} A ${RADIUS} ${RADIUS} 0 ${largeArc} ${sweepFlag} ${end.x} ${end.y}`;
}

export function SpeedGauge({ speed, label, max = 250 }: SpeedGaugeProps): React.ReactElement {
  const ratio = Math.min(Math.max(speed / max, 0), 1);
  const currentDeg = START_DEG + SWEEP * ratio;
  const needleEnd = polar(CENTER_X, CENTER_Y, RADIUS - 4, currentDeg);

  return (
    <View style={styles.container}>
      <Svg viewBox="0 0 100 65" width={120} height={78}>
        <Path
          d={arcPath(START_DEG, END_DEG)}
          stroke={COLORS.border}
          strokeWidth={5}
          strokeLinecap="round"
          fill="none"
        />
        <Path
          d={arcPath(START_DEG, currentDeg)}
          stroke={COLORS.accent}
          strokeWidth={5}
          strokeLinecap="round"
          fill="none"
        />
        <Line
          x1={CENTER_X}
          y1={CENTER_Y}
          x2={needleEnd.x}
          y2={needleEnd.y}
          stroke={COLORS.text}
          strokeWidth={1.6}
          strokeLinecap="round"
        />
        <Circle cx={CENTER_X} cy={CENTER_Y} r={3} fill={COLORS.accent} />
      </Svg>
      <Text style={styles.value}>{speed}</Text>
      <Text style={styles.unit}>km/h</Text>
      <Text style={styles.label}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    flex: 1,
  },
  value: {
    color: COLORS.text,
    fontFamily: FONTS.mono,
    fontSize: 22,
    marginTop: -10,
  },
  unit: {
    color: COLORS.textMuted,
    fontSize: 10,
    marginTop: -2,
  },
  label: {
    color: COLORS.textSub,
    fontSize: 11,
    marginTop: 6,
    fontWeight: '600',
    letterSpacing: 0.4,
  },
});
