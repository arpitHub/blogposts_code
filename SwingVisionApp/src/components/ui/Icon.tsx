import React from 'react';
import Svg, { Circle, Line, Path, Rect } from 'react-native-svg';

export type IconName =
  | 'grid'
  | 'circle-fill'
  | 'bar-chart'
  | 'play-circle'
  | 'user-circle'
  | 'bell'
  | 'chevron-right'
  | 'stop-square'
  | 'check';

interface IconProps {
  name: IconName;
  size?: number;
  color?: string;
}

export function Icon({ name, size = 22, color = '#fff' }: IconProps): React.ReactElement {
  const stroke = color;
  const sw = 2;
  switch (name) {
    case 'grid':
      return (
        <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
          <Rect x={3} y={3} width={7} height={7} rx={1.5} stroke={stroke} strokeWidth={sw} />
          <Rect x={14} y={3} width={7} height={7} rx={1.5} stroke={stroke} strokeWidth={sw} />
          <Rect x={3} y={14} width={7} height={7} rx={1.5} stroke={stroke} strokeWidth={sw} />
          <Rect x={14} y={14} width={7} height={7} rx={1.5} stroke={stroke} strokeWidth={sw} />
        </Svg>
      );
    case 'circle-fill':
      return (
        <Svg width={size} height={size} viewBox="0 0 24 24">
          <Circle cx={12} cy={12} r={10} fill={color} />
          <Circle cx={12} cy={12} r={4} fill="#0a0a0f" />
        </Svg>
      );
    case 'bar-chart':
      return (
        <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
          <Line x1={6} y1={20} x2={6} y2={12} stroke={stroke} strokeWidth={sw} strokeLinecap="round" />
          <Line x1={12} y1={20} x2={12} y2={6} stroke={stroke} strokeWidth={sw} strokeLinecap="round" />
          <Line x1={18} y1={20} x2={18} y2={14} stroke={stroke} strokeWidth={sw} strokeLinecap="round" />
        </Svg>
      );
    case 'play-circle':
      return (
        <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
          <Circle cx={12} cy={12} r={9} stroke={stroke} strokeWidth={sw} />
          <Path d="M10 8 L16 12 L10 16 Z" fill={stroke} />
        </Svg>
      );
    case 'user-circle':
      return (
        <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
          <Circle cx={12} cy={12} r={9} stroke={stroke} strokeWidth={sw} />
          <Circle cx={12} cy={10} r={3} stroke={stroke} strokeWidth={sw} />
          <Path d="M5.5 19c1.5-3 4-4.5 6.5-4.5s5 1.5 6.5 4.5" stroke={stroke} strokeWidth={sw} strokeLinecap="round" />
        </Svg>
      );
    case 'bell':
      return (
        <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
          <Path
            d="M6 8a6 6 0 1 1 12 0c0 4 1.5 6 1.5 6h-15S6 12 6 8Z"
            stroke={stroke}
            strokeWidth={sw}
            strokeLinejoin="round"
          />
          <Path d="M10 18a2 2 0 0 0 4 0" stroke={stroke} strokeWidth={sw} strokeLinecap="round" />
        </Svg>
      );
    case 'chevron-right':
      return (
        <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
          <Path d="M9 6l6 6-6 6" stroke={stroke} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" />
        </Svg>
      );
    case 'stop-square':
      return (
        <Svg width={size} height={size} viewBox="0 0 24 24">
          <Rect x={6} y={6} width={12} height={12} rx={2} fill={color} />
        </Svg>
      );
    case 'check':
      return (
        <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
          <Path d="M5 12l4 4 10-10" stroke={stroke} strokeWidth={sw + 0.5} strokeLinecap="round" strokeLinejoin="round" />
        </Svg>
      );
  }
}
