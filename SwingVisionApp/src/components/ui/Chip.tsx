import React from 'react';
import { Pressable, StyleSheet, Text, ViewStyle, StyleProp } from 'react-native';
import { COLORS, FONTS, RADII } from '../../theme';

interface ChipProps {
  label: string;
  active?: boolean;
  onPress?: () => void;
  variant?: 'default' | 'accent' | 'outline';
  style?: StyleProp<ViewStyle>;
}

export function Chip({ label, active = false, onPress, variant = 'default', style }: ChipProps): React.ReactElement {
  const isAccent = variant === 'accent' || active;
  const isOutline = variant === 'outline';
  return (
    <Pressable
      onPress={onPress}
      style={[
        styles.chip,
        isAccent && styles.chipActive,
        isOutline && styles.chipOutline,
        style,
      ]}
    >
      <Text
        style={[
          styles.label,
          isAccent && styles.labelActive,
          isOutline && styles.labelOutline,
        ]}
      >
        {label}
      </Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: RADII.pill,
    backgroundColor: COLORS.card,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignSelf: 'flex-start',
  },
  chipActive: {
    backgroundColor: COLORS.accent,
    borderColor: COLORS.accent,
  },
  chipOutline: {
    backgroundColor: COLORS.accentGlow,
    borderColor: COLORS.accent,
  },
  label: {
    color: COLORS.textSub,
    fontSize: 13,
    fontFamily: FONTS.body,
    fontWeight: '600',
  },
  labelActive: {
    color: COLORS.bg,
    fontWeight: '700',
  },
  labelOutline: {
    color: COLORS.accent,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
});
