import React from 'react';
import { Tabs } from 'expo-router';
import { Platform } from 'react-native';
import { COLORS } from '../../src/theme';
import { Icon, IconName } from '../../src/components/ui/Icon';

interface TabConfig {
  name: string;
  title: string;
  icon: IconName;
  size: number;
}

const TAB_CONFIG: TabConfig[] = [
  { name: 'index', title: 'Home', icon: 'grid', size: 22 },
  { name: 'record', title: 'Record', icon: 'circle-fill', size: 28 },
  { name: 'stats', title: 'Stats', icon: 'bar-chart', size: 22 },
  { name: 'highlights', title: 'Clips', icon: 'play-circle', size: 22 },
  { name: 'profile', title: 'Profile', icon: 'user-circle', size: 22 },
];

export default function TabsLayout(): React.ReactElement {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: COLORS.surface,
          borderTopColor: COLORS.border,
          borderTopWidth: 1,
          height: Platform.OS === 'ios' ? 84 : 64,
          paddingTop: 6,
          paddingBottom: Platform.OS === 'ios' ? 24 : 8,
        },
        tabBarActiveTintColor: COLORS.accent,
        tabBarInactiveTintColor: COLORS.textMuted,
        tabBarLabelStyle: {
          fontSize: 10,
          fontWeight: '600',
          letterSpacing: 0.3,
        },
      }}
    >
      {TAB_CONFIG.map(t => (
        <Tabs.Screen
          key={t.name}
          name={t.name}
          options={{
            title: t.title,
            tabBarIcon: ({ color }) => <Icon name={t.icon} size={t.size} color={color} />,
          }}
        />
      ))}
    </Tabs>
  );
}
