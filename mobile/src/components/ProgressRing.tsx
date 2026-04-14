import React, { useEffect } from 'react';
import { View, Text, Animated } from 'react-native';
import Svg, { Circle, G } from 'react-native-svg';

interface ProgressRingProps {
  progress: number;
  size: number;
  strokeWidth: number;
  color: string;
}

export const ProgressRing: React.FC<ProgressRingProps> = ({
  progress,
  size,
  strokeWidth,
  color,
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (progress / 100) * circumference;

  const animatedOffset = React.useRef(new Animated.Value(circumference)).current;

  useEffect(() => {
    Animated.timing(animatedOffset, {
      toValue: offset,
      duration: 500,
      useNativeDriver: false,
    }).start();
  }, [progress]);

  return (
    <View
      style={{
        justifyContent: 'center',
        alignItems: 'center',
        width: size,
        height: size,
      }}
    >
      <Svg width={size} height={size}>
        <G>
          {/* Background circle */}
          <Circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="#E0E0E0"
            strokeWidth={strokeWidth}
            fill="none"
          />

          {/* Progress circle */}
          <Circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={color}
            strokeWidth={strokeWidth}
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
          />
        </G>
      </Svg>

      {/* Center text */}
      <View
        style={{
          position: 'absolute',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <Text
          style={{
            fontSize: Math.floor(size / 4),
            fontWeight: '600',
            color: color,
          }}
        >
          {Math.round(progress)}%
        </Text>
      </View>
    </View>
  );
};
