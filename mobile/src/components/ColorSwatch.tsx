import React from 'react';
import { View, Text } from 'react-native';

interface ColorSwatchProps {
  hexColor: string;
  name?: string;
  size?: number;
  showLabel?: boolean;
}

export const ColorSwatch: React.FC<ColorSwatchProps> = ({
  hexColor,
  name,
  size = 32,
  showLabel = false,
}) => {
  const isTransparent =
    hexColor.toLowerCase() === '#00000000' ||
    hexColor.toLowerCase() === 'transparent';

  const getBackgroundStyle = () => {
    if (isTransparent) {
      return {
        backgroundColor: '#FFFFFF',
        backgroundImage: 'repeating-conic-gradient(#D3D3D3 0% 25%, transparent 0% 50%)',
        backgroundSize: '8px 8px',
        backgroundPosition: '0 0, 4px 4px',
      };
    }
    return {
      backgroundColor: hexColor,
    };
  };

  return (
    <View
      style={{
        alignItems: 'center',
        gap: 8,
      }}
    >
      <View
        style={{
          width: size,
          height: size,
          borderRadius: size / 2,
          borderWidth: 2,
          borderColor: '#D0D0D0',
          overflow: 'hidden',
          ...getBackgroundStyle(),
        }}
      >
        {isTransparent && (
          <View
            style={{
              position: 'absolute',
              width: '100%',
              height: '100%',
              backgroundColor: '#E8E8E8',
              opacity: 0.5,
            }}
          />
        )}
      </View>

      {showLabel && name && (
        <Text
          style={{
            fontSize: 12,
            color: '#666666',
            fontWeight: '500',
            textAlign: 'center',
            maxWidth: size + 20,
          }}
        >
          {name}
        </Text>
      )}
    </View>
  );
};
