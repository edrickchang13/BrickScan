import React from 'react';
import { View, TextInput, TouchableOpacity, Text } from 'react-native';
import { Colors } from '../constants/colors';

interface SearchBarProps {
  value: string;
  onChangeText: (text: string) => void;
  placeholder?: string;
  onClear?: () => void;
}

export const SearchBar: React.FC<SearchBarProps> = ({
  value,
  onChangeText,
  placeholder = 'Search...',
  onClear,
}) => {
  const handleClear = () => {
    onChangeText('');
    onClear?.();
  };

  return (
    <View
      style={{
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: Colors.background,
        borderRadius: 24,
        paddingHorizontal: 12,
        paddingVertical: 8,
        marginHorizontal: 16,
        marginVertical: 12,
        borderWidth: 1,
        borderColor: Colors.border,
      }}
    >
      {/* Search Icon */}
      <Text
        style={{
          fontSize: 18,
          marginRight: 8,
          color: Colors.textSecondary,
        }}
      >
        🔍
      </Text>

      {/* Input */}
      <TextInput
        style={{
          flex: 1,
          height: 40,
          fontSize: 14,
          color: Colors.text,
          paddingHorizontal: 0,
        }}
        placeholder={placeholder}
        placeholderTextColor={Colors.textSecondary}
        value={value}
        onChangeText={onChangeText}
        returnKeyType="search"
      />

      {/* Clear Button */}
      {value.length > 0 && (
        <TouchableOpacity
          onPress={handleClear}
          style={{
            padding: 4,
          }}
        >
          <Text
            style={{
              fontSize: 18,
              color: Colors.textSecondary,
            }}
          >
            ✕
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );
};
