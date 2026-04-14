import React from 'react';
import { View, TouchableOpacity, Text, ScrollView } from 'react-native';
import { Colors } from '../constants/colors';

interface ChipOption {
  label: string;
  value: string;
}

interface FilterChipsProps {
  options: ChipOption[];
  selected: string[];
  onToggle: (value: string) => void;
}

export const FilterChips: React.FC<FilterChipsProps> = ({
  options,
  selected,
  onToggle,
}) => {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      style={{
        paddingHorizontal: 16,
        paddingVertical: 8,
      }}
      contentContainerStyle={{
        gap: 8,
      }}
    >
      {options.map((option) => {
        const isSelected = selected.includes(option.value);

        return (
          <TouchableOpacity
            key={option.value}
            onPress={() => onToggle(option.value)}
            style={{
              paddingHorizontal: 16,
              paddingVertical: 8,
              borderRadius: 20,
              backgroundColor: isSelected ? Colors.primary : Colors.background,
              borderWidth: 1,
              borderColor: isSelected ? Colors.primary : Colors.border,
            }}
          >
            <Text
              style={{
                fontSize: 13,
                fontWeight: '500',
                color: isSelected ? '#FFFFFF' : Colors.text,
              }}
            >
              {option.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </ScrollView>
  );
};
