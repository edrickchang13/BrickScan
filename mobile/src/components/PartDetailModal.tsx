import React, { useState } from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  ScrollView,
  Alert,
  Linking,
  ActivityIndicator,
} from 'react-native';
import { BottomSheet } from './BottomSheet';
import { Colors } from '../constants/colors';

interface InventoryItem {
  id: string;
  partNum: string;
  partName: string;
  colorHex: string;
  colorName: string;
  quantity: number;
  imageUrl?: string;
}

interface PartDetailModalProps {
  part: InventoryItem | null;
  isVisible: boolean;
  onClose: () => void;
  onUpdateQuantity: (id: string, quantity: number) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}

export const PartDetailModal: React.FC<PartDetailModalProps> = ({
  part,
  isVisible,
  onClose,
  onUpdateQuantity,
  onDelete,
}) => {
  const [quantity, setQuantity] = useState(part?.quantity || 0);
  const [isUpdating, setIsUpdating] = useState(false);

  React.useEffect(() => {
    if (part) {
      setQuantity(part.quantity);
    }
  }, [part]);

  if (!part) {
    return null;
  }

  const handleQuantityChange = async (newQuantity: number) => {
    if (newQuantity < 0) return;
    setQuantity(newQuantity);
    setIsUpdating(true);
    try {
      await onUpdateQuantity(part.id, newQuantity);
    } catch (error) {
      setQuantity(part.quantity);
    } finally {
      setIsUpdating(false);
    }
  };

  const handleDelete = () => {
    Alert.alert(
      'Delete Part',
      `Are you sure you want to delete ${part.partName} from your inventory?`,
      [
        { text: 'Cancel', onPress: () => {}, style: 'cancel' },
        {
          text: 'Delete',
          onPress: async () => {
            setIsUpdating(true);
            try {
              await onDelete(part.id);
              onClose();
            } finally {
              setIsUpdating(false);
            }
          },
          style: 'destructive',
        },
      ]
    );
  };

  const handleOpenBrickLink = () => {
    const url = `${Colors.bricklink}part.asp?P=${part.partNum}&colorID=${getColorId(part.colorHex)}`;
    Linking.openURL(url);
  };

  const handleOpenRebrickable = () => {
    const url = `https://rebrickable.com/parts/${part.partNum}/`;
    Linking.openURL(url);
  };

  const getColorId = (hex: string): number => {
    const colorMap: { [key: string]: number } = {
      '#FFFFFF': 1,
      '#FFFF00': 3,
      '#FF0000': 5,
      '#0000FF': 9,
      '#000000': 11,
      '#008000': 78,
      '#FF8000': 84,
    };
    return colorMap[hex] || 0;
  };

  return (
    <BottomSheet
      isVisible={isVisible}
      onClose={onClose}
      title="Part Details"
      height={600}
    >
      <ScrollView
        style={{ flex: 1 }}
        contentContainerStyle={{ padding: 16 }}
        scrollEnabled
      >
        {/* Part Image */}
        {part.imageUrl && (
          <View
            style={{
              width: '100%',
              height: 200,
              backgroundColor: Colors.background,
              borderRadius: 12,
              marginBottom: 16,
              overflow: 'hidden',
            }}
          >
            <Image
              source={{ uri: part.imageUrl }}
              style={{ width: '100%', height: '100%' }}
              resizeMode="contain"
            />
          </View>
        )}

        {/* Part Name */}
        <Text
          style={{
            fontSize: 18,
            fontWeight: '600',
            color: Colors.text,
            marginBottom: 8,
          }}
        >
          {part.partName}
        </Text>

        {/* Part Number */}
        <Text
          style={{
            fontSize: 14,
            color: Colors.textSecondary,
            marginBottom: 16,
          }}
        >
          Part #{part.partNum}
        </Text>

        {/* Color */}
        <View
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            marginBottom: 16,
            paddingBottom: 16,
            borderBottomWidth: 1,
            borderBottomColor: Colors.border,
          }}
        >
          <View
            style={{
              width: 40,
              height: 40,
              borderRadius: 20,
              backgroundColor: part.colorHex,
              borderWidth: 1,
              borderColor: Colors.border,
              marginRight: 12,
            }}
          />
          <Text
            style={{
              fontSize: 14,
              color: Colors.text,
              fontWeight: '500',
            }}
          >
            {part.colorName}
          </Text>
        </View>

        {/* Quantity Editor */}
        <Text
          style={{
            fontSize: 14,
            fontWeight: '600',
            color: Colors.text,
            marginBottom: 12,
          }}
        >
          Quantity
        </Text>

        <View
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            marginBottom: 24,
            backgroundColor: Colors.background,
            borderRadius: 8,
            padding: 8,
          }}
        >
          <TouchableOpacity
            onPress={() => handleQuantityChange(quantity - 1)}
            disabled={isUpdating || quantity === 0}
            style={{
              width: 40,
              height: 40,
              justifyContent: 'center',
              alignItems: 'center',
              borderRadius: 6,
              backgroundColor: Colors.surface,
              borderWidth: 1,
              borderColor: Colors.border,
            }}
          >
            <Text
              style={{
                fontSize: 18,
                fontWeight: '600',
                color: Colors.text,
              }}
            >
              -
            </Text>
          </TouchableOpacity>

          <View
            style={{
              flex: 1,
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            {isUpdating ? (
              <ActivityIndicator color={Colors.primary} />
            ) : (
              <Text
                style={{
                  fontSize: 18,
                  fontWeight: '600',
                  color: Colors.text,
                }}
              >
                {quantity}
              </Text>
            )}
          </View>

          <TouchableOpacity
            onPress={() => handleQuantityChange(quantity + 1)}
            disabled={isUpdating}
            style={{
              width: 40,
              height: 40,
              justifyContent: 'center',
              alignItems: 'center',
              borderRadius: 6,
              backgroundColor: Colors.surface,
              borderWidth: 1,
              borderColor: Colors.border,
            }}
          >
            <Text
              style={{
                fontSize: 18,
                fontWeight: '600',
                color: Colors.text,
              }}
            >
              +
            </Text>
          </TouchableOpacity>
        </View>

        {/* Links */}
        <View
          style={{
            gap: 10,
            marginBottom: 24,
          }}
        >
          <TouchableOpacity
            onPress={handleOpenBrickLink}
            style={{
              paddingVertical: 12,
              paddingHorizontal: 16,
              backgroundColor: Colors.background,
              borderRadius: 8,
              borderWidth: 1,
              borderColor: Colors.border,
            }}
          >
            <Text
              style={{
                fontSize: 14,
                fontWeight: '500',
                color: Colors.primary,
                textAlign: 'center',
              }}
            >
              View on BrickLink
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={handleOpenRebrickable}
            style={{
              paddingVertical: 12,
              paddingHorizontal: 16,
              backgroundColor: Colors.background,
              borderRadius: 8,
              borderWidth: 1,
              borderColor: Colors.border,
            }}
          >
            <Text
              style={{
                fontSize: 14,
                fontWeight: '500',
                color: Colors.primary,
                textAlign: 'center',
              }}
            >
              View on Rebrickable
            </Text>
          </TouchableOpacity>
        </View>

        {/* Delete Button */}
        <TouchableOpacity
          onPress={handleDelete}
          disabled={isUpdating}
          style={{
            paddingVertical: 12,
            paddingHorizontal: 16,
            backgroundColor: Colors.error,
            borderRadius: 8,
            opacity: isUpdating ? 0.6 : 1,
          }}
        >
          <Text
            style={{
              fontSize: 14,
              fontWeight: '600',
              color: '#FFFFFF',
              textAlign: 'center',
            }}
          >
            Delete from Inventory
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </BottomSheet>
  );
};
