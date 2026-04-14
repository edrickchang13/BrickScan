import React, { useRef, useEffect } from 'react';
import {
  View,
  Text,
  Modal,
  Animated,
  Easing,
  Dimensions,
  TouchableOpacity,
  PanResponder,
  GestureResponderEvent,
  PanResponderGestureState,
  ReactNode,
} from 'react-native';

interface BottomSheetProps {
  isVisible: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
  height?: number;
}

const { height: screenHeight } = Dimensions.get('window');

export const BottomSheet: React.FC<BottomSheetProps> = ({
  isVisible,
  onClose,
  title,
  children,
  height = screenHeight * 0.6,
}) => {
  const slideAnim = useRef(new Animated.Value(height)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: (_: GestureResponderEvent, state: PanResponderGestureState) => {
        return Math.abs(state.dy) > 10 && state.dy > 0;
      },
      onPanResponderMove: (
        _: GestureResponderEvent,
        state: PanResponderGestureState
      ) => {
        if (state.dy > 0) {
          slideAnim.setValue(state.dy);
        }
      },
      onPanResponderRelease: (
        _: GestureResponderEvent,
        state: PanResponderGestureState
      ) => {
        if (state.dy > height * 0.2) {
          closeSheet();
        } else {
          openSheet();
        }
      },
    })
  ).current;

  const openSheet = () => {
    Animated.parallel([
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 300,
        easing: Easing.out(Easing.ease),
        useNativeDriver: false,
      }),
      Animated.timing(fadeAnim, {
        toValue: 0.5,
        duration: 300,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: false,
      }),
    ]).start();
  };

  const closeSheet = () => {
    Animated.parallel([
      Animated.timing(slideAnim, {
        toValue: height,
        duration: 300,
        easing: Easing.in(Easing.ease),
        useNativeDriver: false,
      }),
      Animated.timing(fadeAnim, {
        toValue: 0,
        duration: 300,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: false,
      }),
    ]).start(() => {
      onClose();
    });
  };

  useEffect(() => {
    if (isVisible) {
      openSheet();
    }
  }, [isVisible]);

  const handleBackdropPress = () => {
    closeSheet();
  };

  if (!isVisible) {
    return null;
  }

  return (
    <Modal
      visible={isVisible}
      transparent
      animationType="none"
      onRequestClose={handleBackdropPress}
    >
      <TouchableOpacity
        activeOpacity={1}
        style={{
          flex: 1,
          backgroundColor: 'transparent',
        }}
        onPress={handleBackdropPress}
      >
        <Animated.View
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: '#000000',
            opacity: fadeAnim,
          }}
        />
      </TouchableOpacity>

      <Animated.View
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: height,
          backgroundColor: '#FFFFFF',
          borderTopLeftRadius: 20,
          borderTopRightRadius: 20,
          overflow: 'hidden',
          transform: [{ translateY: slideAnim }],
        }}
        {...panResponder.panHandlers}
      >
        {/* Handle */}
        <View
          style={{
            paddingVertical: 12,
            alignItems: 'center',
            borderBottomWidth: 1,
            borderBottomColor: '#E0E0E0',
          }}
        >
          <View
            style={{
              width: 40,
              height: 4,
              backgroundColor: '#D0D0D0',
              borderRadius: 2,
            }}
          />
        </View>

        {/* Title */}
        {title && (
          <View
            style={{
              paddingHorizontal: 16,
              paddingVertical: 12,
              borderBottomWidth: 1,
              borderBottomColor: '#E0E0E0',
            }}
          >
            <Text
              style={{
                fontSize: 18,
                fontWeight: '600',
                color: '#1A1A1A',
              }}
            >
              {title}
            </Text>
          </View>
        )}

        {/* Content */}
        <View
          style={{
            flex: 1,
            overflow: 'hidden',
          }}
        >
          {children}
        </View>
      </Animated.View>
    </Modal>
  );
};
