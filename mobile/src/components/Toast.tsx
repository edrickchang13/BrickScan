import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { View, Text, Animated, Easing } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

export type ToastType = 'success' | 'error' | 'info';

interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

interface ToastContextType {
  show: (message: string, type?: ToastType) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

const TOAST_DURATION = 3000;

const getBackgroundColor = (type: ToastType): string => {
  switch (type) {
    case 'success':
      return '#22C55E';
    case 'error':
      return '#EF4444';
    case 'info':
      return '#3B82F6';
    default:
      return '#000000';
  }
};

interface ToastItemProps {
  toast: Toast;
  onDismiss: (id: string) => void;
}

const ToastItem: React.FC<ToastItemProps> = ({ toast, onDismiss }) => {
  const [fadeAnim] = React.useState(new Animated.Value(0));
  const [slideAnim] = React.useState(new Animated.Value(100));

  React.useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 300,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 300,
        easing: Easing.out(Easing.ease),
        useNativeDriver: true,
      }),
    ]).start();

    const timer = setTimeout(() => {
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 300,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(slideAnim, {
          toValue: 100,
          duration: 300,
          easing: Easing.in(Easing.ease),
          useNativeDriver: true,
        }),
      ]).start(() => {
        onDismiss(toast.id);
      });
    }, TOAST_DURATION);

    return () => clearTimeout(timer);
  }, [toast.id, fadeAnim, slideAnim, onDismiss]);

  return (
    <Animated.View
      style={{
        opacity: fadeAnim,
        transform: [{ translateY: slideAnim }],
        marginBottom: 12,
      }}
    >
      <View
        style={{
          backgroundColor: getBackgroundColor(toast.type),
          paddingHorizontal: 16,
          paddingVertical: 12,
          borderRadius: 8,
          marginHorizontal: 16,
        }}
      >
        <Text
          style={{
            color: '#FFFFFF',
            fontSize: 14,
            fontWeight: '500',
            lineHeight: 20,
          }}
        >
          {toast.message}
        </Text>
      </View>
    </Animated.View>
  );
};

export const ToastProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const insets = useSafeAreaInsets();

  const show = useCallback((message: string, type: ToastType = 'info') => {
    const id = Date.now().toString() + Math.random();
    setToasts((prev) => [...prev, { id, message, type }]);
  }, []);

  const onDismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ show }}>
      {children}
      <View
        pointerEvents="none"
        style={{
          position: 'absolute',
          bottom: insets.bottom + 16,
          left: 0,
          right: 0,
          zIndex: 9999,
        }}
      >
        {toasts.map((toast) => (
          <ToastItem
            key={toast.id}
            toast={toast}
            onDismiss={onDismiss}
          />
        ))}
      </View>
    </ToastContext.Provider>
  );
};

export const useToast = (): ToastContextType => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
};
