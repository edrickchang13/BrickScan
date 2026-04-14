import { useState, useCallback, useRef } from 'react';
import { CameraView, useCameraPermissions } from 'expo-camera';

interface CameraRef {
  takePicture?: () => Promise<{ uri: string; base64?: string }>;
}

interface UseCameraReturn {
  hasPermission: boolean | null;
  cameraRef: React.RefObject<CameraView>;
  takePicture: () => Promise<string>;
  requestPermission: () => Promise<void>;
  isLoading: boolean;
}

export const useCamera = (): UseCameraReturn => {
  const cameraRef = useRef<CameraView>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();

  const handleRequestPermission = useCallback(async () => {
    try {
      const result = await requestPermission();
      if (!result.granted) {
        throw new Error('Camera permission denied');
      }
    } catch (error) {
      console.error('Failed to request camera permission:', error);
      throw error;
    }
  }, [requestPermission]);

  const takePicture = useCallback(async (): Promise<string> => {
    if (!cameraRef.current) {
      throw new Error('Camera reference not available');
    }

    setIsLoading(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.8,
      });

      if (!photo || !photo.base64) {
        throw new Error('Failed to capture image');
      }

      return `data:image/jpeg;base64,${photo.base64}`;
    } catch (error) {
      console.error('Failed to take picture:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    hasPermission: permission?.granted ?? null,
    cameraRef,
    takePicture,
    requestPermission: handleRequestPermission,
    isLoading,
  };
};
