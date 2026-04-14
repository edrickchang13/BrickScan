import { NativeModules, Platform } from 'react-native';

const { DepthCaptureModule } = NativeModules;

/**
 * Check if the device has LiDAR depth sensor available.
 *
 * @returns Promise<boolean> True if depth capture is available (iPhone 12 Pro+), false otherwise
 */
export const isDepthAvailable = async (): Promise<boolean> => {
  if (Platform.OS !== 'ios') {
    // Depth capture currently only supported on iOS with LiDAR
    return false;
  }

  try {
    if (!DepthCaptureModule) {
      console.warn('[DepthCapture] Native module not available');
      return false;
    }

    const available = await DepthCaptureModule.isDepthAvailable();
    return available ?? false;
  } catch (error) {
    console.error('[DepthCapture] Error checking depth availability:', error);
    return false;
  }
};

/**
 * Capture a synchronized RGB + depth frame.
 *
 * Returns file paths to temporary RGB and depth images if successful,
 * null if LiDAR is not available or capture failed.
 *
 * @returns Promise<{ rgbPath: string; depthPath: string } | null>
 *   rgbPath: Path to temporary JPEG RGB image
 *   depthPath: Path to temporary 16-bit PNG depth map (depth in mm)
 *   null if capture failed or LiDAR not available
 */
export const captureRGBD = async (): Promise<{ rgbPath: string; depthPath: string } | null> => {
  if (Platform.OS !== 'ios') {
    console.warn('[DepthCapture] Depth capture not supported on this platform');
    return null;
  }

  try {
    if (!DepthCaptureModule) {
      console.warn('[DepthCapture] Native module not available');
      return null;
    }

    const result = await DepthCaptureModule.captureRGBD();
    return result || null;
  } catch (error: any) {
    console.error('[DepthCapture] Capture failed:', error?.message || error);
    return null;
  }
};
