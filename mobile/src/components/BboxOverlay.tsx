/**
 * BboxOverlay — draws tracked detection bounding boxes on top of the camera
 * preview in ContinuousScanScreen.
 *
 * Each box's colour reflects its track state:
 *   - yellow   — pending (seen, not locked)
 *   - green    — locked in
 *   - gray     — decaying (track missed the last frame)
 *
 * Bbox coords are normalised [0-1] in the backend's original image space.
 * We scale them to the overlay's pixel dims. Because the CameraView has
 * `resizeMode='cover'` (implicit), the image may be cropped to fit the
 * screen — we compensate with an aspect-ratio-aware letterbox map.
 */
import React from 'react';
import { View, Text, StyleSheet, ViewProps } from 'react-native';
import { C, R } from '@/constants/theme';

export type BboxTrackVisual = {
  id: string;
  /** Normalised [x1, y1, x2, y2] in [0,1] image-space coords. */
  bbox: [number, number, number, number];
  partNum: string;
  /** Final display confidence (fused EMA across frames) in [0,1]. */
  confidence: number;
  state: 'pending' | 'locked' | 'decaying';
};

interface Props extends ViewProps {
  tracks: BboxTrackVisual[];
  /** Aspect ratio of the SOURCE image (width / height). Helps letterbox. */
  sourceAspectRatio?: number;
}

export const BboxOverlay: React.FC<Props> = ({
  tracks,
  sourceAspectRatio,
  style,
  ...rest
}) => (
  <View
    style={[StyleSheet.absoluteFill, style]}
    pointerEvents="none"
    {...rest}
  >
    {tracks.map(track => (
      <BboxRect
        key={track.id}
        track={track}
        sourceAspectRatio={sourceAspectRatio}
      />
    ))}
  </View>
);

const BboxRect: React.FC<{
  track: BboxTrackVisual;
  sourceAspectRatio?: number;
}> = ({ track, sourceAspectRatio }) => {
  const [layout, setLayout] = React.useState<{ w: number; h: number } | null>(null);

  // Percent-based layout: x1/y1 = top-left, width/height = right-left etc.
  // Conversion to % works even if we don't know pixel dims, because RN
  // honors '%' values in style. We only need the container dimensions to
  // handle letterbox cover-mode aspect-ratio compensation.
  const [x1, y1, x2, y2] = track.bbox;

  // Basic letterbox compensation for 'cover' scaling when source aspect
  // differs from screen aspect. If we don't know the source aspect we
  // just pass the raw normalised coords through.
  let adjX1 = x1, adjY1 = y1, adjX2 = x2, adjY2 = y2;
  if (layout && sourceAspectRatio) {
    const screenAR = layout.w / layout.h;
    if (screenAR > sourceAspectRatio) {
      // Screen wider than source → image scales to fill width → crops top/bottom
      const scale = screenAR / sourceAspectRatio;
      const cropPct = (scale - 1) / 2 / scale;
      adjY1 = (y1 - cropPct) * scale;
      adjY2 = (y2 - cropPct) * scale;
    } else if (screenAR < sourceAspectRatio) {
      // Screen taller than source → scales to fill height → crops sides
      const scale = sourceAspectRatio / screenAR;
      const cropPct = (scale - 1) / 2 / scale;
      adjX1 = (x1 - cropPct) * scale;
      adjX2 = (x2 - cropPct) * scale;
    }
  }

  const left = `${Math.max(0, adjX1 * 100)}%`;
  const top = `${Math.max(0, adjY1 * 100)}%`;
  const width = `${Math.max(0, (adjX2 - adjX1) * 100)}%`;
  const height = `${Math.max(0, (adjY2 - adjY1) * 100)}%`;

  // Skip boxes that are entirely off-screen after letterbox adjust
  if (adjX2 <= 0 || adjY2 <= 0 || adjX1 >= 1 || adjY1 >= 1) return null;

  const colour =
    track.state === 'locked'  ? C.green
    : track.state === 'pending' ? C.yellow
    : 'rgba(255,255,255,0.35)';

  return (
    <View
      onLayout={e => setLayout({
        w: e.nativeEvent.layout.width,
        h: e.nativeEvent.layout.height,
      })}
      style={StyleSheet.absoluteFill}
    >
      <View
        style={[
          styles.box,
          {
            left: left as any,
            top: top as any,
            width: width as any,
            height: height as any,
            borderColor: colour,
            borderWidth: track.state === 'locked' ? 3 : 2,
          },
        ]}
      >
        {track.state !== 'decaying' && (
          <View style={[styles.labelPill, { backgroundColor: colour }]}>
            <Text style={styles.labelText} numberOfLines={1}>
              {track.partNum ? `#${track.partNum}` : '…'}
              {track.confidence > 0 && ` ${Math.round(track.confidence * 100)}%`}
            </Text>
          </View>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  box: {
    position: 'absolute',
    borderRadius: 4,
  },
  labelPill: {
    position: 'absolute',
    top: -22,
    left: 0,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: R.sm,
    maxWidth: 160,
  },
  labelText: {
    color: C.text,
    fontSize: 11,
    fontWeight: '700',
  },
});
