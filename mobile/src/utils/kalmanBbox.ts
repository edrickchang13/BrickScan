/**
 * Minimal 1D constant-velocity Kalman filter — applied per bbox coordinate
 * (x1, y1, x2, y2). Smooths jitter between frames without needing a full
 * 2D state vector per bbox (which would be overkill for our use case where
 * the subject is a mostly-static pile of bricks on a table).
 *
 * State per coord:
 *   value (position), velocity
 * Process noise:  q  — how much the system drifts frame to frame
 * Measurement noise: r — how noisy detections are
 * Ratio controls smoothing:
 *   r >> q → heavy smoothing (slow to follow fast motion)
 *   r << q → trust every measurement (jittery)
 *
 * For a phone held steady over LEGO, we want moderate smoothing:
 *   q = 0.0001  (bricks barely drift in normalised coords)
 *   r = 0.005   (detector jitter is ~±0.7% of image width per frame)
 */

interface KalmanCoord {
  value: number;
  velocity: number;
  covariance: [[number, number], [number, number]];
}

const Q_PROCESS = 1e-4;
const R_MEASURE = 5e-3;

function initCoord(initial: number): KalmanCoord {
  return {
    value: initial,
    velocity: 0,
    covariance: [[1, 0], [0, 1]],
  };
}

function stepCoord(state: KalmanCoord, measurement: number, dt: number): KalmanCoord {
  // Predict: x' = x + v*dt
  const predicted_value = state.value + state.velocity * dt;
  const predicted_velocity = state.velocity;

  // Covariance predict (F * P * F^T + Q), where F = [[1, dt], [0, 1]]
  const [[p00, p01], [p10, p11]] = state.covariance;
  const pred_p00 = p00 + dt*p10 + dt*p01 + dt*dt*p11 + Q_PROCESS;
  const pred_p01 = p01 + dt*p11;
  const pred_p10 = p10 + dt*p11;
  const pred_p11 = p11 + Q_PROCESS;

  // Innovation (measurement - predicted position)
  const innovation = measurement - predicted_value;
  // Innovation covariance:  S = H * P * H^T + R, with H = [1, 0]
  const s = pred_p00 + R_MEASURE;
  // Kalman gain:  K = P * H^T * S^-1
  const k0 = pred_p00 / s;
  const k1 = pred_p10 / s;

  const new_value = predicted_value + k0 * innovation;
  const new_velocity = predicted_velocity + k1 * innovation;

  // Update covariance:  P' = (I - K * H) * P
  const new_p00 = (1 - k0) * pred_p00;
  const new_p01 = (1 - k0) * pred_p01;
  const new_p10 = -k1 * pred_p00 + pred_p10;
  const new_p11 = -k1 * pred_p01 + pred_p11;

  return {
    value: new_value,
    velocity: new_velocity,
    covariance: [[new_p00, new_p01], [new_p10, new_p11]],
  };
}

export interface KalmanBboxState {
  x1: KalmanCoord;
  y1: KalmanCoord;
  x2: KalmanCoord;
  y2: KalmanCoord;
  lastUpdateTs: number;
}

export function initBboxKalman(bbox: [number, number, number, number], now: number): KalmanBboxState {
  return {
    x1: initCoord(bbox[0]),
    y1: initCoord(bbox[1]),
    x2: initCoord(bbox[2]),
    y2: initCoord(bbox[3]),
    lastUpdateTs: now,
  };
}

export function stepBboxKalman(
  state: KalmanBboxState,
  measurement: [number, number, number, number],
  now: number,
): KalmanBboxState {
  // Use ms-scale dt so process-noise numbers above feel natural. The dt is
  // typically ~1.2 (one FRAME_INTERVAL_MS). Cap at 2s in case of long pauses
  // so the velocity doesn't explode after a resume.
  const dt = Math.min(2.0, (now - state.lastUpdateTs) / 1000);
  return {
    x1: stepCoord(state.x1, measurement[0], dt),
    y1: stepCoord(state.y1, measurement[1], dt),
    x2: stepCoord(state.x2, measurement[2], dt),
    y2: stepCoord(state.y2, measurement[3], dt),
    lastUpdateTs: now,
  };
}

export function kalmanBbox(state: KalmanBboxState): [number, number, number, number] {
  return [state.x1.value, state.y1.value, state.x2.value, state.y2.value];
}
