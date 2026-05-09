/**
 * Unit tests for the 1D constant-velocity Kalman filter that smooths bbox
 * coordinates between frames. We're not validating mathematical optimality
 * here (that's covered by Kalman's published guarantees) — we're checking
 * the practical properties the UI relies on:
 *
 *   1. Initialised filter outputs the seed value.
 *   2. Repeating the same measurement converges to that value.
 *   3. A noisy measurement gets smoothed (output amplitude < input amplitude).
 *   4. A genuine drift in measurement is followed within a few frames.
 *   5. Long dt between updates doesn't blow up the velocity estimate.
 */
import { initBboxKalman, stepBboxKalman, kalmanBbox } from '@/utils/kalmanBbox';

describe('kalmanBbox', () => {
  it('initialises to the seed bbox', () => {
    const t0 = 1_000_000;
    const seed: [number, number, number, number] = [0.1, 0.2, 0.4, 0.5];
    const s = initBboxKalman(seed, t0);
    expect(kalmanBbox(s)).toEqual(seed);
  });

  it('converges to a stable measurement', () => {
    const t0 = 1_000_000;
    const target: [number, number, number, number] = [0.3, 0.4, 0.6, 0.7];
    let s = initBboxKalman(target, t0);
    for (let i = 1; i <= 20; i++) {
      s = stepBboxKalman(s, target, t0 + i * 1200);
    }
    const out = kalmanBbox(s);
    out.forEach((v, i) => expect(v).toBeCloseTo(target[i], 3));
  });

  it('smooths injected noise (output amplitude < input amplitude)', () => {
    const t0 = 1_000_000;
    const truth: [number, number, number, number] = [0.3, 0.4, 0.6, 0.7];
    let s = initBboxKalman(truth, t0);
    const input_amplitudes: number[] = [];
    const output_amplitudes: number[] = [];
    for (let i = 1; i <= 30; i++) {
      // Inject ±0.02 noise on each coord
      const noise = (Math.sin(i * 1.7) * 0.02);
      const noisy: [number, number, number, number] = [
        truth[0] + noise, truth[1] + noise, truth[2] + noise, truth[3] + noise,
      ];
      input_amplitudes.push(Math.abs(noise));
      s = stepBboxKalman(s, noisy, t0 + i * 1200);
      output_amplitudes.push(Math.abs(kalmanBbox(s)[0] - truth[0]));
    }
    const avgIn  = input_amplitudes.reduce((a, b) => a + b, 0) / input_amplitudes.length;
    const avgOut = output_amplitudes.reduce((a, b) => a + b, 0) / output_amplitudes.length;
    expect(avgOut).toBeLessThan(avgIn);
  });

  it('follows real motion within a few frames', () => {
    const t0 = 1_000_000;
    let s = initBboxKalman([0.1, 0.2, 0.3, 0.4], t0);
    // Bbox slides linearly to a new position
    const target: [number, number, number, number] = [0.5, 0.6, 0.7, 0.8];
    for (let i = 1; i <= 15; i++) {
      s = stepBboxKalman(s, target, t0 + i * 1200);
    }
    const out = kalmanBbox(s);
    // Should be within 5% of the target after 15 frames at the new position
    out.forEach((v, i) => expect(Math.abs(v - target[i])).toBeLessThan(0.05));
  });

  it('caps dt so a long pause does not explode the velocity term', () => {
    const t0 = 1_000_000;
    let s = initBboxKalman([0.3, 0.4, 0.6, 0.7], t0);
    // Several normal updates first
    for (let i = 1; i <= 5; i++) {
      s = stepBboxKalman(s, [0.3 + i * 0.01, 0.4, 0.6, 0.7], t0 + i * 1200);
    }
    // Then a 30-second gap (pause / app backgrounded). The implementation
    // caps internal dt so the prediction step can't extrapolate wildly.
    const sAfter = stepBboxKalman(s, [0.3, 0.4, 0.6, 0.7], t0 + 30_000_000);
    const out = kalmanBbox(sAfter);
    // Output stays within the [0,1] bbox space
    out.forEach((v) => {
      expect(v).toBeGreaterThan(-0.5);
      expect(v).toBeLessThan(1.5);
    });
  });
});
