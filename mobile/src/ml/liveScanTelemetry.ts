/**
 * Phase 4 OUTER-loop telemetry recorder for the on-device live scan.
 *
 * The closed-loop eval is two loops sharing ONE telemetry schema:
 *   - INNER loop (no phone): ml/scripts/livescan_harness.py replays recorded
 *     frames through a Python mirror of the pipeline and emits
 *     `brickscan.livescan.telemetry/v1`.
 *   - OUTER loop (this file): the real app, on a tethered iPhone, records the
 *     SAME schema for a live sweep so the device run is observable and diffable
 *     against the inner-loop baseline.
 *
 * This module is the device-side recorder. It is:
 *   - DEBUG-FLAG-GATED: a session is created only when SETTINGS_KEYS.liveScanTelemetry
 *     is on. When off, every method is a cheap no-op (the module-level singleton
 *     stays `enabled=false`), so there is ZERO overhead on the hot frame path in
 *     normal use.
 *   - PURE w.r.t. its own state: no React, no native imports. Sinks (file write,
 *     backend POST) are injected by the caller (ContinuousScanScreen) so this
 *     stays unit-testable and mirrors the framework-free style of trackFusion.ts.
 *
 * Schema parity with the inner-loop harness (`schema` / `source` differ; the
 * per-track + per-frame shapes match field-for-field where the device can
 * supply them):
 *   { schema, source:"outer_loop_device", meta{…}, aggregate{…}, tracks[ {
 *       piece_id?, n_frames, frames[ { frame, top5, maxSim, fused_top5,
 *         committed_here, latency_ms, color? } ],
 *       fused_top1, committed, commit_frame, color, latency_ms{…},
 *       events[ {kind, at, …} ] } ] }
 *
 * `piece_id` (ground truth) is unknown on a live sweep, so accuracy fields the
 * harness computes (correct/purity) are left null here; the device telemetry is
 * for latency, commit behaviour, prediction drift and error surfacing. When the
 * user scans a KNOWN piece they can tag it via `setExpectedPart` to recover
 * accuracy on the device too.
 */

import { DEFAULT_FUSION_OPTS, type FusedMatch } from '@/utils/trackFusion';

/** A single retrieval hit recorded into telemetry. */
export interface TelemetryMatch {
  partNum: string;
  score: number;
}

/** One processed frame for a track (mirrors the harness per-frame record). */
export interface TelemetryFrame {
  frame: number;
  /** Single-frame top-k (best first). */
  top5: TelemetryMatch[];
  /** This frame's single-frame max similarity (top-1 score). */
  maxSim: number;
  /** Fused top-k after folding this frame in (best first). */
  fused_top5: TelemetryMatch[];
  /** Whether the track became committed on this frame. */
  committed_here: boolean;
  /** Per-frame processing latency in ms (crop+embed+retrieve+fuse). */
  latency_ms: number;
  /** Best colour id for this frame's crop, when classified. */
  colorId?: string;
  colorName?: string;
  /** Set if processing this frame threw (best-effort path still records it). */
  error?: string;
}

/** A discrete event in a track's life (commit, inventory add, server correction). */
export interface TelemetryEvent {
  kind:
    | 'committed'
    | 'inventory_add'
    | 'inventory_add_failed'
    | 'server_verify_start'
    | 'server_verify_result'
    | 'server_correction'
    | 'error';
  /** ms since the session started. */
  at: number;
  partNum?: string;
  colorId?: string;
  detail?: string;
}

interface TrackRecord {
  piece_id: string | null;
  frames: TelemetryFrame[];
  events: TelemetryEvent[];
  commit_frame: number | null;
  firstSeenAt: number;
}

/** The serialized telemetry document — same top-level shape as the harness. */
export interface LiveScanTelemetryDoc {
  schema: 'brickscan.livescan.telemetry/v1';
  source: 'outer_loop_device';
  meta: {
    generated_at: string;
    session_id: string;
    platform: string;
    app_started_at: string;
    embedding_ready: boolean;
    gallery_size: number | null;
    fusion_opts: Record<string, number>;
    device?: Record<string, unknown>;
  };
  aggregate: {
    n_tracks: number;
    n_frames: number;
    commit_rate: number | null;
    mean_frames_to_commit: number | null;
    /** fused top-1 accuracy over tracks that had an expected part tagged. */
    fused_top1: number | null;
    n_tagged: number;
    retrieval_latency_ms: {
      mean: number;
      p50: number;
      p90: number;
      p99: number;
      max: number;
    };
    errors: number;
  };
  tracks: Array<{
    track_id: string;
    piece_id: string | null;
    n_frames: number;
    frames: TelemetryFrame[];
    fused_top1: string;
    fused_top5: TelemetryMatch[];
    committed: boolean;
    commit_frame: number | null;
    fused_correct: boolean | null;
    color: { colorId: string; colorName: string } | null;
    latency_ms: { mean: number; max: number };
    events: TelemetryEvent[];
  }>;
}

/** A sink that persists a finished telemetry doc (file write, network POST…). */
export type TelemetrySink = (doc: LiveScanTelemetryDoc) => Promise<void>;

function percentile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const pos = ((sorted.length - 1) * q) / 100;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  const frac = pos - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

function round(n: number, dp = 4): number {
  const f = 10 ** dp;
  return Math.round(n * f) / f;
}

/**
 * Per-session live-scan telemetry recorder. One instance per scan session.
 * Disabled instances are a no-op on every method. Keyed by the tracker's stable
 * track id; the caller records one frame per `processTrackFrame` and discrete
 * events as commits / inventory adds / server corrections happen.
 */
export class LiveScanTelemetry {
  private readonly tracks = new Map<string, TrackRecord>();
  private readonly startedAtMs: number;
  private readonly sessionId: string;
  private platform = 'unknown';
  private embeddingReady = false;
  private gallerySize: number | null = null;
  private device: Record<string, unknown> | undefined;
  private errorCount = 0;
  public enabled: boolean;

  constructor(enabled: boolean, opts?: { platform?: string; now?: number }) {
    this.enabled = enabled;
    this.startedAtMs = opts?.now ?? Date.now();
    this.platform = opts?.platform ?? 'unknown';
    this.sessionId = `${this.platform}-${this.startedAtMs}`;
  }

  /** Record one-time session context (called once after the engine loads). */
  setContext(ctx: {
    embeddingReady?: boolean;
    gallerySize?: number | null;
    device?: Record<string, unknown>;
  }): void {
    if (!this.enabled) return;
    if (ctx.embeddingReady !== undefined) this.embeddingReady = ctx.embeddingReady;
    if (ctx.gallerySize !== undefined) this.gallerySize = ctx.gallerySize;
    if (ctx.device !== undefined) this.device = ctx.device;
  }

  /**
   * Tag a track with its known ground-truth part (when scanning a labelled
   * reference piece). Enables fused-top-1 accuracy on the device, matching the
   * harness's `fused_correct`. No-op for live sweeps where the part is unknown.
   */
  setExpectedPart(trackId: string, partNum: string): void {
    if (!this.enabled) return;
    this.ensure(trackId).piece_id = partNum;
  }

  private ensure(trackId: string): TrackRecord {
    let rec = this.tracks.get(trackId);
    if (!rec) {
      rec = { piece_id: null, frames: [], events: [], commit_frame: null, firstSeenAt: Date.now() };
      this.tracks.set(trackId, rec);
    }
    return rec;
  }

  /**
   * Record one processed frame for a track. `committed` is the engine's commit
   * verdict for THIS frame; the first frame where it flips true is the
   * commit_frame (1-indexed, matching the harness).
   */
  recordFrame(
    trackId: string,
    args: {
      top: FusedMatch[];
      maxSim: number;
      fused: FusedMatch[];
      committed: boolean;
      latencyMs: number;
      colorId?: string;
      colorName?: string;
      error?: string;
    },
  ): void {
    if (!this.enabled) return;
    const rec = this.ensure(trackId);
    const idx = rec.frames.length;
    if (args.error) this.errorCount++;
    rec.frames.push({
      frame: idx,
      top5: args.top.map((m) => ({ partNum: m.partNum, score: round(m.score) })),
      maxSim: round(args.maxSim),
      fused_top5: args.fused.map((m) => ({ partNum: m.partNum, score: round(m.score) })),
      committed_here: args.committed,
      latency_ms: round(args.latencyMs, 3),
      ...(args.colorId ? { colorId: args.colorId, colorName: args.colorName } : {}),
      ...(args.error ? { error: args.error } : {}),
    });
    if (args.committed && rec.commit_frame === null) {
      rec.commit_frame = idx + 1;
      rec.events.push({ kind: 'committed', at: this.elapsed(), partNum: args.fused[0]?.partNum });
    }
  }

  /** Record a discrete track event (inventory add, server correction, error…). */
  recordEvent(trackId: string, event: Omit<TelemetryEvent, 'at'> & { at?: number }): void {
    if (!this.enabled) return;
    if (event.kind === 'error' || event.kind === 'inventory_add_failed') this.errorCount++;
    this.ensure(trackId).events.push({ ...event, at: event.at ?? this.elapsed() });
  }

  /** Drop a track entirely (e.g. tracker GC). Keeps the doc bounded in long sweeps. */
  forget(trackId: string): void {
    if (!this.enabled) return;
    this.tracks.delete(trackId);
  }

  private elapsed(): number {
    return Date.now() - this.startedAtMs;
  }

  /** True if any frame was recorded — used to skip writing an empty session. */
  hasData(): boolean {
    return this.enabled && this.tracks.size > 0;
  }

  /**
   * Materialize the telemetry document. The per-track `fused_top1` is taken from
   * the LAST frame's fused top-1 (the freshest fused estimate), matching how the
   * harness reports the final fused read.
   */
  build(): LiveScanTelemetryDoc {
    const trackDocs: LiveScanTelemetryDoc['tracks'] = [];
    const allLatencies: number[] = [];
    const framesToCommit: number[] = [];
    let committedCount = 0;
    let taggedCount = 0;
    let correctCount = 0;
    let totalFrames = 0;

    for (const [trackId, rec] of this.tracks) {
      const lastFrame = rec.frames[rec.frames.length - 1];
      const fusedTop = lastFrame?.fused_top5 ?? [];
      const fusedTop1 = fusedTop[0]?.partNum ?? '';
      const committed = rec.commit_frame !== null;
      if (committed) {
        committedCount++;
        if (rec.commit_frame !== null) framesToCommit.push(rec.commit_frame);
      }
      let fusedCorrect: boolean | null = null;
      if (rec.piece_id !== null) {
        taggedCount++;
        fusedCorrect = fusedTop1 === rec.piece_id;
        if (fusedCorrect) correctCount++;
      }
      // Colour: take the most recent frame that classified one.
      let color: { colorId: string; colorName: string } | null = null;
      for (let i = rec.frames.length - 1; i >= 0; i--) {
        const f = rec.frames[i];
        if (f.colorId) {
          color = { colorId: f.colorId, colorName: f.colorName ?? '' };
          break;
        }
      }
      const lats = rec.frames.map((f) => f.latency_ms);
      for (const l of lats) allLatencies.push(l);
      totalFrames += rec.frames.length;

      trackDocs.push({
        track_id: trackId,
        piece_id: rec.piece_id,
        n_frames: rec.frames.length,
        frames: rec.frames,
        fused_top1: fusedTop1,
        fused_top5: fusedTop,
        committed,
        commit_frame: rec.commit_frame,
        fused_correct: fusedCorrect,
        color,
        latency_ms: {
          mean: lats.length ? round(lats.reduce((a, b) => a + b, 0) / lats.length, 3) : 0,
          max: lats.length ? round(Math.max(...lats), 3) : 0,
        },
        events: rec.events,
      });
    }

    const sortedLat = [...allLatencies].sort((a, b) => a - b);
    const n = trackDocs.length;
    return {
      schema: 'brickscan.livescan.telemetry/v1',
      source: 'outer_loop_device',
      meta: {
        generated_at: new Date().toISOString(),
        session_id: this.sessionId,
        platform: this.platform,
        app_started_at: new Date(this.startedAtMs).toISOString(),
        embedding_ready: this.embeddingReady,
        gallery_size: this.gallerySize,
        // Same fusion constants the harness records, sourced from the single
        // on-device source of truth so device + inner-loop docs stay comparable.
        fusion_opts: {
          softmaxScale: DEFAULT_FUSION_OPTS.softmaxScale,
          topK: DEFAULT_FUSION_OPTS.topK,
          commitStability: DEFAULT_FUSION_OPTS.commitStability,
          commitMargin: DEFAULT_FUSION_OPTS.commitMargin,
          maxFrames: DEFAULT_FUSION_OPTS.maxFrames,
        },
        ...(this.device ? { device: this.device } : {}),
      },
      aggregate: {
        n_tracks: n,
        n_frames: totalFrames,
        commit_rate: n > 0 ? round((100 * committedCount) / n, 2) : null,
        mean_frames_to_commit: framesToCommit.length
          ? round(framesToCommit.reduce((a, b) => a + b, 0) / framesToCommit.length, 2)
          : null,
        fused_top1: taggedCount > 0 ? round((100 * correctCount) / taggedCount, 2) : null,
        n_tagged: taggedCount,
        retrieval_latency_ms: {
          mean: sortedLat.length
            ? round(sortedLat.reduce((a, b) => a + b, 0) / sortedLat.length, 3)
            : 0,
          p50: round(percentile(sortedLat, 50), 3),
          p90: round(percentile(sortedLat, 90), 3),
          p99: round(percentile(sortedLat, 99), 3),
          max: sortedLat.length ? round(sortedLat[sortedLat.length - 1], 3) : 0,
        },
        errors: this.errorCount,
      },
      tracks: trackDocs,
    };
  }

  /**
   * Build the doc and hand it to each sink (best-effort: a failing sink never
   * throws into the caller). Returns the doc so the caller can also inspect it.
   */
  async flush(sinks: TelemetrySink[]): Promise<LiveScanTelemetryDoc> {
    const doc = this.build();
    if (!this.enabled) return doc;
    for (const sink of sinks) {
      try {
        await sink(doc);
      } catch (err) {
        console.warn('[liveScanTelemetry] sink failed:', err);
      }
    }
    return doc;
  }
}

/**
 * Module-level singleton, DISABLED by default so the engine can call into it
 * unconditionally with zero overhead until a session opts in. ContinuousScanScreen
 * replaces it via `startTelemetrySession` when the debug flag is on.
 */
let active: LiveScanTelemetry = new LiveScanTelemetry(false);

/** The current session recorder (disabled no-op until a session starts). */
export function telemetry(): LiveScanTelemetry {
  return active;
}

/** Begin a new telemetry session (called by the screen when the flag is on). */
export function startTelemetrySession(
  enabled: boolean,
  opts?: { platform?: string },
): LiveScanTelemetry {
  active = new LiveScanTelemetry(enabled, opts);
  return active;
}

/** End the current session (swap back to a disabled no-op recorder). */
export function endTelemetrySession(): void {
  active = new LiveScanTelemetry(false);
}
