/**
 * Feedback API — active learning corrections and stats.
 *
 * HOW TO USE IN ScanResultScreen.tsx:
 * ------------------------------------
 * import { submitFeedback, getFeedbackStats } from '@/services/feedbackApi';
 *
 * // After a scan:
 * await submitFeedback({
 *   scanId,
 *   predictedPartNum: primary.partNum,
 *   correctPartNum: correctedPartNum,
 *   confidence: primary.confidence,
 *   source: primary.source ?? 'unknown',
 *   imageBase64: capturedImageBase64,   // optional
 * });
 *
 * // In ScanHistoryScreen to show the counter:
 * const stats = await getFeedbackStats();
 * // stats.totalCorrections, stats.agreementCount, etc.
 */

import axios from 'axios';
import { NativeModules } from 'react-native';

// ---------------------------------------------------------------------------
// API base URL (mirrors the logic in api.ts)
// ---------------------------------------------------------------------------
function getApiBase(): string {
  if (__DEV__) {
    const scriptURL: string | undefined = NativeModules.SourceCode?.scriptURL;
    if (scriptURL) {
      try { return `http://${new URL(scriptURL).hostname}:8000`; } catch {}
    }
  }
  return (globalThis as any).process?.env?.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000';
}

const API_BASE = getApiBase();

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * v2 feedback taxonomy. Mapped 1:1 to the backend enum.
 *   top_correct          — top pick was right              (rank 0)
 *   alternative_correct  — one of the other predictions was right (rank 1..N)
 *   none_correct         — user searched for the right part (not shown) (rank -1)
 *   partially_correct    — right brick, wrong colour
 */
export type FeedbackType =
  | 'top_correct'
  | 'alternative_correct'
  | 'none_correct'
  | 'partially_correct';

export interface PredictionShown {
  partNum: string;
  partName?: string;
  confidence: number;
  source?: string;
  colorId?: string;
  colorHex?: string;
}

export interface SubmitFeedbackParams {
  scanId: string;
  predictedPartNum: string;
  correctPartNum: string;
  correctColorId?: string;
  correctColorName?: string;
  confidence: number;
  source: string;
  /** Base64 JPEG of the scan image. Include for mis-predictions to build training set. */
  imageBase64?: string;

  /** v2: explicit three-way signal. Backend derives from part_num comparison when absent. */
  feedbackType?: FeedbackType;
  /** v2: position of the correct answer in the shown top-5 (0..N), or -1 if not shown. */
  correctRank?: number;
  /** v2: full top-5 that was rendered (for confusion analysis / per-source accuracy). */
  predictionsShown?: PredictionShown[];
  /** v2: how long the user deliberated, in milliseconds. */
  timeToConfirmMs?: number;
}

export interface FeedbackResult {
  saved: boolean;
  willImproveModel: boolean;
  feedbackId: string;
}

export interface ConfusionPair {
  predictedPartNum: string;
  correctPartNum: string;
  count: number;
}

export interface SourceStats {
  source: string;
  count: number;
  correct: number;
  accuracy: number;
}

export interface AccuracyTrendPoint {
  weekEnding: string;
  top1Accuracy: number;
  top3Accuracy: number;
  sampleSize: number;
}

export interface FeedbackStats {
  totalCorrections: number;
  agreementCount: number;
  topConfusedPairs: ConfusionPair[];
  partsWithFeedback: number;
  imagesSaved: number;
  pendingTraining: number;

  // v2
  top1Accuracy: number;
  top3Accuracy: number;
  bySource: SourceStats[];
  accuracyTrend: AccuracyTrendPoint[];
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

/** Submit a user correction (or confirmation) after a scan. */
export async function submitFeedback(
  params: SubmitFeedbackParams,
): Promise<FeedbackResult> {
  const res = await axios.post<{
    saved: boolean;
    will_improve_model: boolean;
    feedback_id: string;
  }>(`${API_BASE}/api/local-inventory/scan-feedback`, {
    scan_id:                params.scanId,
    predicted_part_num:     params.predictedPartNum,
    correct_part_num:       params.correctPartNum,
    correct_color_id:       params.correctColorId ?? null,
    correct_color_name:     params.correctColorName ?? null,
    confidence:             params.confidence,
    source:                 params.source,
    image_base64:           params.imageBase64 ?? null,
    feedback_type:          params.feedbackType ?? null,
    correct_rank:           params.correctRank ?? null,
    predictions_shown:      params.predictionsShown
      ? params.predictionsShown.map(p => ({
          part_num:  p.partNum,
          part_name: p.partName,
          confidence: p.confidence,
          source:    p.source,
          color_id:  p.colorId,
          color_hex: p.colorHex,
        }))
      : null,
    time_to_confirm_ms:     params.timeToConfirmMs ?? null,
  });
  return {
    saved:             res.data.saved,
    willImproveModel:  res.data.will_improve_model,
    feedbackId:        res.data.feedback_id,
  };
}

/** Fetch aggregate feedback stats for the history screen counter + stats dashboard. */
export async function getFeedbackStats(): Promise<FeedbackStats> {
  const res = await axios.get<{
    total_corrections: number;
    agreement_count: number;
    top_confused_pairs: Array<{
      predicted_part_num: string;
      correct_part_num: string;
      count: number;
    }>;
    parts_with_feedback: number;
    images_saved: number;
    pending_training: number;
    top1_accuracy?: number;
    top3_accuracy?: number;
    by_source?: Array<{
      source: string; count: number; correct: number; accuracy: number;
    }>;
    accuracy_trend?: Array<{
      week_ending: string; top1_accuracy: number; top3_accuracy: number; sample_size: number;
    }>;
  }>(`${API_BASE}/api/local-inventory/feedback/stats`);

  return {
    totalCorrections:   res.data.total_corrections,
    agreementCount:     res.data.agreement_count,
    topConfusedPairs:   res.data.top_confused_pairs.map(p => ({
      predictedPartNum: p.predicted_part_num,
      correctPartNum:   p.correct_part_num,
      count:            p.count,
    })),
    partsWithFeedback:  res.data.parts_with_feedback,
    imagesSaved:        res.data.images_saved,
    pendingTraining:    res.data.pending_training,
    top1Accuracy:       res.data.top1_accuracy ?? 0,
    top3Accuracy:       res.data.top3_accuracy ?? 0,
    bySource:           (res.data.by_source ?? []).map(s => ({
      source:    s.source,
      count:     s.count,
      correct:   s.correct,
      accuracy:  s.accuracy,
    })),
    accuracyTrend:      (res.data.accuracy_trend ?? []).map(p => ({
      weekEnding:       p.week_ending,
      top1Accuracy:     p.top1_accuracy,
      top3Accuracy:     p.top3_accuracy,
      sampleSize:       p.sample_size,
    })),
  };
}

/**
 * Helper: submit "user tapped alternative rank N" feedback.
 * Called by ScanResultScreen when the user taps one of the non-top prediction cards.
 */
export async function submitAlternativeFeedback(args: {
  scanId: string;
  predictedPartNum: string;
  predictedConfidence: number;
  predictedSource: string;
  chosen: { partNum: string; partName?: string };
  chosenRank: number;              // 1 for the 2nd card, 2 for the 3rd
  predictionsShown: PredictionShown[];
  timeToConfirmMs?: number;
  imageBase64?: string;
}): Promise<FeedbackResult> {
  return submitFeedback({
    scanId:           args.scanId,
    predictedPartNum: args.predictedPartNum,
    correctPartNum:   args.chosen.partNum,
    confidence:       args.predictedConfidence,
    source:           args.predictedSource,
    feedbackType:     'alternative_correct',
    correctRank:      args.chosenRank,
    predictionsShown: args.predictionsShown,
    timeToConfirmMs:  args.timeToConfirmMs,
    imageBase64:      args.imageBase64,
  });
}
