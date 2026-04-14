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

export interface FeedbackStats {
  totalCorrections: number;
  agreementCount: number;
  topConfusedPairs: ConfusionPair[];
  partsWithFeedback: number;
  imagesSaved: number;
  pendingTraining: number;
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
    scan_id:             params.scanId,
    predicted_part_num:  params.predictedPartNum,
    correct_part_num:    params.correctPartNum,
    correct_color_id:    params.correctColorId ?? null,
    correct_color_name:  params.correctColorName ?? null,
    confidence:          params.confidence,
    source:              params.source,
    image_base64:        params.imageBase64 ?? null,
  });
  return {
    saved:             res.data.saved,
    willImproveModel:  res.data.will_improve_model,
    feedbackId:        res.data.feedback_id,
  };
}

/** Fetch aggregate feedback stats for the history screen counter. */
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
  };
}
