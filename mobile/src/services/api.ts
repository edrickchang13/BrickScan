import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios';
import * as SecureStore from 'expo-secure-store';
import { getApiBaseUrl } from '@/constants/config';
import {
  Part,
  InventoryItem,
  SetSummary,
  SetDetail,
  ScanResult,
  VideoScanResult,
  MultiPieceScanResult,
  BuildCheckResult,
  SetCompletionResult,
  ApiResponse,
  PartSubstitute,
  InventoryValueResult,
} from '@/types';

// ---------------------------------------------------------------------------
// Pile Scan Types
// ---------------------------------------------------------------------------
export interface PileResult {
  partNum: string;
  partName: string;
  count: number;
  confidence: number;
  colorId?: number;
  colorName?: string;
  cropImageBase64?: string;
}

// ---------------------------------------------------------------------------
// Dynamic API host detection
// ---------------------------------------------------------------------------
const API_BASE_URL = getApiBaseUrl();
const TOKEN_KEY = 'brickscan_token';

// ---------------------------------------------------------------------------
// Retry helper – retries transient errors (network / 5xx) with back-off
// ---------------------------------------------------------------------------
async function withRetry<T>(
  fn: () => Promise<T>,
  { retries = 2, baseDelay = 1000 }: { retries?: number; baseDelay?: number } = {},
): Promise<T> {
  let lastError: any;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err: any) {
      lastError = err;
      const status = err?.response?.status;
      const isRetryable =
        !status || // network error (no response)
        status === 502 ||
        status === 503 ||
        status === 504 ||
        err.code === 'ECONNABORTED' || // timeout
        err.code === 'ERR_NETWORK';
      if (!isRetryable || attempt === retries) throw err;
      const delay = baseDelay * Math.pow(2, attempt);
      console.log(`[BrickScan] Retry ${attempt + 1}/${retries} in ${delay}ms…`);
      await new Promise<void>((r) => setTimeout(r, delay));
    }
  }
  throw lastError;
}

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.client.interceptors.request.use(
      async (config) => {
        const token = await SecureStore.getItemAsync(TOKEN_KEY);
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error),
    );

    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          await SecureStore.deleteItemAsync(TOKEN_KEY);
        }
        return Promise.reject(error);
      },
    );
  }

  // ------------------------------------------------------------------
  // Health / connectivity
  // ------------------------------------------------------------------
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.client.get('/health', { timeout: 5000 });
      return response.data?.status === 'ok';
    } catch {
      return false;
    }
  }

  getBaseUrl(): string {
    return API_BASE_URL;
  }

  // ------------------------------------------------------------------
  // Auth endpoints
  // ------------------------------------------------------------------
  async login(email: string, password: string): Promise<{ token: string; user: { id: string; email: string } }> {
    const response = await this.client.post('/auth/login', { email, password });
    return response.data;
  }

  async register(email: string, password: string): Promise<{ token: string; user: { id: string; email: string } }> {
    const response = await this.client.post('/auth/register', { email, password });
    return response.data;
  }

  async logout(): Promise<void> {
    await SecureStore.deleteItemAsync(TOKEN_KEY);
  }

  // ------------------------------------------------------------------
  // Part endpoints
  // ------------------------------------------------------------------
  async searchParts(query: string): Promise<Part[]> {
    return withRetry(async () => {
      const response = await this.client.get('/parts/search', { params: { q: query } });
      return response.data.data || [];
    });
  }

  async getPartByNum(partNum: string): Promise<Part> {
    return withRetry(async () => {
      const response = await this.client.get(`/parts/${partNum}`);
      return response.data.data;
    });
  }

  // ------------------------------------------------------------------
  // Set endpoints — backed by local Rebrickable CSV cache
  // ------------------------------------------------------------------
  async searchSets(query: string, theme?: string): Promise<SetSummary[]> {
    return withRetry(async () => {
      const response = await this.client.get('/api/local-inventory/sets', {
        params: { q: query || '', theme: theme || '', limit: 48 },
      });
      return Array.isArray(response.data) ? response.data : [];
    });
  }

  async getSet(setNum: string): Promise<SetDetail> {
    return withRetry(async () => {
      const response = await this.client.get(`/api/local-inventory/sets/${setNum}`);
      return response.data;
    });
  }

  async getSetsStatus(): Promise<{
    sets_ready: boolean;
    parts_ready: boolean;
    downloading: boolean;
    sets_count: number;
    error: string | null;
  }> {
    return withRetry(async () => {
      const response = await this.client.get('/api/local-inventory/sets/status');
      return response.data;
    });
  }

  async getSetParts(setNum: string): Promise<any[]> {
    return withRetry(async () => {
      const response = await this.client.get(`/api/local-inventory/sets/${setNum}`);
      return response.data?.parts || [];
    });
  }

  // ------------------------------------------------------------------
  // BrickLink image helper
  // ------------------------------------------------------------------
  // BrickLink color 11 = Black — a neutral that exists for virtually
  // every catalogued part.  Color 0 (default) returns 404.
  private partImageUrl(partNum: string): string {
    return `https://img.bricklink.com/ItemImage/PN/11/${partNum}.png`;
  }

  // Helper to map snake_case API responses to InventoryItem
  private mapLocalItem(item: any): InventoryItem {
    return {
      id: item.id,
      partNum: item.part_num,
      partName: item.part_name || item.part_num,
      colorId: String(item.color_id ?? ''),
      colorName: item.color_name || '',
      colorHex: item.color_hex ? (item.color_hex.startsWith('#') ? item.color_hex : '#' + item.color_hex) : '',
      quantity: item.quantity,
      imageUrl: item.part_num ? this.partImageUrl(item.part_num) : undefined,
      createdAt: item.created_at || new Date().toISOString(),
      updatedAt: item.updated_at || new Date().toISOString(),
    };
  }

  // ------------------------------------------------------------------
  // Inventory endpoints
  // ------------------------------------------------------------------
  async getInventory(): Promise<InventoryItem[]> {
    return withRetry(async () => {
      const response = await this.client.get('/api/local-inventory/inventory');
      return (response.data || []).map((item: any) => this.mapLocalItem(item));
    });
  }

  async addToInventory(
    partNum: string,
    colorId: string,
    quantity: number,
    colorName?: string,
    colorHex?: string,
  ): Promise<InventoryItem> {
    return withRetry(async () => {
      const response = await this.client.post('/api/local-inventory/inventory/add', {
        part_num: partNum,
        color_id: colorId ? parseInt(colorId) || null : null,
        color_name: colorName || null,
        color_hex: colorHex || null,
        quantity,
      });
      return this.mapLocalItem(response.data);
    });
  }

  async updateInventory(id: string, quantity: number): Promise<InventoryItem> {
    return withRetry(async () => {
      const response = await this.client.put(`/api/local-inventory/inventory/${id}`, { quantity });
      return this.mapLocalItem(response.data);
    });
  }

  async deleteFromInventory(id: string): Promise<void> {
    await this.client.delete(`/api/local-inventory/inventory/${id}`);
  }

  async checkInventoryDuplicate(partNum: string, colorId: string): Promise<{ exists: boolean; quantity: number; inventory_part_id: string | null }> {
    return withRetry(async () => {
      const response = await this.client.get('/api/inventory/check', {
        params: { part_num: partNum, color_id: colorId },
      });
      return response.data;
    });
  }

  async exportInventory(): Promise<string> {
    return withRetry(async () => {
      const response = await this.client.get('/api/local-inventory/inventory/export');
      return response.data;
    });
  }

  // ------------------------------------------------------------------
  // Scan endpoint — longer timeout for image processing + ML inference
  // Supports optional depth file for RGBD input on LiDAR-equipped devices
  // ------------------------------------------------------------------
  async scanImage(imageBase64: string, depthFilePath?: string): Promise<ScanResult> {
    return withRetry(
      async () => {
        // If no depth file, use simple JSON POST
        if (!depthFilePath) {
          const response = await this.client.post(
            '/api/local-inventory/scan',
            { image_base64: imageBase64 },
            { timeout: 60000 }, // 60 s – image upload + ML inference can be slow
          );
          const data = response.data;
          const predictions = (data.predictions || []).map((p: any) => ({
            partNum: p.part_num || '',
            partName: p.part_name || p.part_num || 'Unknown Part',
            colorId: String(p.color_id ?? ''),
            colorName: p.color_name || '',
            colorHex: p.color_hex ? (p.color_hex.startsWith('#') ? p.color_hex : '#' + p.color_hex) : '',
            confidence: p.confidence || 0,
            imageUrl: p.image_url || (p.part_num ? this.partImageUrl(p.part_num) : undefined),
            source: p.source || undefined,
          }));
          return { predictions };
        }

        // With depth file, use multipart form data
        const FormData = (globalThis as any).FormData || require('form-data');
        const formData = new FormData();
        formData.append('file', {
          uri: `data:image/jpeg;base64,${imageBase64}`,
          name: 'image.jpg',
          type: 'image/jpeg',
        });

        // Append depth file if provided
        formData.append('depth_file', {
          uri: `file://${depthFilePath}`,
          name: 'depth.png',
          type: 'image/png',
        });

        const response = await this.client.post(
          '/api/scan',
          formData,
          {
            timeout: 60000,
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          },
        );

        const data = response.data;
        const predictions = (data.predictions || []).map((p: any) => ({
          partNum: p.part_num || '',
          partName: p.part_name || p.part_num || 'Unknown Part',
          colorId: String(p.color_id ?? ''),
          colorName: p.color_name || '',
          colorHex: p.color_hex ? (p.color_hex.startsWith('#') ? p.color_hex : '#' + p.color_hex) : '',
          confidence: p.confidence || 0,
          imageUrl: p.image_url || (p.part_num ? this.partImageUrl(p.part_num) : undefined),
          source: p.source || undefined,
        }));
        return { predictions };
      },
      { retries: 1, baseDelay: 2000 }, // one retry for scans
    );
  }

  // ------------------------------------------------------------------
  // Part info / enrichment from Rebrickable CSV catalog
  // ------------------------------------------------------------------
  async getPartInfo(partNum: string): Promise<{ partNum: string; partName: string; categoryName?: string }> {
    try {
      const response = await this.client.get(`/api/local-inventory/parts/info/${encodeURIComponent(partNum)}`, { timeout: 5000 });
      const d = response.data;
      return {
        partNum: d.part_num || partNum,
        partName: d.part_name || partNum,
        categoryName: d.category_name || undefined,
      };
    } catch {
      return { partNum, partName: partNum };
    }
  }

  async searchPartsLocal(query: string, limit = 20): Promise<Array<{ partNum: string; partName: string; categoryId?: string }>> {
    try {
      const response = await this.client.get('/api/local-inventory/parts/search', {
        params: { q: query, limit },
        timeout: 5000,
      });
      return (response.data || []).map((p: any) => ({
        partNum: p.part_num,
        partName: p.part_name,
        categoryId: p.category_id,
      }));
    } catch {
      return [];
    }
  }

  // ------------------------------------------------------------------
  // Part substitutes endpoint
  // ------------------------------------------------------------------
  async getPartSubstitutes(partNum: string): Promise<PartSubstitute[]> {
    return withRetry(async () => {
      const response = await this.client.get(`/api/parts/${partNum}/substitutes`, { timeout: 10000 });
      return (response.data || []).map((s: any) => ({
        partNum: s.part_num,
        name: s.name,
        similarity: s.similarity || 0,
        reason: s.reason || '',
        imageUrl: s.image_url || '',
      }));
    }, { retries: 1, baseDelay: 1000 });
  }

  // ------------------------------------------------------------------
  // Video scan endpoint — multi-frame analysis with voting
  // ------------------------------------------------------------------
  async scanVideo(framesBase64: string[]): Promise<VideoScanResult> {
    return withRetry(
      async () => {
        const response = await this.client.post(
          '/api/local-inventory/scan-video',
          { frames: framesBase64 },
          { timeout: 120000 }, // 2 min — multiple frames × ML inference
        );
        const data = response.data;
        const predictions = (data.predictions || []).map((p: any) => ({
          partNum: p.part_num || '',
          partName: p.part_name || p.part_num || 'Unknown Part',
          colorId: String(p.color_id ?? ''),
          colorName: p.color_name || '',
          colorHex: p.color_hex ? (p.color_hex.startsWith('#') ? p.color_hex : '#' + p.color_hex) : '',
          confidence: p.confidence || 0,
          imageUrl: p.image_url || (p.part_num ? this.partImageUrl(p.part_num) : undefined),
          source: p.source || undefined,
        }));
        return {
          predictions,
          framesAnalyzed: data.frames_analyzed || 0,
          agreementScore: data.agreement_score || 0,
          status: data.status || 'unknown',
        };
      },
      { retries: 1, baseDelay: 2000 },
    );
  }

  // ------------------------------------------------------------------
  // Multi-piece scan endpoint — detect multiple pieces in one image
  // ------------------------------------------------------------------
  async scanMultiPiece(imageBase64: string): Promise<MultiPieceScanResult> {
    return withRetry(
      async () => {
        const response = await this.client.post(
          '/api/local-inventory/scan-multi',
          { image_base64: imageBase64 },
          { timeout: 120000 },
        );
        const data = response.data;
        const pieces = (data.pieces || []).map((piece: any) => ({
          pieceIndex: piece.piece_index,
          predictions: (piece.predictions || []).map((p: any) => ({
            partNum: p.part_num || '',
            partName: p.part_name || p.part_num || 'Unknown Part',
            colorId: String(p.color_id ?? ''),
            colorName: p.color_name || '',
            colorHex: p.color_hex ? (p.color_hex.startsWith('#') ? p.color_hex : '#' + p.color_hex) : '',
            confidence: p.confidence || 0,
            imageUrl: p.image_url || (p.part_num ? this.partImageUrl(p.part_num) : undefined),
            source: p.source || undefined,
          })),
          primaryPrediction: {
            partNum: piece.primary_prediction?.part_num || '',
            partName: piece.primary_prediction?.part_name || '',
            colorId: String(piece.primary_prediction?.color_id ?? ''),
            colorName: piece.primary_prediction?.color_name || '',
            colorHex: piece.primary_prediction?.color_hex || '',
            confidence: piece.primary_prediction?.confidence || 0,
            imageUrl: piece.primary_prediction?.image_url || undefined,
            source: piece.primary_prediction?.source || undefined,
          },
          bbox: piece.bbox || undefined,
        }));
        return {
          piecesDetected: data.pieces_detected || 0,
          pieces,
          status: data.status || 'unknown',
        };
      },
      { retries: 1, baseDelay: 2000 },
    );
  }

  // ------------------------------------------------------------------
  // Pile scan endpoint — detect and classify multiple bricks in one image
  // ------------------------------------------------------------------
  async scanPile(imageBase64: string): Promise<PileResult[]> {
    return withRetry(
      async () => {
        const response = await this.client.post(
          '/api/scan/pile',
          { image_base64: imageBase64 },
          { timeout: 120000 }, // 2 min – full image detection + classification
        );
        const data = response.data;
        const results = (data || []).map((r: any) => ({
          partNum: r.part_num || '',
          partName: r.part_name || r.part_num || 'Unknown Part',
          count: r.count || 0,
          confidence: r.confidence || 0,
          colorId: r.color_id ? Number(r.color_id) : undefined,
          colorName: r.color_name || undefined,
          cropImageBase64: r.crop_image_base64 || undefined,
        }));
        return results;
      },
      { retries: 1, baseDelay: 2000 },
    );
  }

  // ------------------------------------------------------------------
  // Build check endpoint
  // ------------------------------------------------------------------
  async compareToSet(setNum: string): Promise<BuildCheckResult> {
    return withRetry(async () => {
      const response = await this.client.post('/api/local-inventory/builds/check', { setNum });
      return response.data;
    });
  }

  // ------------------------------------------------------------------
  // BrickLink endpoint
  // ------------------------------------------------------------------
  async generateWantedList(setNum: string, condition: string = 'N'): Promise<string> {
    return withRetry(async () => {
      const response = await this.client.post(`/bricklink/wanted-list/${setNum}`, null, { params: { condition } });
      return response.data;
    });
  }

  // ------------------------------------------------------------------
  // Set completion endpoints
  // ------------------------------------------------------------------
  async getSetCompletion(setNum: string): Promise<SetCompletionResult> {
    return withRetry(async () => {
      const response = await this.client.get(`/api/sets/${setNum}/completion`);
      return response.data;
    });
  }

  async scanInventoryForSets(): Promise<SetCompletionResult[]> {
    return withRetry(async () => {
      const response = await this.client.get('/api/sets/completion/scan');
      return Array.isArray(response.data) ? response.data : [];
    });
  }

  // ------------------------------------------------------------------
  // Inventory value endpoint
  // ------------------------------------------------------------------
  async getInventoryValue(): Promise<InventoryValueResult> {
    return withRetry(async () => {
      const response = await this.client.get('/api/inventory/value');
      return response.data;
    });
  }
}

export const apiClient = new ApiClient();
