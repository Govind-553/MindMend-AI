// src/services/api.ts
// API service for MindMend backend integration

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

// Response types
export interface EmotionResult {
  label: string;
  confidence: number;
  raw_emotion?: string;
  all_probabilities?: Record<string, number>;
  mock?: boolean;
}

export interface KeystrokeResult {
  score: number;
  wpm: number;
  pauseCount: number;
  avgDwellTime: number;
  mock?: boolean;
}

export interface WellnessAnalysis {
  speechEmotion: EmotionResult | null;
  facialEmotion: (EmotionResult & { detectedFaces?: number }) | null;
  keystrokeMetrics: KeystrokeResult | null;
  wellnessIndex: number;
  timestamp: string;
  sessionId: string;
  documentId?: string;
}

export interface WellnessHistory {
  id: string;
  wellnessIndex: number;
  timestamp: string;
  speechEmotion?: EmotionResult;
  facialEmotion?: EmotionResult;
}

export interface WellnessStats {
  average: number;
  minimum: number;
  maximum: number;
  latest: number;
  data_points: number;
  period_days: number;
  trend: string;
}

// API Client Class
class ApiClient {
  private baseURL: string;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Health check
  async healthCheck() {
    return this.request<{
      status: string;
      version: string;
      services: string[];
      environment: string;
    }>('/api/health');
  }

  // Speech emotion analysis
  async analyzeSpeech(audioFile: File): Promise<EmotionResult> {
    const formData = new FormData();
    formData.append('audio', audioFile);

    const response = await fetch(`${this.baseURL}/analyze-speech`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Speech analysis failed');
    }

    return response.json();
  }

  // Facial emotion analysis
  async analyzeFace(imageFile: File | string): Promise<EmotionResult & { detectedFaces?: number }> {
    const formData = new FormData();
    
    if (typeof imageFile === 'string') {
      // Base64 image data
      const response = await fetch(`${this.baseURL}/analyze-face`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageFile }),
      });

      if (!response.ok) {
        throw new Error('Facial analysis failed');
      }

      return response.json();
    } else {
      // File upload
      formData.append('image', imageFile);

      const response = await fetch(`${this.baseURL}/analyze-face`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Facial analysis failed');
      }

      return response.json();
    }
  }

  // Keystroke analysis
  async analyzeKeystroke(keystrokes: any[]): Promise<KeystrokeResult> {
    return this.request<KeystrokeResult>('/analyze-keystroke', {
      method: 'POST',
      body: JSON.stringify({ keystrokes }),
    });
  }

  // Full wellness analysis
  async analyzeWellness(data: {
    audio?: File;
    image?: File;
    keystrokes?: any[];
    sessionId?: string;
  }): Promise<{ success: boolean; data: WellnessAnalysis }> {
    const formData = new FormData();

    if (data.audio) {
      formData.append('audio', data.audio);
    }

    if (data.image) {
      formData.append('image', data.image);
    }

    if (data.keystrokes) {
      formData.append('keystrokes', JSON.stringify(data.keystrokes));
    }

    if (data.sessionId) {
      formData.append('sessionId', data.sessionId);
    }

    const response = await fetch(`${this.baseURL}/api/wellness/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Wellness analysis failed');
    }

    return response.json();
  }

  // Get wellness history
  async getWellnessHistory(days: number = 7): Promise<{
    success: boolean;
    data: WellnessHistory[];
    count: number;
    period_days: number;
  }> {
    return this.request(`/api/history/wellness?days=${days}`);
  }

  // Get wellness statistics
  async getWellnessStats(days: number = 30): Promise<{
    success: boolean;
    data: { stats: WellnessStats };
  }> {
    return this.request(`/api/history/stats?days=${days}`);
  }

  // Quick wellness check
  async quickCheck(data: {
    moodRating: number;
    energyLevel: number;
    stressLevel: number;
    sessionId?: string;
  }): Promise<{
    success: boolean;
    data: {
      wellnessIndex: number;
      recommendation: any;
      timestamp: string;
    };
  }> {
    return this.request('/api/wellness/quick-check', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Get wellness recommendation
  async getRecommendation(wellnessIndex: number): Promise<{
    success: boolean;
    data: {
      level: string;
      score: number;
      message: string;
      suggestions: string[];
      urgency: string;
    };
  }> {
    return this.request('/api/wellness/recommendation', {
      method: 'POST',
      body: JSON.stringify({ wellnessIndex }),
    });
  }

  // Get app configuration
  async getConfig(): Promise<{
    environment: string;
    version: string;
    features: Record<string, boolean>;
    mock_mode: boolean;
  }> {
    return this.request('/api/config');
  }
}

// Export singleton instance
export const apiClient = new ApiClient(API_BASE_URL);

// Convenience exports
export const healthCheck = () => apiClient.healthCheck();
export const analyzeSpeech = (audio: File) => apiClient.analyzeSpeech(audio);
export const analyzeFace = (image: File | string) => apiClient.analyzeFace(image);
export const analyzeKeystroke = (keystrokes: any[]) => apiClient.analyzeKeystroke(keystrokes);
export const analyzeWellness = (data: any) => apiClient.analyzeWellness(data);
export const getWellnessHistory = (days?: number) => apiClient.getWellnessHistory(days);
export const getWellnessStats = (days?: number) => apiClient.getWellnessStats(days);
export const quickCheck = (data: any) => apiClient.quickCheck(data);
export const getRecommendation = (index: number) => apiClient.getRecommendation(index);
export const getConfig = () => apiClient.getConfig();