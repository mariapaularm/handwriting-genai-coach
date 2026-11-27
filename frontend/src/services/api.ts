import axios, { AxiosError, type AxiosProgressEvent } from "axios";

/** Base de la API del backend FastAPI (sin barra final). */
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://localhost:8000/api/v1"
).replace(/\/$/, "");

/** Mismo patrón que valida el backend para `profile_name`. */
export const PROFILE_NAME_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;
export const DEFAULT_PROFILE_NAME = "default";

// El análisis encadena varias llamadas a Gemini: puede tardar varios minutos.
const ANALYZE_TIMEOUT_MS = 10 * 60 * 1000;

const apiClient = axios.create({ baseURL: API_BASE_URL, timeout: 30_000 });

// ---------------------------------------------------------------------------
// Tipos (reflejan los esquemas Pydantic del backend)
// ---------------------------------------------------------------------------

export type EmotionLabel = "neutral" | "anxiety" | "stress" | "depression";
export type IndicatorKey = "pressure" | "rhythm" | "slant" | "spacing" | "stroke_variability";
export type NodeStatus = "ok" | "stopped" | "degraded";

export type IndicatorAssessment = { score: number; observation: string };
export type VisualIndicators = Record<IndicatorKey, IndicatorAssessment>;

export type NodeTrace = {
  node: string;
  status: NodeStatus;
  duration_ms: number;
  detail: string | null;
};

export type RagSource = {
  source: string;
  title: string;
  section: string | null;
  score: number;
  snippet: string;
  cited: boolean;
};

export type EmotionalTrend = {
  profile_name: string;
  window_size: number;
  total_analyses: number;
  distribution: Record<EmotionLabel, number>;
  percentages: Record<EmotionLabel, number>;
  dominant_emotion: EmotionLabel | null;
  stress_anxiety_count: number;
  stress_anxiety_ratio: number;
  consecutive_stress_anxiety: number;
  depression_count: number;
  recurrent_stress_anxiety: boolean;
  alert_level: "none" | "watch" | "elevated";
  summary: string;
};

export type AnalyzeResponse = {
  id: string;
  profile_name: string;
  created_at: string;
  contains_handwriting: true;
  transcribed_text: string;
  emotion: EmotionLabel;
  confidence: number;
  visual_indicators: VisualIndicators;
  visual_summary: string;
  biomarkers: Record<IndicatorKey, number>;
  feedback: string;
  rag_sources: RagSource[];
  cited_sources: string[];
  history: { previous_analyses: number; trend: EmotionalTrend | null };
  risk_flags: string[];
  trace: NodeTrace[];
  latency_ms: number;
  models: Record<string, string>;
  image_path: string;
  processed_image_path: string;
};

/** Registro tal como lo devuelve GET /api/v1/history. */
export type HistoryRecord = {
  id: string;
  profile_name: string;
  image_path: string;
  processed_image_path: string | null;
  transcribed_text: string | null;
  emotion: EmotionLabel | null;
  confidence: number | null;
  visual_indicators: Partial<VisualIndicators> | null;
  visual_summary: string | null;
  agent_feedback: string | null;
  rag_sources: RagSource[] | null;
  trace: NodeTrace[] | null;
  created_at: string;
};

export type HistoryDeleteResponse = { deleted: number; profile_name: string | null };

// ---------------------------------------------------------------------------
// Errores
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  readonly status: number | null;
  readonly title: string;
  readonly detail: string;
  readonly reason: string | null;
  readonly trace: NodeTrace[];

  constructor(args: {
    status: number | null;
    title: string;
    detail: string;
    reason?: string | null;
    trace?: NodeTrace[];
  }) {
    super(args.detail);
    this.name = "ApiError";
    this.status = args.status;
    this.title = args.title;
    this.detail = args.detail;
    this.reason = args.reason ?? null;
    this.trace = args.trace ?? [];
  }
}

const TITLES_BY_STATUS: Record<number, string> = {
  400: "Archivo no válido",
  422: "No detectamos escritura manuscrita",
  429: "Demasiadas peticiones a Gemini",
  502: "Respuesta inesperada del modelo",
  503: "Servicio de IA no disponible",
  504: "El modelo tardó demasiado en responder",
};

/** Convierte `detail` (string, o lista de errores de validación de FastAPI) en texto legible. */
function detailToText(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const field = Array.isArray(item?.loc) ? item.loc.filter((p: unknown) => p !== "body").join(".") : "";
        return field ? `${field}: ${item?.msg ?? ""}` : String(item?.msg ?? item);
      })
      .join(" · ");
  }
  return "";
}

function toApiError(error: unknown): ApiError {
  if (!axios.isAxiosError(error)) {
    return new ApiError({
      status: null,
      title: "Error inesperado",
      detail: error instanceof Error ? error.message : "No se pudo completar la operación.",
    });
  }

  const axiosError = error as AxiosError<any>;
  const response = axiosError.response;

  if (!response) {
    const timedOut = axiosError.code === "ECONNABORTED" || axiosError.code === "ETIMEDOUT";
    return new ApiError({
      status: null,
      title: timedOut ? "Tiempo de espera agotado" : "No se pudo conectar con el backend",
      detail: timedOut
        ? "El análisis no respondió a tiempo. Inténtalo de nuevo en unos minutos."
        : `Comprueba que el backend esté en marcha en ${API_BASE_URL.replace(/\/api\/v1$/, "")} y que CORS permita este origen.`,
    });
  }

  const data = response.data ?? {};
  const detail = detailToText(data.detail) || response.statusText || "Error sin detalle.";
  const isNoHandwriting = response.status === 422 && data.contains_handwriting === false;

  return new ApiError({
    status: response.status,
    title: isNoHandwriting
      ? TITLES_BY_STATUS[422]
      : response.status === 422
        ? "Datos no válidos"
        : (TITLES_BY_STATUS[response.status] ?? `Error del servidor (${response.status})`),
    detail,
    reason: typeof data.reason === "string" ? data.reason : null,
    trace: Array.isArray(data.trace) ? data.trace : [],
  });
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export async function analyzeHandwriting(args: {
  file: File;
  profileName: string;
  onUploadProgress?: (event: AxiosProgressEvent) => void;
  signal?: AbortSignal;
}): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", args.file);
  formData.append("profile_name", args.profileName);

  try {
    const { data } = await apiClient.post<AnalyzeResponse>("/analyze", formData, {
      timeout: ANALYZE_TIMEOUT_MS,
      onUploadProgress: args.onUploadProgress,
      signal: args.signal,
    });
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export async function fetchHistory(profileName?: string): Promise<HistoryRecord[]> {
  try {
    const { data } = await apiClient.get<HistoryRecord[]>("/history", {
      params: profileName ? { profile_name: profileName } : undefined,
    });
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

export async function clearHistory(profileName?: string): Promise<HistoryDeleteResponse> {
  try {
    const { data } = await apiClient.delete<HistoryDeleteResponse>("/history", {
      params: profileName ? { profile_name: profileName } : undefined,
    });
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

/** GET /health vive fuera de /api/v1. */
export async function checkHealth(): Promise<boolean> {
  try {
    const healthUrl = `${API_BASE_URL.replace(/\/api\/v1$/, "")}/health`;
    const { data } = await axios.get(healthUrl, { timeout: 5_000 });
    return data?.status === "ok";
  } catch {
    return false;
  }
}
