import type {
  AnalyzeResponse,
  EmotionalTrend,
  EmotionLabel,
  HistoryRecord,
  IndicatorKey,
  NodeTrace,
  RagSource,
  VisualIndicators,
} from "../services/api";

type ChipColor = "accent" | "danger" | "default" | "success" | "warning";

type EmotionMeta = {
  label: string;
  /** Frase corta, en segunda persona, que acompaña a la etiqueta en la lectura. */
  reading: string;
  /** Etiqueta con fondo (`.emo-tag .emo-*`). */
  tagClass: string;
  /** Solo color de texto y punto (`.emo-text-*`). */
  textClass: string;
  /** Relleno de barra (`.emo-bar-*`). */
  barClass: string;
};

export const EMOTION_META: Record<EmotionLabel, EmotionMeta> = {
  neutral: {
    label: "Neutral",
    reading: "Tu trazo se ve estable y sin señales marcadas de tensión.",
    tagClass: "emo-tag emo-neutral",
    textClass: "emo-text-neutral",
    barClass: "emo-bar-neutral",
  },
  anxiety: {
    label: "Ansiedad",
    reading: "Tu trazo muestra rasgos que suelen acompañar a la inquietud.",
    tagClass: "emo-tag emo-anxiety",
    textClass: "emo-text-anxiety",
    barClass: "emo-bar-anxiety",
  },
  stress: {
    label: "Estrés",
    reading: "Tu trazo muestra rasgos que suelen acompañar a la tensión sostenida.",
    tagClass: "emo-tag emo-stress",
    textClass: "emo-text-stress",
    barClass: "emo-bar-stress",
  },
  depression: {
    label: "Ánimo bajo",
    reading: "Tu trazo muestra rasgos que suelen acompañar al cansancio o la falta de energía.",
    tagClass: "emo-tag emo-depression",
    textClass: "emo-text-depression",
    barClass: "emo-bar-depression",
  },
};

export function emotionMeta(emotion: string | null | undefined) {
  return EMOTION_META[(emotion as EmotionLabel) ?? "neutral"] ?? EMOTION_META.neutral;
}

export const INDICATOR_ORDER: IndicatorKey[] = ["pressure", "rhythm", "slant", "spacing", "stroke_variability"];

export const INDICATOR_META: Record<IndicatorKey, { label: string; low: string; high: string }> = {
  pressure: { label: "Presión", low: "tenue", high: "marcada" },
  rhythm: { label: "Ritmo", low: "entrecortado", high: "fluido" },
  slant: { label: "Inclinación", low: "izquierda", high: "derecha" },
  spacing: { label: "Espaciado", low: "apretado", high: "amplio" },
  stroke_variability: { label: "Variabilidad del trazo", low: "uniforme", high: "variable" },
};

/** Bandas de la escala estándar del backend (0-33 bajo, 34-66 medio, 67-100 alto). */
export function scoreBand(score: number): { label: string; color: ChipColor } {
  if (score <= 33) return { label: "bajo", color: "default" };
  if (score <= 66) return { label: "medio", color: "accent" };
  return { label: "alto", color: "warning" };
}

export const NODE_META: Record<string, { label: string; description: string }> = {
  visual_analysis: {
    label: "Análisis visual",
    description: "Gemini lee la imagen original y la preprocesada",
  },
  mcp_history: {
    label: "Historial MCP",
    description: "Herramientas FastMCP sobre el historial del perfil",
  },
  rag_retrieval: {
    label: "Búsqueda RAG",
    description: "ChromaDB recupera fragmentos de la base de conocimiento",
  },
  synthesis_guardrail: {
    label: "Síntesis y guardrails",
    description: "Retroalimentación personalizada con aviso obligatorio",
  },
};

export const NODE_STATUS_META: Record<string, { label: string; color: ChipColor }> = {
  ok: { label: "ok", color: "success" },
  degraded: { label: "degradado", color: "warning" },
  stopped: { label: "detenido", color: "danger" },
};

export const TREND_ALERT_META: Record<EmotionalTrend["alert_level"], { label: string; color: ChipColor }> = {
  none: { label: "Sin alerta", color: "success" },
  watch: { label: "Vigilar", color: "warning" },
  elevated: { label: "Alerta elevada", color: "danger" },
};

export function formatDate(value: string): string {
  // SQLite devuelve fechas sin zona horaria: se interpretan como UTC.
  const iso = /[zZ]|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`;
  return new Intl.DateTimeFormat("es", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(iso));
}

const secondsFormat = new Intl.NumberFormat("es", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const integerFormat = new Intl.NumberFormat("es", { maximumFractionDigits: 0 });

export function formatMs(ms: number): string {
  return ms >= 1000 ? `${secondsFormat.format(ms / 1000)} s` : `${integerFormat.format(ms)} ms`;
}

/** Porcentaje 0-100 con espacio fino no separable, según la norma tipográfica del español. */
export function formatPercent(value: number): string {
  return `${integerFormat.format(value)} %`;
}

// ---------------------------------------------------------------------------
// Modelo de vista común (resultado en vivo o registro del historial)
// ---------------------------------------------------------------------------

export type AnalysisView = {
  id: string;
  source: "live" | "history";
  profileName: string;
  createdAt: string;
  transcribedText: string;
  emotion: EmotionLabel;
  confidence: number;
  indicators: Partial<VisualIndicators>;
  visualSummary: string;
  feedback: string;
  ragSources: RagSource[];
  trace: NodeTrace[];
  trend: EmotionalTrend | null;
  previousAnalyses: number | null;
  riskFlags: string[];
  latencyMs: number | null;
  model: string | null;
};

export function fromAnalyzeResponse(response: AnalyzeResponse): AnalysisView {
  return {
    id: response.id,
    source: "live",
    profileName: response.profile_name,
    createdAt: response.created_at,
    transcribedText: response.transcribed_text,
    emotion: response.emotion,
    confidence: response.confidence,
    indicators: response.visual_indicators,
    visualSummary: response.visual_summary,
    feedback: response.feedback,
    ragSources: response.rag_sources,
    trace: response.trace,
    trend: response.history.trend,
    previousAnalyses: response.history.previous_analyses,
    riskFlags: response.risk_flags,
    latencyMs: response.latency_ms,
    model: response.models?.vision ?? null,
  };
}

export function fromHistoryRecord(record: HistoryRecord): AnalysisView {
  return {
    id: record.id,
    source: "history",
    profileName: record.profile_name,
    createdAt: record.created_at,
    transcribedText: record.transcribed_text ?? "",
    emotion: record.emotion ?? "neutral",
    confidence: record.confidence ?? 0,
    indicators: record.visual_indicators ?? {},
    visualSummary: record.visual_summary ?? "",
    feedback: record.agent_feedback ?? "",
    ragSources: record.rag_sources ?? [],
    trace: record.trace ?? [],
    trend: null,
    previousAnalyses: null,
    riskFlags: [],
    latencyMs: null,
    model: null,
  };
}
