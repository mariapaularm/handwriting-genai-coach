import { Alert, Card, Chip } from "@heroui/react";
import { Activity, BookOpenText, History, MessageSquareHeart, Paperclip, PenLine, Timer } from "lucide-react";
import { type CSSProperties, type KeyboardEvent, type ReactNode, useId, useRef, useState } from "react";

import {
  type AnalysisView,
  EMOTION_META,
  INDICATOR_META,
  INDICATOR_ORDER,
  NODE_META,
  NODE_STATUS_META,
  TREND_ALERT_META,
  emotionMeta,
  formatDate,
  formatMs,
  formatPercent,
  scoreBand,
} from "../../lib/presentation";
import type { EmotionLabel, IndicatorAssessment, IndicatorKey } from "../../services/api";
import AnalysisProgress from "./AnalysisProgress";
import FeedbackMarkdown from "./FeedbackMarkdown";

type ResultsViewProps = {
  view: AnalysisView | null;
  isLoading: boolean;
  startedAt: number | null;
  uploadPercent: number;
  profileName: string;
};

export default function ResultsView({ view, isLoading, startedAt, uploadPercent, profileName }: ResultsViewProps) {
  if (isLoading && startedAt) {
    return <AnalysisProgress profileName={profileName} startedAt={startedAt} uploadPercent={uploadPercent} />;
  }

  if (!view) return <EmptyReading />;

  return (
    <div className="@container">
      {/* `key` reinicia la entrada en cascada cada vez que se abre otra lectura. */}
      <div className="reading-in grid grid-cols-1 gap-5 @3xl:grid-cols-2 [&>*]:min-w-0" key={view.id}>
        <EmotionCard view={view} />

        {view.riskFlags.length ? (
          <div className="@3xl:col-span-2">
            <Alert status="danger">
              <Alert.Indicator />
              <Alert.Content>
                <Alert.Title>Lo primero es tu seguridad</Alert.Title>
                <Alert.Description>
                  Lo que escribiste contiene expresiones de riesgo. Si estás pasando por un momento difícil, llama a
                  emergencias (112 / 911) o a una línea de ayuda en crisis, como el 024 en España o el 988 en EE.&nbsp;UU.
                </Alert.Description>
              </Alert.Content>
            </Alert>
          </div>
        ) : null}

        <TranscriptCard text={view.transcribedText} />

        <BentoCard icon={Activity} subtitle="Cinco rasgos puntuados de 0 a 100: bajo, medio o alto." title="Cómo es tu trazo">
          {view.visualSummary ? <p className="prose-serif mb-6 text-pretty">{view.visualSummary}</p> : null}
          <IndicatorList indicators={view.indicators} />
        </BentoCard>

        <BentoCard
          className="@3xl:col-span-2"
          icon={MessageSquareHeart}
          subtitle="Redactada por el coach a partir de tu trazo, tu historial y la base de conocimiento."
          title="Retroalimentación del coach"
        >
          {view.feedback ? (
            <FeedbackMarkdown text={view.feedback} />
          ) : (
            <p className="text-caption text-ink-soft">Este análisis no tiene retroalimentación guardada.</p>
          )}
        </BentoCard>

        <DetailsCard view={view} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tarjetas
// ---------------------------------------------------------------------------

/** Tarjeta protagonista: emoción con pulso, anillo de confianza y rasgos clave. */
function EmotionCard({ view }: { view: AnalysisView }) {
  const meta = emotionMeta(view.emotion);
  const confidence = Math.round(view.confidence * 100);
  const traits = keyTraits(view.indicators);

  return (
    <section aria-labelledby="emotion-heading" className="feature-card mt-3 p-6 sm:p-8 @3xl:col-span-2">
      <Paperclip aria-hidden="true" className="feature-clip" size={30} strokeWidth={1.75} />

      <p className="flex flex-wrap items-center gap-x-3 gap-y-1 pr-10 text-caption text-ink-soft">
        <span>
          Perfil «<span translate="no">{view.profileName}</span>», {formatDate(view.createdAt)}
        </span>
        {view.model ? (
          <span className="font-mono text-meta" translate="no">
            {view.model}
          </span>
        ) : null}
        {view.source === "history" ? (
          <Chip size="sm" variant="secondary">
            Del historial
          </Chip>
        ) : null}
      </p>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-x-8 gap-y-6">
        <div className="min-w-0 flex-1 basis-72">
          <h2
            className={`flex items-center gap-3.5 font-display text-display font-medium ${meta.textClass}`}
            id="emotion-heading"
          >
            <span aria-hidden="true" className="emo-dot emo-dot-pulse size-3.5" />
            <span>
              <span className="sr-only">Lectura: </span>
              {meta.label}
            </span>
          </h2>
          <p className="mt-2 max-w-[52ch] text-lead text-pretty text-ink">{meta.reading}</p>

          {traits.length ? (
            <div className="mt-5">
              <p className="mb-2 text-caption font-medium text-ink-soft" id="traits-label">
                Rasgos que más destacan en tu trazo
              </p>
              <ul aria-labelledby="traits-label" className="flex flex-wrap gap-2">
                {traits.map(({ key, score }) => {
                  const info = INDICATOR_META[key];
                  const band = scoreBand(score);
                  return (
                    <li key={key}>
                      <button
                        aria-label={`${info.label}: ${score} de 100, banda ${band.label}. Ver en «Cómo es tu trazo».`}
                        className="trait-pill"
                        onClick={() => focusTrait(key)}
                        title={`${info.label}: ${band.label}`}
                        type="button"
                      >
                        {info.label}
                        <span className="trait-pill-value">{score}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ) : null}
        </div>

        <ConfidenceRing value={confidence} />
      </div>

      <p className="mt-6 text-caption text-ink-soft">Lectura orientativa, no es un diagnóstico.</p>
    </section>
  );
}

/** Los tres rasgos más alejados del punto medio (50) son los más característicos. */
function keyTraits(indicators: AnalysisView["indicators"]) {
  return INDICATOR_ORDER.filter((key) => indicators[key])
    .map((key) => ({ key, score: (indicators[key] as IndicatorAssessment).score }))
    .sort((a, b) => Math.abs(b.score - 50) - Math.abs(a.score - 50))
    .slice(0, 3);
}

/** Lleva al indicador dentro de «Cómo es tu trazo» y lo resalta brevemente. */
function focusTrait(key: IndicatorKey) {
  const element = document.getElementById(`trait-${key}`);
  if (!element) return;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  element.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" });
  element.focus({ preventScroll: true });
  element.classList.remove("indicator-flash");
  void element.offsetWidth; // reinicia la animación si se pulsa dos veces
  element.classList.add("indicator-flash");
}

/** Hoja rayada en su propia tarjeta, con encabezado diferenciado. */
function TranscriptCard({ text }: { text: string }) {
  return (
    <section aria-labelledby="transcript-heading" className="surface-card flex flex-col overflow-hidden">
      <header className="card-band flex items-center gap-3 px-5 py-3.5 sm:px-6">
        <span aria-hidden="true" className="grid size-8 shrink-0 place-items-center rounded-lg bg-paper text-salvia">
          <PenLine size={16} />
        </span>
        <div className="min-w-0">
          <h3 className="font-display text-h3 font-medium" id="transcript-heading">
            Lo que escribiste
          </h3>
          <p className="text-meta text-ink-soft">Transcripción literal de tu muestra.</p>
        </div>
      </header>
      <div className="flex flex-1 p-5 sm:p-6">
        <div className="notebook flex-1">
          <p className="notebook-text">{text || "No se pudo transcribir el texto."}</p>
        </div>
      </div>
    </section>
  );
}

function BentoCard({
  title,
  subtitle,
  icon: Icon,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  icon: typeof Activity;
  children: ReactNode;
  className?: string;
}) {
  const headingId = useId();
  return (
    <section aria-labelledby={headingId} className={`surface-card p-5 sm:p-6 ${className}`}>
      <header className="mb-5 flex items-start gap-3">
        <span aria-hidden="true" className="grid size-8 shrink-0 place-items-center rounded-lg bg-olive-soft text-olive">
          <Icon size={16} />
        </span>
        <div className="min-w-0">
          <h3 className="font-display text-h3 font-medium text-balance" id={headingId}>
            {title}
          </h3>
          {subtitle ? <p className="text-meta text-ink-soft">{subtitle}</p> : null}
        </div>
      </header>
      {children}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Indicadores del trazo: barras oliva con valor en pizarra.
// ---------------------------------------------------------------------------

function IndicatorList({ indicators }: { indicators: AnalysisView["indicators"] }) {
  const available = INDICATOR_ORDER.filter((key) => indicators[key]);
  if (!available.length) return <p className="text-caption text-ink-soft">No hay indicadores para este análisis.</p>;

  return (
    <ul className="space-y-5">
      {available.map((key) => (
        <Indicator assessment={indicators[key] as IndicatorAssessment} indicatorKey={key} key={key} />
      ))}
    </ul>
  );
}

function Indicator({ indicatorKey, assessment }: { indicatorKey: IndicatorKey; assessment: IndicatorAssessment }) {
  const info = INDICATOR_META[indicatorKey];
  const band = scoreBand(assessment.score);
  const value = Math.min(100, Math.max(0, assessment.score));

  return (
    <li className="-mx-2 min-w-0 scroll-mt-24 rounded-lg px-2 py-1.5" id={`trait-${indicatorKey}`} tabIndex={-1}>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-body font-semibold">{info.label}</span>
        <span className="tabular flex items-baseline gap-2 text-caption">
          <span className="rounded-md bg-lino px-1.5 py-0.5 text-meta font-semibold text-salvia">{band.label}</span>
          <span className="w-7 text-right font-semibold text-salvia">{assessment.score}</span>
        </span>
      </div>
      <div className="relative mt-2.5">
        <div
          aria-label={info.label}
          aria-valuemax={100}
          aria-valuemin={0}
          aria-valuenow={value}
          aria-valuetext={`${value} de 100, banda ${band.label}. Escala de ${info.low} a ${info.high}.`}
          className="meter-track"
          role="meter"
        >
          <div className="relative z-[1] h-full rounded-full bg-salvia-graph" style={{ width: `${value}%` }} />
        </div>
        <span aria-hidden="true" className="meter-marker" style={{ left: `${value}%` }} />
      </div>
      <div aria-hidden="true" className="mt-2 flex justify-between text-meta text-ink-soft">
        <span>{info.low}</span>
        <span>{info.high}</span>
      </div>
      {assessment.observation ? <p className="mt-1.5 text-caption text-pretty">{assessment.observation}</p> : null}
    </li>
  );
}

// ---------------------------------------------------------------------------
// Detalles técnicos en pestañas: Fuentes RAG · Historial MCP · Traza LangGraph
// ---------------------------------------------------------------------------

type TabKey = "rag" | "mcp" | "trace";

const TABS: { key: TabKey; label: string; icon: typeof Activity }[] = [
  { key: "rag", label: "Fuentes RAG", icon: BookOpenText },
  { key: "mcp", label: "Historial MCP", icon: History },
  { key: "trace", label: "Traza LangGraph", icon: Timer },
];

function DetailsCard({ view }: { view: AnalysisView }) {
  const [active, setActive] = useState<TabKey>("rag");
  const tabRefs = useRef<Record<TabKey, HTMLButtonElement | null>>({ rag: null, mcp: null, trace: null });
  const baseId = useId();

  // Patrón WAI-ARIA de pestañas: flechas, Inicio y Fin mueven el foco y activan.
  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const index = TABS.findIndex((tab) => tab.key === active);
    const next =
      event.key === "ArrowRight"
        ? (index + 1) % TABS.length
        : event.key === "ArrowLeft"
          ? (index - 1 + TABS.length) % TABS.length
          : event.key === "Home"
            ? 0
            : event.key === "End"
              ? TABS.length - 1
              : null;
    if (next === null) return;
    event.preventDefault();
    const key = TABS[next].key;
    setActive(key);
    tabRefs.current[key]?.focus();
  }

  const subtitle =
    active === "rag"
      ? "Fragmentos de la base de conocimiento que el coach tuvo a mano."
      : active === "mcp"
        ? "Cómo ha evolucionado tu perfil en los últimos análisis."
        : view.latencyMs
          ? `Tiempo total: ${formatMs(view.latencyMs)}. Duración de cada etapa del grafo.`
          : "Duración de cada etapa del grafo.";

  return (
    <section aria-labelledby={`${baseId}-heading`} className="surface-card p-5 sm:p-6 @3xl:col-span-2">
      <h3 className="font-display text-h3 font-medium" id={`${baseId}-heading`}>
        Cómo se hizo esta lectura
      </h3>
      <p className="mb-4 text-meta text-ink-soft">{subtitle}</p>

      <div aria-label="Detalles del análisis" className="tablist scrollbar-soft overflow-x-auto" onKeyDown={onKeyDown} role="tablist">
        {TABS.map(({ key, label, icon: Icon }) => (
          <button
            aria-controls={`${baseId}-panel-${key}`}
            aria-selected={active === key}
            className="tab"
            id={`${baseId}-tab-${key}`}
            key={key}
            onClick={() => setActive(key)}
            ref={(node) => {
              tabRefs.current[key] = node;
            }}
            role="tab"
            tabIndex={active === key ? 0 : -1}
            type="button"
          >
            <Icon aria-hidden="true" className="hidden shrink-0 sm:block" size={15} />
            {label}
          </button>
        ))}
      </div>

      {TABS.map(({ key }) => (
        <div
          aria-labelledby={`${baseId}-tab-${key}`}
          className="pt-5"
          hidden={active !== key}
          id={`${baseId}-panel-${key}`}
          key={key}
          role="tabpanel"
          tabIndex={0}
        >
          {key === "rag" ? <RagSection view={view} /> : key === "mcp" ? <TrendSection view={view} /> : <TraceSection view={view} />}
        </div>
      ))}
    </section>
  );
}

function TrendSection({ view }: { view: AnalysisView }) {
  if (!view.trend) {
    return (
      <p className="text-caption text-ink-soft">
        {view.source === "history"
          ? "La tendencia se calcula al hacer un análisis nuevo. Analiza otra muestra con este perfil para verla."
          : "No se pudo consultar tu historial en este análisis."}
      </p>
    );
  }

  const trend = view.trend;
  const alert = TREND_ALERT_META[trend.alert_level];
  const emotions = Object.keys(EMOTION_META) as EmotionLabel[];
  const previous = view.previousAnalyses ?? trend.total_analyses;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Chip color={alert.color} size="sm" variant="soft">
          {alert.label}
        </Chip>
        <span className="tabular text-caption text-ink-soft">
          {previous === 1 ? "1 análisis previo" : `${previous} análisis previos`}
        </span>
      </div>
      {trend.recurrent_stress_anxiety ? (
        <p className="text-caption font-semibold text-alert">El estrés o la ansiedad se repiten en tus últimos análisis.</p>
      ) : null}
      <p className="text-body text-pretty">{trend.summary}</p>
      {trend.total_analyses > 0 ? (
        <dl className="max-w-xl space-y-2.5">
          {emotions.map((emotion) => {
            const percentage = trend.percentages[emotion] ?? 0;
            const emotionInfo = EMOTION_META[emotion];
            return (
              <div className="grid grid-cols-[6.5rem_1fr_2.5rem] items-center gap-3 text-caption" key={emotion}>
                <dt className="font-medium">{emotionInfo.label}</dt>
                <dd aria-hidden="true" className="h-2 overflow-hidden rounded-full bg-lino">
                  <div className={`h-full rounded-full ${emotionInfo.barClass}`} style={{ width: `${percentage}%` }} />
                </dd>
                <dd className="tabular text-right text-ink-soft">
                  {trend.distribution[emotion] ?? 0}
                  <span className="sr-only"> análisis, {formatPercent(percentage)}</span>
                </dd>
              </div>
            );
          })}
        </dl>
      ) : null}
    </div>
  );
}

/** Oliva para el nodo principal, salvia para los paralelos y terracota para la síntesis. */
const NODE_BAR: Record<string, string> = {
  visual_analysis: "bg-olive-deep",
  mcp_history: "bg-salvia-node",
  rag_retrieval: "bg-salvia-node",
  synthesis_guardrail: "bg-terracota",
};

function TraceSection({ view }: { view: AnalysisView }) {
  if (!view.trace.length) return <p className="text-caption text-ink-soft">No hay traza disponible para este análisis.</p>;
  const max = Math.max(...view.trace.map((t) => t.duration_ms), 1);

  return (
    <ol className="max-w-2xl space-y-4">
      {view.trace.map((entry) => {
        const info = NODE_META[entry.node] ?? { label: entry.node, description: "" };
        const status = NODE_STATUS_META[entry.status] ?? NODE_STATUS_META.ok;
        const parallel = entry.node === "mcp_history" || entry.node === "rag_retrieval";
        return (
          <li key={entry.node}>
            <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2 text-caption">
              <span className="flex min-w-0 items-center gap-2 font-semibold">
                {info.label}
                {parallel ? <span className="font-normal text-ink-soft">en paralelo</span> : null}
                {entry.status !== "ok" ? (
                  <Chip color={status.color} size="sm" variant="soft">
                    {status.label}
                  </Chip>
                ) : null}
              </span>
              <span className="tabular font-mono text-meta">{formatMs(entry.duration_ms)}</span>
            </div>
            <div aria-hidden="true" className="h-2 rounded-full bg-lino">
              <div
                className={`h-full rounded-full ${NODE_BAR[entry.node] ?? "bg-salvia-node"}`}
                style={{ width: `${Math.max(2, (entry.duration_ms / max) * 100)}%` }}
              />
            </div>
            {entry.detail ? <p className="mt-1.5 text-meta break-words text-ink-soft">{entry.detail}</p> : null}
          </li>
        );
      })}
    </ol>
  );
}

/** Los fragmentos llegan en Markdown crudo: se quitan marcas de título, negrita y viñetas. */
function plainSnippet(snippet: string): string {
  return snippet
    .replace(/(^|\s)#{1,6}\s+/g, "$1")
    .replace(/\*\*|__/g, "")
    .replace(/(^|\s)[-*]\s+/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

function RagSection({ view }: { view: AnalysisView }) {
  if (!view.ragSources.length) {
    return (
      <p className="text-caption text-ink-soft">
        No se recuperaron fuentes en este análisis. Revisa en la pestaña de traza si la búsqueda quedó degradada.
      </p>
    );
  }

  const sources = [...view.ragSources].sort((a, b) => Number(b.cited) - Number(a.cited) || b.score - a.score);

  return (
    <ul className="divide-y divide-line">
      {sources.map((source, index) => (
        <li
          className="grid gap-1.5 py-4 first:pt-0 last:pb-0 md:grid-cols-[minmax(0,14rem)_1fr] md:gap-6"
          key={`${source.source}-${index}`}
        >
          <div className="min-w-0">
            <p className="font-mono text-meta font-medium break-all text-salvia" translate="no">
              {source.source}
            </p>
            <p className="tabular mt-0.5 text-meta text-ink-soft">
              {source.cited ? "Citada en tu lectura. " : ""}Similitud {formatPercent(Math.round(source.score * 100))}
            </p>
          </div>
          <div className="min-w-0">
            <p className="text-caption font-semibold">
              {source.title}
              {source.section ? <span className="font-normal text-ink-soft"> › {source.section}</span> : null}
            </p>
            <p className="mt-1 text-caption text-pretty text-ink-soft">{plainSnippet(source.snippet)}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// Estado vacío y piezas comunes
// ---------------------------------------------------------------------------

function EmptyReading() {
  return (
    <Card className="surface-card" variant="default">
      <Card.Content className="gap-6 py-4 sm:px-4">
        <div>
          <h2 className="font-display text-h2 font-medium">Tu lectura aparecerá aquí</h2>
          <p className="mt-1 max-w-[60ch] text-caption text-ink-soft">
            Verás la emoción orientativa, lo que escribiste, cómo es tu trazo y consejos para hoy.
          </p>
        </div>
        <div aria-hidden="true" className="notebook min-h-64">
          <p className="notebook-text text-ink-faint italic">
            Sube una foto de tu escritura y pulsa «Analizar muestra».{"\n"}También puedes abrir un análisis de tu
            historial.
          </p>
        </div>
        <p className="sr-only">
          Sube una foto de tu escritura y pulsa «Analizar muestra», o abre un análisis de tu historial.
        </p>
      </Card.Content>
    </Card>
  );
}

/** Anillo SVG que dibuja el porcentaje de confianza al cargar la lectura. */
function ConfidenceRing({ value }: { value: number }) {
  const labelId = useId();
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - value / 100);
  const ringStyle = {
    strokeDasharray: `${circumference}px`,
    strokeDashoffset: `${offset}px`,
    "--ring-circumference": `${circumference}px`,
  } as CSSProperties;

  return (
    <div className="flex shrink-0 flex-col items-center gap-2">
      <div
        aria-labelledby={labelId}
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={value}
        aria-valuetext={formatPercent(value)}
        className="relative size-36"
        role="meter"
      >
        <svg aria-hidden="true" className="size-full -rotate-90" viewBox="0 0 120 120">
          <circle cx="60" cy="60" fill="none" r={radius} stroke="var(--hw-lino)" strokeWidth="10" />
          <circle
            className="ring-progress"
            cx="60"
            cy="60"
            fill="none"
            r={radius}
            stroke="var(--hw-olive-deep)"
            strokeLinecap="round"
            strokeWidth="10"
            style={ringStyle}
          />
        </svg>
        <span className="tabular absolute inset-0 grid place-items-center font-display text-h2 font-semibold">
          {formatPercent(value)}
        </span>
      </div>
      <p className="text-caption font-medium text-ink-soft" id={labelId}>
        Confianza del modelo
      </p>
    </div>
  );
}
