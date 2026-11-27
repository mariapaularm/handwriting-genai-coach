import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { type AnalysisView, emotionMeta, fromAnalyzeResponse, fromHistoryRecord } from "../../lib/presentation";
import {
  ApiError,
  DEFAULT_PROFILE_NAME,
  type HistoryRecord,
  PROFILE_NAME_PATTERN,
  analyzeHandwriting,
  clearHistory,
  fetchHistory,
} from "../../services/api";
import ResultsView from "../analysis/ResultsView";
import HistoryPanel, { type HistoryScope } from "../history/HistoryPanel";
import Uploader from "../upload/Uploader";

const PROFILE_STORAGE_KEY = "handwriting-profile-name";

function asApiError(error: unknown): ApiError {
  return error instanceof ApiError
    ? error
    : new ApiError({ status: null, title: "Error inesperado", detail: String(error) });
}

function scrollToSection(id: "lectura" | "muestra") {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.getElementById(id)?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
}

export default function Dashboard({ onHistoryChanged }: { onHistoryChanged?: () => void }) {
  const [profileName, setProfileName] = useState(
    () => window.localStorage.getItem(PROFILE_STORAGE_KEY) || DEFAULT_PROFILE_NAME,
  );
  const isProfileValid = PROFILE_NAME_PATTERN.test(profileName);

  // Análisis en curso
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [analysisError, setAnalysisError] = useState<ApiError | null>(null);
  const [lastSuccessAt, setLastSuccessAt] = useState<string | null>(null);
  const [wasCancelled, setWasCancelled] = useState(false);
  const [activeView, setActiveView] = useState<AnalysisView | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Historial
  const [scope, setScope] = useState<HistoryScope>("profile");
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<ApiError | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (isProfileValid) window.localStorage.setItem(PROFILE_STORAGE_KEY, profileName);
  }, [isProfileValid, profileName]);

  // Cancela una petición pendiente si el componente se desmonta.
  useEffect(() => () => abortRef.current?.abort(), []);

  const historyFilter = scope === "profile" ? profileName : undefined;

  const loadHistory = useCallback(async () => {
    if (scope === "profile" && !isProfileValid) {
      setRecords([]);
      return;
    }
    const requestId = ++requestIdRef.current;
    setHistoryLoading(true);
    try {
      const data = await fetchHistory(historyFilter);
      if (requestId === requestIdRef.current) {
        setRecords(data);
        setHistoryError(null);
      }
    } catch (error) {
      if (requestId === requestIdRef.current) setHistoryError(asApiError(error));
    } finally {
      if (requestId === requestIdRef.current) setHistoryLoading(false);
    }
  }, [historyFilter, isProfileValid, scope]);

  // Recarga con un pequeño debounce mientras se escribe el perfil.
  useEffect(() => {
    const timer = window.setTimeout(() => void loadHistory(), 300);
    return () => window.clearTimeout(timer);
  }, [loadHistory]);

  async function handleSubmit(file: File) {
    const controller = new AbortController();
    abortRef.current = controller;
    setIsAnalyzing(true);
    setStartedAt(Date.now());
    setUploadPercent(0);
    setAnalysisError(null);
    setLastSuccessAt(null);
    setWasCancelled(false);
    scrollToSection("lectura");

    try {
      const response = await analyzeHandwriting({
        file,
        profileName,
        signal: controller.signal,
        onUploadProgress: (event) => {
          setUploadPercent(event.total ? Math.round((event.loaded / event.total) * 100) : 100);
        },
      });
      setActiveView(fromAnalyzeResponse(response));
      setLastSuccessAt(response.created_at);
      await loadHistory();
      onHistoryChanged?.();
    } catch (error) {
      if (controller.signal.aborted) {
        setWasCancelled(true);
      } else {
        setAnalysisError(asApiError(error));
        // El aviso de error vive junto a la muestra: se lleva la vista hasta él.
        window.requestAnimationFrame(() => scrollToSection("muestra"));
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setIsAnalyzing(false);
    }
  }

  function handleCancel() {
    abortRef.current?.abort();
  }

  async function handleClear() {
    setIsDeleting(true);
    try {
      await clearHistory(historyFilter);
      const removed = new Set(records.map((r) => r.id));
      setActiveView((view) => (view && removed.has(view.id) ? null : view));
      setHistoryError(null);
      await loadHistory();
      onHistoryChanged?.();
    } catch (error) {
      setHistoryError(asApiError(error));
    } finally {
      setIsDeleting(false);
    }
  }

  function handleSelect(record: HistoryRecord) {
    setActiveView(fromHistoryRecord(record));
    setAnalysisError(null);
    scrollToSection("lectura");
  }

  const dominant = useMemo(() => {
    const counts = new Map<string, number>();
    for (const record of records) {
      if (record.emotion) counts.set(record.emotion, (counts.get(record.emotion) ?? 0) + 1);
    }
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
    return top ? { label: emotionMeta(top[0]).label, count: top[1] } : null;
  }, [records]);

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-8">
      <div className="grid items-start gap-6 xl:grid-cols-[minmax(340px,0.78fr)_minmax(0,1.5fr)]">
        <div className="xl:sticky xl:top-6">
          <Uploader
            error={analysisError}
            isLoading={isAnalyzing}
            lastSuccessAt={lastSuccessAt}
            onCancel={handleCancel}
            onDismissError={() => {
              setAnalysisError(null);
              setWasCancelled(false);
            }}
            onProfileNameChange={setProfileName}
            onSubmit={handleSubmit}
            profileName={profileName}
            wasCancelled={wasCancelled}
          />
        </div>
        <section aria-label="Lectura del análisis" className="min-w-0 scroll-mt-6" id="lectura">
          <ResultsView
            isLoading={isAnalyzing}
            profileName={profileName}
            startedAt={startedAt}
            uploadPercent={uploadPercent}
            view={activeView}
          />
        </section>
      </div>

      <HistoryPanel
        dominant={dominant}
        error={historyError}
        isDeleting={isDeleting}
        isLoading={historyLoading}
        onClear={handleClear}
        onRefresh={() => void loadHistory()}
        onScopeChange={setScope}
        onSelect={handleSelect}
        profileName={profileName}
        records={records}
        scope={scope}
        selectedId={activeView?.id ?? null}
      />
    </div>
  );
}
