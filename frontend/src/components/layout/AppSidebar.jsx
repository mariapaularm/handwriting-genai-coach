import { PenLine } from "lucide-react";

import { API_BASE_URL } from "../../services/api";

export const WELLBEING_NOTICE =
  "Herramienta de bienestar con fines orientativos y preventivos. No constituye un diagnóstico clínico ni sustituye la evaluación de un profesional de la salud mental.";

/** Pasos reales del grafo LangGraph. Historial y búsqueda se ejecutan a la vez. */
const STEPS = [
  { title: "Mirar el trazo", text: "Gemini lee la imagen, transcribe el texto y puntúa cinco rasgos del trazo." },
  {
    title: "Recordar y consultar",
    text: "A la vez: revisa tus análisis anteriores (MCP) y busca en la base de conocimiento (RAG).",
  },
  { title: "Redactar la lectura", text: "Escribe tu retroalimentación, comprueba las citas y añade el aviso de bienestar." },
];

const STATUS_META = {
  checking: { label: "Comprobando conexión…", help: null },
  online: { label: "Conectado", help: null },
  offline: {
    label: "Sin conexión",
    help: "Arranca el backend con uvicorn en el puerto 8000 y la página se reconectará sola.",
  },
};

/** Variante sobre fondo claro (cabecera móvil): solo aparece si hay algo que avisar. */
export function ConnectionStatus({ status }) {
  const meta = STATUS_META[status] ?? STATUS_META.checking;
  if (status === "online") return null;

  return (
    <span
      aria-live="polite"
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-meta font-semibold ${
        status === "offline" ? "bg-alert-soft text-alert" : "bg-paper text-ink-soft"
      }`}
      role="status"
    >
      <span
        aria-hidden="true"
        className={`size-2 rounded-full ${status === "offline" ? "bg-alert" : "bg-ink-faint"}`}
      />
      {meta.label}
    </span>
  );
}

/** Variante de la barra lateral: insignia salvia (conectado / comprobando) o coral (sin conexión). */
function SidebarConnection({ status }) {
  const meta = STATUS_META[status] ?? STATUS_META.checking;
  const isOffline = status === "offline";
  const badge = isOffline ? "bg-sidebar-alert-bg text-sidebar-alert" : "bg-sidebar-badge text-sidebar-badge-fg";
  const dot = isOffline ? "bg-sidebar-alert" : status === "online" ? "bg-sidebar-dot" : "bg-sidebar-badge-fg/45";

  return (
    <div aria-live="polite" role="status">
      <span className={`inline-flex items-center gap-2 rounded-full px-3 py-0.5 text-caption font-semibold ${badge}`}>
        <span aria-hidden="true" className={`size-2 rounded-full ${dot}`} />
        {meta.label}
      </span>
      <p className="mt-1.5 break-all font-mono text-meta text-sidebar-text" translate="no">
        {API_BASE_URL}
      </p>
      {meta.help ? <p className="mt-1.5 text-meta font-medium text-sidebar-alert-text">{meta.help}</p> : null}
    </div>
  );
}

export default function AppSidebar({ backendStatus }) {
  return (
    <aside className="hidden w-76 shrink-0 border-r border-sidebar-edge bg-sidebar text-sidebar-text lg:block">
      <div className="scrollbar-sidebar sticky top-0 flex h-screen flex-col gap-5 overflow-y-auto px-6 py-5">
        <div className="flex items-center gap-3">
          <span
            aria-hidden="true"
            className="grid size-9 shrink-0 place-items-center rounded-lg bg-sidebar-brand text-sidebar-brand-fg"
          >
            <PenLine size={18} />
          </span>
          <div className="min-w-0">
            <p className="font-display text-lead font-semibold leading-tight text-sidebar-title" translate="no">
              Handwriting Coach
            </p>
            <p className="text-meta text-sidebar-text">Bienestar a partir de tu escritura</p>
          </div>
        </div>

        <section aria-labelledby="sidebar-connection">
          <h2 className="mb-1.5 text-meta font-semibold text-sidebar-text" id="sidebar-connection">
            Servidor de análisis
          </h2>
          <SidebarConnection status={backendStatus} />
        </section>

        <section aria-labelledby="sidebar-steps">
          <h2 className="mb-1 font-display text-h3 font-medium text-sidebar-title" id="sidebar-steps">
            Cómo se lee tu muestra
          </h2>
          <p className="mb-3.5 text-meta text-sidebar-text">
            Tras subirla y limpiarla con OpenCV, el agente sigue tres pasos.
          </p>
          <ol className="relative space-y-3.5 border-l border-sidebar-rule pl-5">
            {STEPS.map((step, index) => (
              <li className="relative" key={step.title}>
                <span
                  aria-hidden="true"
                  className="tabular absolute top-0 -left-[2rem] grid size-6 place-items-center rounded-full bg-sidebar-step text-meta font-bold text-sidebar-on-step"
                >
                  {index + 1}
                </span>
                <p className="text-caption font-semibold text-sidebar-title">{step.title}</p>
                <p className="mt-0.5 text-meta text-sidebar-text">{step.text}</p>
              </li>
            ))}
          </ol>
        </section>

        <p className="mt-auto border-t border-sidebar-rule pt-3 text-meta text-sidebar-text">{WELLBEING_NOTICE}</p>
      </div>
    </aside>
  );
}
