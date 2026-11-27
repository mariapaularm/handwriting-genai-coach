import { Card, ProgressBar, Spinner } from "@heroui/react";
import { Check } from "lucide-react";
import { useEffect, useState } from "react";

type AnalysisProgressProps = {
  startedAt: number;
  uploadPercent: number;
  profileName: string;
};

type StepState = "done" | "active" | "pending";

// Umbrales orientativos (segundos tras la subida) para resaltar la etapa probable.
// El backend no emite progreso: los tiempos reales llegan en la traza al terminar.
const VISUAL_ESTIMATE_S = 20;
const RETRIEVAL_ESTIMATE_S = 4;

export default function AnalysisProgress({ startedAt, uploadPercent, profileName }: AnalysisProgressProps) {
  const [now, setNow] = useState(() => Date.now());
  const [uploadDoneAt, setUploadDoneAt] = useState<number | null>(null);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, []);

  const elapsed = Math.max(0, (now - startedAt) / 1000);
  // Si el navegador no informa el progreso de subida, se da por enviada a los pocos segundos.
  const uploaded = uploadPercent >= 100 || elapsed > 8;

  useEffect(() => {
    if (uploaded && uploadDoneAt === null) setUploadDoneAt(Date.now());
  }, [uploaded, uploadDoneAt]);

  const sinceUpload = uploadDoneAt ? Math.max(0, (now - uploadDoneAt) / 1000) : 0;
  const stage = !uploaded
    ? 0
    : sinceUpload < VISUAL_ESTIMATE_S
      ? 1
      : sinceUpload < VISUAL_ESTIMATE_S + RETRIEVAL_ESTIMATE_S
        ? 2
        : 3;

  const stateOf = (index: number): StepState => (index < stage ? "done" : index === stage ? "active" : "pending");

  const steps = [
    {
      title: "Subir y limpiar la imagen",
      description: uploaded ? "Imagen validada y binarizada con OpenCV." : `Enviando la imagen: ${uploadPercent} %`,
    },
    {
      title: "Mirar el trazo",
      description: "Gemini detecta la escritura, la transcribe y puntúa cinco rasgos del trazo.",
    },
    {
      title: "Recordar y consultar",
      description: `A la vez: tendencia del perfil «${profileName}» (MCP) y fragmentos de la base de conocimiento (RAG).`,
    },
    {
      title: "Redactar la lectura",
      description: "Retroalimentación personalizada, citas verificadas y aviso de bienestar.",
    },
  ];

  return (
    <Card aria-busy="true" className="surface-card" variant="default">
      <Card.Header className="flex flex-row flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="font-display text-h2 font-medium">Leyendo tu muestra…</h2>
          <p className="mt-1 text-caption text-ink-soft">
            Suele tardar entre 20&nbsp;s y 1&nbsp;min, según la carga de Gemini.
          </p>
        </div>
        <p className="tabular flex items-center gap-2 rounded-full bg-lino px-3 py-1 text-caption font-semibold text-salvia">
          <Spinner color="current" size="sm" />
          <span>
            {Math.floor(elapsed)}&nbsp;s<span className="sr-only"> transcurridos</span>
          </span>
        </p>
      </Card.Header>

      <Card.Content className="space-y-6">
        <ProgressBar aria-label="Subida de la imagen" value={uploaded ? 100 : uploadPercent}>
          <div className="mb-2 flex items-center justify-between gap-3 text-caption">
            <span className="font-medium">{uploaded ? "Imagen enviada" : "Subiendo imagen…"}</span>
            <ProgressBar.Output className="tabular font-semibold" />
          </div>
          <ProgressBar.Track className="h-1.5 rounded-full bg-lino">
            <ProgressBar.Fill className="rounded-full bg-salvia" />
          </ProgressBar.Track>
        </ProgressBar>

        <ol className="relative space-y-1">
          {steps.map((step, index) => {
            const state = stateOf(index);
            return (
              <li
                aria-current={state === "active" ? "step" : undefined}
                className={`relative flex items-start gap-4 rounded-lg px-3 py-3 ${
                  state === "active" ? "bg-lino" : ""
                }`}
                key={step.title}
              >
                <span
                  aria-hidden="true"
                  className={`tabular grid size-7 shrink-0 place-items-center rounded-full text-meta font-semibold ${
                    state === "done"
                      ? "bg-salvia text-paper"
                      : state === "active"
                        ? "step-active border-2 border-salvia bg-paper text-salvia"
                        : "border border-line bg-paper text-ink-soft"
                  }`}
                >
                  {state === "done" ? <Check size={15} strokeWidth={3} /> : index + 1}
                </span>
                <div className={`min-w-0 ${state === "pending" ? "text-ink-soft" : ""}`}>
                  <p className="text-body font-semibold">
                    {step.title}
                    <span className="sr-only">
                      {state === "done" ? " (hecho)" : state === "active" ? " (en curso)" : " (pendiente)"}
                    </span>
                  </p>
                  <p className="text-caption text-ink-soft">{step.description}</p>
                </div>
              </li>
            );
          })}
        </ol>

        <p aria-live="polite" className="sr-only">
          {`Paso ${stage + 1} de ${steps.length}: ${steps[stage].title}.`}
        </p>

        <p className="text-meta text-ink-soft">
          El paso resaltado es una estimación. Al terminar verás el tiempo real de cada etapa en la traza.
        </p>
      </Card.Content>
    </Card>
  );
}
