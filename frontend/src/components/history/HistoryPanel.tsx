import { Alert, Button, Card, Modal, Spinner, useOverlayState } from "@heroui/react";
import { RefreshCcw, Trash2 } from "lucide-react";

import { emotionMeta, formatDate, formatPercent } from "../../lib/presentation";
import type { ApiError, HistoryRecord } from "../../services/api";

export type HistoryScope = "profile" | "all";

type HistoryPanelProps = {
  records: HistoryRecord[];
  scope: HistoryScope;
  profileName: string;
  dominant: { label: string; count: number } | null;
  isLoading: boolean;
  isDeleting: boolean;
  error: ApiError | null;
  selectedId: string | null;
  onScopeChange: (scope: HistoryScope) => void;
  onRefresh: () => void;
  onSelect: (record: HistoryRecord) => void;
  onClear: () => Promise<void>;
};

const COLUMNS = "md:grid-cols-[9.5rem_8rem_9rem_7rem_minmax(0,1fr)]";

function countLabel(count: number) {
  return count === 1 ? "1 análisis" : `${count} análisis`;
}

export default function HistoryPanel({
  records,
  scope,
  profileName,
  dominant,
  isLoading,
  isDeleting,
  error,
  selectedId,
  onScopeChange,
  onRefresh,
  onSelect,
  onClear,
}: HistoryPanelProps) {
  const confirmModal = useOverlayState();
  const target = scope === "profile" ? `del perfil «${profileName}»` : "de todos los perfiles";

  const summary = records.length
    ? `${countLabel(records.length)} ${target}.${
        dominant ? ` La emoción más frecuente es ${dominant.label.toLowerCase()} (${dominant.count}).` : ""
      }`
    : null;

  return (
    <Card className="surface-card" variant="default">
      <Card.Header className="flex flex-row flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <h2 className="font-display text-h2 font-medium">Tu historial</h2>
          <p aria-live="polite" className="tabular mt-1 text-caption text-ink-soft">
            {isLoading && !records.length ? "Cargando historial…" : (summary ?? "Elige un análisis para volver a leerlo.")}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div aria-label="Qué análisis mostrar" className="flex rounded-lg border border-line p-0.5" role="group">
            <Button
              aria-pressed={scope === "profile"}
              onPress={() => onScopeChange("profile")}
              size="sm"
              variant={scope === "profile" ? "primary" : "ghost"}
            >
              <span className="max-w-[12rem] truncate">
                Perfil «<span translate="no">{profileName}</span>»
              </span>
            </Button>
            <Button
              aria-pressed={scope === "all"}
              onPress={() => onScopeChange("all")}
              size="sm"
              variant={scope === "all" ? "primary" : "ghost"}
            >
              Todos los perfiles
            </Button>
          </div>
          <Button aria-label="Recargar historial" isDisabled={isLoading} isIconOnly onPress={onRefresh} size="sm" variant="secondary">
            {isLoading ? <Spinner color="current" size="sm" /> : <RefreshCcw aria-hidden="true" size={15} />}
          </Button>
          {/* Botón nativo: icono y texto heredan el mismo currentColor (sin opacidades de HeroUI). */}
          <button
            aria-busy={isDeleting || undefined}
            className="btn-clay"
            disabled={!records.length || isDeleting}
            onClick={confirmModal.open}
            type="button"
          >
            {isDeleting ? <Spinner color="current" size="sm" /> : <Trash2 aria-hidden="true" size={15} strokeWidth={2} />}
            {isDeleting ? "Borrando…" : "Borrar historial"}
          </button>
        </div>
      </Card.Header>

      <Card.Content className="space-y-3">
        {error ? (
          <Alert status="danger">
            <Alert.Indicator />
            <Alert.Content>
              <Alert.Title>{error.title}</Alert.Title>
              <Alert.Description>
                {error.detail} Pulsa el botón de recargar cuando el backend vuelva a estar disponible.
              </Alert.Description>
            </Alert.Content>
          </Alert>
        ) : null}

        {!records.length && !isLoading && !error ? (
          <p className="rounded-lg border border-dashed border-line-strong px-4 py-8 text-center text-caption text-ink-soft">
            Aún no hay análisis guardados {target}. Cuando analices una muestra, aparecerá aquí.
          </p>
        ) : null}

        {records.length ? (
          <div>
            <div
              aria-hidden="true"
              className={`hidden gap-4 px-3 pb-2 text-meta font-semibold text-ink-soft md:grid ${COLUMNS}`}
            >
              <span>Fecha</span>
              <span>Perfil</span>
              <span>Emoción</span>
              <span>Confianza</span>
              <span>Lo que escribiste</span>
            </div>
            <ul className="space-y-1.5">
              {records.map((record) => (
                <li key={record.id}>
                  <HistoryRow isSelected={record.id === selectedId} onSelect={() => onSelect(record)} record={record} />
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </Card.Content>

      <Modal state={confirmModal}>
        <Modal.Backdrop variant="blur">
          <Modal.Container placement="center" size="sm">
            <Modal.Dialog className="overscroll-contain">
              <Modal.CloseTrigger aria-label="Cerrar" />
              <Modal.Header>
                <Modal.Icon className="bg-danger-soft text-danger">
                  <Trash2 aria-hidden="true" size={20} />
                </Modal.Icon>
                <Modal.Heading>¿Borrar {countLabel(records.length)}?</Modal.Heading>
              </Modal.Header>
              <Modal.Body>
                <p className="text-caption text-ink-soft">
                  Se borrarán los análisis {target} y sus imágenes del servidor. No se puede deshacer.
                </p>
              </Modal.Body>
              <Modal.Footer>
                <Button onPress={confirmModal.close} variant="secondary">
                  Conservar historial
                </Button>
                <Button
                  isPending={isDeleting}
                  onPress={async () => {
                    await onClear();
                    confirmModal.close();
                  }}
                  variant="danger"
                >
                  Borrar {countLabel(records.length)}
                </Button>
              </Modal.Footer>
            </Modal.Dialog>
          </Modal.Container>
        </Modal.Backdrop>
      </Modal>
    </Card>
  );
}

function HistoryRow({
  record,
  isSelected,
  onSelect,
}: {
  record: HistoryRecord;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const meta = emotionMeta(record.emotion);
  const confidence = Math.round((record.confidence ?? 0) * 100);

  return (
    <button
      aria-current={isSelected ? "true" : undefined}
      className={`grid w-full cursor-pointer grid-cols-[1fr_auto] items-center gap-x-4 gap-y-2 rounded-lg border px-3 py-3 text-left transition-colors ${COLUMNS} ${
        isSelected
          ? "border-olive bg-olive-soft"
          : "border-transparent bg-lino/60 hover:border-line-strong hover:bg-lino"
      }`}
      onClick={onSelect}
      type="button"
    >
      <span className="tabular text-caption font-medium">{formatDate(record.created_at)}</span>
      <span className="truncate text-right font-mono text-meta text-ink-soft md:text-left" translate="no">
        {record.profile_name}
      </span>
      <span>
        <span className={meta.tagClass}>{meta.label}</span>
      </span>
      <span className="flex items-center justify-end gap-2 md:justify-start">
        <span aria-hidden="true" className="h-1.5 w-12 overflow-hidden rounded-full bg-paper">
          <span className="block h-full rounded-full bg-olive" style={{ width: `${confidence}%` }} />
        </span>
        <span className="tabular text-caption font-semibold">{formatPercent(confidence)}</span>
      </span>
      <span className="col-span-2 truncate font-serif text-body text-ink-soft italic md:col-span-1">
        {record.transcribed_text || "Sin transcripción"}
      </span>
    </button>
  );
}
