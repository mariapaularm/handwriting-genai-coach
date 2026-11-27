import { Alert, Button, Card, Description, FieldError, Input, Label, Spinner, TextField } from "@heroui/react";
import { CheckCircle2, ImageUp, Info, ScanText, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { type ApiError, PROFILE_NAME_PATTERN } from "../../services/api";

type UploaderProps = {
  profileName: string;
  onProfileNameChange: (value: string) => void;
  isLoading: boolean;
  error: ApiError | null;
  lastSuccessAt: string | null;
  wasCancelled: boolean;
  onSubmit: (file: File) => void;
  onCancel: () => void;
  onDismissError: () => void;
};

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const VALID_TYPES = ["image/jpeg", "image/png"];
const sizeFormat = new Intl.NumberFormat("es", { maximumFractionDigits: 0 });

export default function Uploader({
  profileName,
  onProfileNameChange,
  isLoading,
  error,
  lastSuccessAt,
  wasCancelled,
  onSubmit,
  onCancel,
  onDismissError,
}: UploaderProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const dropzoneRef = useRef<HTMLButtonElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [fileError, setFileError] = useState("");

  const isProfileValid = PROFILE_NAME_PATTERN.test(profileName);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function selectFile(nextFile?: File) {
    onDismissError();
    const validationError = validateFile(nextFile);
    setFileError(validationError);
    setFile(validationError ? null : (nextFile ?? null));
  }

  function removeFile() {
    setFile(null);
    setFileError("");
    onDismissError();
    if (inputRef.current) inputRef.current.value = "";
    dropzoneRef.current?.focus();
  }

  function handleAnalyze() {
    if (!isProfileValid) {
      document.getElementById("profile-name")?.focus();
      return;
    }
    if (!file) {
      setFileError("Elige una imagen antes de analizar.");
      dropzoneRef.current?.focus();
      return;
    }
    onSubmit(file);
  }

  const openPicker = () => inputRef.current?.click();

  return (
    <Card className="surface-card scroll-mt-6" id="muestra" variant="default">
      <Card.Header className="block">
        <h2 className="font-display text-h2 font-medium">Tu muestra</h2>
        <p className="mt-1 text-caption text-ink-soft">
          Una foto nítida de texto escrito a mano, en JPG o PNG de hasta 10&nbsp;MB.
        </p>
      </Card.Header>

      <Card.Content className="space-y-5">
        <TextField
          fullWidth
          isDisabled={isLoading}
          isInvalid={!isProfileValid}
          name="profile_name"
          onChange={onProfileNameChange}
          value={profileName}
        >
          <Label>Perfil</Label>
          <Input
            autoComplete="off"
            className="border border-field-line bg-field-soft shadow-none"
            id="profile-name"
            placeholder="p. ej. maria-diario…"
            spellCheck={false}
          />
          {isProfileValid ? (
            <Description>Tu historial y tu tendencia se guardan con este nombre.</Description>
          ) : (
            <FieldError>Usa de 1 a 64 letras, números, «-» o «_», sin espacios.</FieldError>
          )}
        </TextField>

        <input
          accept="image/png,image/jpeg"
          aria-hidden="true"
          className="hidden"
          onChange={(event) => selectFile(Array.from(event.target.files ?? [])[0])}
          ref={inputRef}
          tabIndex={-1}
          type="file"
        />

        <button
          aria-describedby="dropzone-help"
          className={`dropzone flex w-full cursor-pointer flex-col items-center justify-center gap-3 p-5 text-center ${
            isLoading ? "pointer-events-none opacity-60" : ""
          }`}
          data-dragging={isDragging}
          disabled={isLoading}
          onClick={openPicker}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            setIsDragging(false);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            setIsDragging(false);
            selectFile(Array.from(event.dataTransfer.files)[0]);
          }}
          ref={dropzoneRef}
          type="button"
        >
          {previewUrl ? (
            <img
              alt="Vista previa de tu muestra manuscrita"
              className="h-56 w-full rounded-md bg-paper object-contain"
              height={448}
              src={previewUrl}
              width={640}
            />
          ) : (
            <span aria-hidden="true" className="grid size-14 place-items-center rounded-full bg-paper text-olive">
              <ImageUp size={26} />
            </span>
          )}

          <span className="block max-w-sm min-w-0">
            <span className="block text-body font-semibold break-words">
              {file ? file.name : "Arrastra aquí tu foto o pulsa para elegirla"}
            </span>
            <span className="tabular mt-0.5 block text-caption text-ink-soft" id="dropzone-help">
              {file
                ? `${sizeFormat.format(file.size / 1024)} KB. Pulsa para cambiarla.`
                : "JPG o PNG, máximo 10 MB"}
            </span>
          </span>
        </button>

        {!file ? (
          <p className="flex gap-2 text-caption text-ink-soft">
            <Info aria-hidden="true" className="mt-0.5 shrink-0 text-salvia" size={16} />
            Para una lectura fiable: luz uniforme, la hoja entera en el encuadre y al menos tres líneas de texto.
          </p>
        ) : null}

        <div aria-live="polite" className="space-y-3 empty:hidden">
          {fileError ? (
            <Alert status="danger">
              <Alert.Indicator />
              <Alert.Content>
                <Alert.Title>No podemos usar ese archivo</Alert.Title>
                <Alert.Description>{fileError}</Alert.Description>
              </Alert.Content>
            </Alert>
          ) : null}

          {error ? <ApiErrorAlert error={error} onDismiss={onDismissError} /> : null}

          {wasCancelled && !isLoading ? (
            <p className="rounded-lg bg-lino px-3 py-2.5 text-caption text-ink">
              Dejaste de esperar el análisis. Si el servidor llegó a terminarlo, aparecerá en tu historial.
            </p>
          ) : null}

          {lastSuccessAt && !isLoading && !error ? (
            <p className="flex items-center gap-2 rounded-lg bg-olive-soft px-3 py-2.5 text-caption font-medium text-olive">
              <CheckCircle2 aria-hidden="true" size={16} />
              Análisis completado y guardado en tu historial.
            </p>
          ) : null}
        </div>
      </Card.Content>

      <Card.Footer className="flex flex-row flex-wrap gap-2">
        {isLoading ? (
          <>
            <Button className="w-full sm:w-auto" isPending variant="primary">
              <Spinner color="current" size="sm" />
              Analizando…
            </Button>
            <Button className="w-full sm:w-auto" onPress={onCancel} variant="secondary">
              <X aria-hidden="true" size={17} />
              Cancelar análisis
            </Button>
          </>
        ) : (
          <>
            <Button className="w-full sm:w-auto" onPress={handleAnalyze} variant="primary">
              <ScanText aria-hidden="true" size={17} />
              Analizar muestra
            </Button>
            {file ? (
              <Button className="w-full sm:w-auto" onPress={removeFile} variant="ghost">
                <X aria-hidden="true" size={17} />
                Quitar imagen
              </Button>
            ) : null}
          </>
        )}
      </Card.Footer>
    </Card>
  );
}

function ApiErrorAlert({ error, onDismiss }: { error: ApiError; onDismiss: () => void }) {
  const isNoHandwriting = error.status === 422 && Boolean(error.reason);
  const hint = HINTS[error.status ?? 0] ?? HINTS[0];

  return (
    <Alert status={isNoHandwriting ? "warning" : "danger"}>
      <Alert.Indicator />
      <Alert.Content className="min-w-0">
        <Alert.Title>
          {error.title}
          {error.status ? (
            <span className="tabular ml-2 font-mono text-meta opacity-80" translate="no">
              HTTP {error.status}
            </span>
          ) : null}
        </Alert.Title>
        <Alert.Description className="block space-y-1.5">
          <span className="block break-words">{error.detail}</span>
          {hint ? <span className="block text-meta">{hint}</span> : null}
        </Alert.Description>
      </Alert.Content>
      <Button aria-label="Cerrar aviso" className="shrink-0" isIconOnly onPress={onDismiss} size="sm" variant="ghost">
        <X aria-hidden="true" size={16} />
      </Button>
    </Alert>
  );
}

const HINTS: Record<number, string> = {
  0: "Comprueba que el backend esté arrancado (uvicorn en el puerto 8000).",
  400: "Sube una imagen PNG o JPEG legible de menos de 10 MB.",
  422: "Prueba con una foto nítida de una hoja escrita a mano, no de texto impreso ni de una pantalla.",
  429: "Espera unos segundos y vuelve a analizar la muestra.",
  502: "Vuelve a analizar la muestra; si se repite, revisa los registros del backend.",
  503: "Revisa GEMINI_API_KEY y GEMINI_MODEL en el .env del backend.",
  504: "El modelo está saturado. Vuelve a intentarlo o usa un modelo más rápido en GEMINI_MODEL.",
};

function validateFile(file?: File): string {
  if (!file) return "Elige una imagen con texto escrito a mano.";
  if (!VALID_TYPES.includes(file.type)) return "Ese formato no es compatible. Usa una imagen JPG o PNG.";
  if (file.size > MAX_FILE_SIZE) return "La imagen pesa más de 10 MB. Redúcela o recórtala y vuelve a subirla.";
  return "";
}
