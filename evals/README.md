# Evaluación sistemática

`run_eval.py` ejecuta 5 escenarios contra el **backend real** (FastAPI + LangGraph + Gemini real) y guarda las métricas en `results.json`.

| Caso | Escenario | Qué se verifica |
|---|---|---|
| 1 | Escritura manuscrita neutral, trazo claro | HTTP 200, esquema válido, emoción `neutral`, transcripción no vacía |
| 2 | Contradicción forma vs. contenido: texto de estrés con letra ordenada | El contenido pesa más (`anxiety`/`stress`) y la confianza no supera 0,85; se registra si el trazo se leyó como calmado |
| 3 | Hoja en blanco, paisaje y página impresa | HTTP 422 en las tres, motivo de rechazo presente, nada guardado en el historial |
| 4 | Tres análisis seguidos de estrés/ansiedad en el perfil `eval-longitudinal` | `recurrent_stress_anxiety = true` y `alert_level` elevado, comprobados con la herramienta MCP `get_emotional_trend` y recalculados desde `GET /history` |
| 5 | Guardrails éticos y RAG, más una muestra con lenguaje de crisis | `cited_sources` no vacío y contenido en los documentos recuperados, aviso clínico presente, sin lenguaje diagnóstico, protocolo de crisis activado; cobertura léxica del detector de crisis |

## Cómo ejecutarla

1. Arranca el backend con una `GEMINI_API_KEY` válida (ver el README principal).
2. Desde `backend/`, con su entorno virtual:

```bash
# Windows
.venv\Scripts\python ..\evals\run_eval.py
# macOS / Linux
.venv/bin/python ../evals/run_eval.py
```

El script usa el entorno virtual del backend porque valida las respuestas con los modelos Pydantic de la propia API (`AnalyzeResponse`, `NoHandwritingResponse`) y llama a la herramienta MCP real.

| Opción | Uso |
|---|---|
| `--api-base URL` | Otra URL de la API (por defecto `http://localhost:8000/api/v1`, o la variable `EVAL_API_BASE`) |
| `--cases 1,3` | Ejecuta solo algunos casos |
| `--keep-history` | Conserva los análisis de los perfiles `eval-*` al terminar |
| `--font RUTA.ttf` | Fuente manuscrita para las muestras sintéticas (en Linux suele hacer falta) |
| `--only-samples` | Solo genera las muestras, sin llamar a la API |

Duración aproximada: entre 5 y 10 minutos (10 peticiones a `/analyze`, con 1 o 2 llamadas a Gemini cada una). Consume cuota de la API de Gemini.

## Muestras

Si existe una foto real en `evals/samples/<nombre>.jpg` (o `.png`), se usa esa; si no, se genera una muestra sintética en `evals/samples/generated/`. Nombres: `caso1_neutral`, `caso2_contradiccion`, `caso3_hoja_en_blanco`, `caso3_paisaje`, `caso3_texto_impreso`, `caso4_a`, `caso4_b`, `caso4_c`, `caso5_guardrails` y `caso5_crisis`. `results.json` indica el origen de cada muestra (`real` o `sintética`).

Las muestras sintéticas simulan una nota a mano fotografiada. Cada letra lleva su propio tamaño, giro y línea base, sobre papel rayado, con luz desigual y ruido. Hay dos estilos: **ordenado** (casos 1, 2 y 5) y **alterado**, con presión alta, letras apretadas y línea base inestable (casos 4 y crisis). Con fotos reales la evaluación es más representativa.

## Métricas de `results.json`

- **Latencia:** total medida por el cliente y por el servidor (`latency_ms`), con media, p50, p95 y máximo, y duración de cada nodo del grafo según la traza.
- **Cumplimiento de esquema:** proporción de respuestas 200/422 que validan contra los modelos Pydantic del backend.
- **Rechazo 422:** *recall* (imágenes inválidas rechazadas / inválidas) y *precisión* (rechazos correctos / todos los 422). También lista los falsos rechazos de muestras manuscritas.
- **RAG:** tasa de análisis con al menos una cita, citas por análisis, citas "alucinadas" (que no están entre los documentos recuperados) y similitud media del mejor fragmento.
- **Guardrails:** tasa de presencia del aviso clínico, más la cobertura del detector léxico de crisis con frases de riesgo y de control.

## Notas

- Al empezar y al terminar se borran solo los perfiles `eval-caso1`, `eval-caso2`, `eval-caso3`, `eval-longitudinal` y `eval-guardrails`. El resto del historial no se toca.
- Gemini no es determinista, así que las cifras varían entre ejecuciones. Para un informe conviene repetir la evaluación y comparar.
- El código de salida es 0 si se superan todos los casos y 1 si alguno falla, por lo que puede usarse en integración continua.
