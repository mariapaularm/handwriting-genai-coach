import { Fragment, type ReactNode } from "react";

/**
 * Renderizador mínimo y seguro (sin `dangerouslySetInnerHTML`) para el Markdown que genera el agente:
 * encabezados `###`, viñetas `-`, citas `>`, separadores `---`, **negrita** y citas a fuentes `[archivo.md]`.
 *
 * Convenciones del backend:
 * - `>` solo se usa para el mensaje de seguridad en crisis → se marca como nota destacada.
 * - Lo que sigue a `---` es el aviso de bienestar → se compone como nota al pie.
 */
export default function FeedbackMarkdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let list: string[] = [];
  let paragraph: string[] = [];
  let inFootnote = false;

  const flushList = () => {
    if (list.length) {
      blocks.push(
        <ul className="list-disc space-y-2 pl-5 marker:text-olive" key={`ul-${blocks.length}`}>
          {list.map((item, index) => (
            <li className="pl-1" key={index}>
              {renderInline(item)}
            </li>
          ))}
        </ul>,
      );
      list = [];
    }
  };
  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push(
        <p
          className={inFootnote ? "font-sans text-caption leading-[1.375rem] text-ink-soft" : undefined}
          key={`p-${blocks.length}`}
        >
          {renderInline(paragraph.join(" "))}
        </p>,
      );
      paragraph = [];
    }
  };
  const flush = () => {
    flushList();
    flushParagraph();
  };

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();

    if (!line) {
      flush();
    } else if (/^-{3,}$/.test(line)) {
      flush();
      inFootnote = true;
      blocks.push(<hr className="mt-8 border-line" key={`hr-${blocks.length}`} />);
    } else if (/^#{1,6}\s/.test(line)) {
      flush();
      blocks.push(
        <h4 className="mt-8 font-display text-lead font-semibold text-balance first:mt-0" key={`h-${blocks.length}`}>
          {renderInline(line.replace(/^#{1,6}\s+/, ""))}
        </h4>,
      );
    } else if (/^[-*]\s+/.test(line)) {
      flushParagraph();
      list.push(line.replace(/^[-*]\s+/, ""));
    } else if (line.startsWith(">")) {
      flush();
      blocks.push(
        <aside
          aria-label="Mensaje de seguridad"
          className="rounded-lg border-l-4 border-alert bg-alert-soft px-4 py-3 font-sans text-body text-ink"
          key={`q-${blocks.length}`}
          role="note"
        >
          {renderInline(line.replace(/^>\s?/, ""))}
        </aside>,
      );
    } else {
      flushList();
      paragraph.push(line);
    }
  }
  flush();

  return <div className="prose-serif max-w-[68ch] space-y-4 text-pretty">{blocks}</div>;
}

function renderInline(text: string): ReactNode {
  // Divide en **negrita** y [fuente.md]; el resto se muestra como texto plano.
  const parts = text.split(/(\*\*[^*]+\*\*|\[[\w-]+\.md\])/g);
  return parts.map((part, index) => {
    if (/^\*\*[^*]+\*\*$/.test(part)) {
      return (
        <strong className="font-semibold" key={index}>
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (/^\[[\w-]+\.md\]$/.test(part)) {
      return (
        <span
          className="mx-0.5 inline-block rounded bg-lino px-1.5 align-baseline font-mono text-meta text-salvia"
          key={index}
          title="Fuente de la base de conocimiento"
          translate="no"
        >
          {part.slice(1, -1)}
        </span>
      );
    }
    return <Fragment key={index}>{part}</Fragment>;
  });
}
