import { Button, Tooltip } from "@heroui/react";
import { Moon, PenLine, Sun } from "lucide-react";

import AppSidebar, { ConnectionStatus, WELLBEING_NOTICE } from "./AppSidebar.jsx";

export default function AppShell({ children, mode, backendStatus, onToggleMode }) {
  const ThemeIcon = mode === "dark" ? Sun : Moon;
  const nextModeLabel = mode === "dark" ? "Usar tema claro" : "Usar tema oscuro";

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <a className="skip-link" href="#contenido">
        Saltar al contenido
      </a>

      <div className="flex min-h-screen">
        <AppSidebar backendStatus={backendStatus} />

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="px-4 pt-6 pb-2 md:px-8 md:pt-10">
            <div className="mx-auto flex max-w-7xl items-start justify-between gap-6">
              <div className="min-w-0 max-w-[62ch]">
                <p className="mb-4 flex items-center gap-2 text-caption font-semibold text-olive lg:hidden">
                  <PenLine aria-hidden="true" size={18} />
                  <span translate="no">Handwriting GenAI Coach</span>
                </p>
                <h1 className="font-display text-h1 font-medium tracking-[-0.01em] text-balance md:text-[2.75rem] md:leading-[3.125rem]">
                  Lo que tu escritura cuenta de cómo estás
                </h1>
                <p className="mt-3 text-lead text-pretty text-ink-soft">
                  Fotografía una hoja escrita a mano. El coach describe tu trazo y te devuelve una lectura orientativa
                  de tu bienestar, con consejos para hoy.
                </p>
              </div>

              <div className="flex shrink-0 items-center gap-2 pt-1">
                <div className="lg:hidden">
                  <ConnectionStatus status={backendStatus} />
                </div>
                <Tooltip>
                  <Tooltip.Trigger>
                    <Button
                      aria-label={nextModeLabel}
                      className="border border-card-line bg-paper"
                      isIconOnly
                      onPress={onToggleMode}
                      variant="secondary"
                    >
                      <ThemeIcon aria-hidden="true" size={18} />
                    </Button>
                  </Tooltip.Trigger>
                  <Tooltip.Content showArrow>
                    <Tooltip.Arrow />
                    {nextModeLabel}
                  </Tooltip.Content>
                </Tooltip>
              </div>
            </div>
          </header>

          <main className="flex-1 px-4 py-6 md:px-8 md:py-8" id="contenido" tabIndex={-1}>
            {children}
          </main>

          <footer className="border-t border-line px-4 py-5 text-meta text-ink-soft md:px-8 lg:hidden">
            <p className="mx-auto max-w-[72ch]">{WELLBEING_NOTICE}</p>
          </footer>
        </div>
      </div>
    </div>
  );
}
