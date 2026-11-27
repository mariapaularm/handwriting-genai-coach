import { useEffect, useState } from "react";

const STORAGE_KEY = "handwriting-theme-mode";

export function useThemeMode() {
  const [mode, setMode] = useState(() => {
    const storedMode = window.localStorage.getItem(STORAGE_KEY);

    if (storedMode === "dark" || storedMode === "light") {
      return storedMode;
    }

    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    const root = document.documentElement;

    root.classList.toggle("dark", mode === "dark");
    root.classList.toggle("light", mode === "light");
    root.dataset.theme = mode;
    window.localStorage.setItem(STORAGE_KEY, mode);

    // La barra del navegador acompaña al fondo del tema elegido (no solo al del sistema).
    const background = mode === "dark" ? "#15140F" : "#F2EDE4";
    document.querySelectorAll('meta[name="theme-color"]').forEach((meta) => {
      meta.setAttribute("content", background);
    });
  }, [mode]);

  function toggleMode() {
    setMode((currentMode) => (currentMode === "dark" ? "light" : "dark"));
  }

  return { mode, toggleMode };
}
