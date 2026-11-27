import { useCallback, useEffect, useState } from "react";

import Dashboard from "./components/dashboard/Dashboard";
import AppShell from "./components/layout/AppShell.jsx";
import { useThemeMode } from "./hooks/useThemeMode.js";
import { checkHealth } from "./services/api";

const HEALTH_INTERVAL_MS = 15_000;

export default function App() {
  const { mode, toggleMode } = useThemeMode();
  const [backendStatus, setBackendStatus] = useState("checking");

  const refreshHealth = useCallback(async () => {
    setBackendStatus((await checkHealth()) ? "online" : "offline");
  }, []);

  useEffect(() => {
    void refreshHealth();
    const timer = window.setInterval(() => void refreshHealth(), HEALTH_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [refreshHealth]);

  return (
    <AppShell backendStatus={backendStatus} mode={mode} onToggleMode={toggleMode}>
      <Dashboard onHistoryChanged={refreshHealth} />
    </AppShell>
  );
}
