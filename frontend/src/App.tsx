import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { useQuery } from "@tanstack/react-query";
import { api } from "./api/client";
import ComparePage from "./pages/ComparePage";
import DashboardPage from "./pages/DashboardPage";
import EvaluationRunPage from "./pages/EvaluationRunPage";
import NewEvaluationPage from "./pages/NewEvaluationPage";
import NewTrainingPage from "./pages/NewTrainingPage";
import TrainingRunPage from "./pages/TrainingRunPage";

function Logo() {
  return (
    <div className="flex items-center gap-2.5 px-4 py-5">
      <svg viewBox="0 0 32 32" className="h-7 w-7">
        <circle cx="16" cy="16" r="13" fill="none" stroke="#5eb0ff" strokeWidth="3" />
        <circle cx="16" cy="16" r="5" fill="#5eb0ff" />
      </svg>
      <div>
        <div className="text-sm font-semibold tracking-[0.2em] text-slate-100">
          CYBERWHEEL
        </div>
        <div className="text-[10px] uppercase tracking-widest text-slate-500">
          experimentation
        </div>
      </div>
    </div>
  );
}

function NavItem({ to, label, end }: { to: string; label: string; end?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `mx-2 flex items-center rounded-md px-3 py-2 text-sm transition-colors ${
          isActive
            ? "bg-ink-700 font-medium text-slate-100"
            : "text-slate-400 hover:bg-ink-800 hover:text-slate-200"
        }`
      }
    >
      {label}
    </NavLink>
  );
}

function HealthDot() {
  const health = useQuery<{ status: string; active_runs: number }>({
    queryKey: ["health"],
    queryFn: () => api.get("/api/health"),
    refetchInterval: 5000,
    retry: false,
  });
  const ok = health.data?.status === "ok";
  return (
    <div className="mx-4 mb-4 mt-auto flex items-center gap-2 border-t border-ink-700 pt-4 text-xs text-slate-500">
      <span
        className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-400" : "bg-red-500"}`}
      />
      {ok
        ? `server ok · ${health.data?.active_runs ?? 0} active`
        : "server unreachable"}
    </div>
  );
}

export default function App() {
  return (
    <div className="flex h-screen">
      <aside className="flex w-56 shrink-0 flex-col border-r border-ink-700 bg-ink-900">
        <Logo />
        <nav className="mt-2 flex flex-col gap-0.5">
          <NavItem to="/" label="Dashboard" end />
          <NavItem to="/train/new" label="New training run" />
          <NavItem to="/evaluate/new" label="New evaluation" />
        </nav>
        <HealthDot />
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/train/new" element={<NewTrainingPage />} />
          <Route path="/evaluate/new" element={<NewEvaluationPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/runs/train/:runId" element={<TrainingRunPage />} />
          <Route path="/runs/evaluate/:runId" element={<EvaluationRunPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
