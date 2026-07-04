import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useDeleteRun, useRuns, useStopRun } from "../api/hooks";
import type { RunRecord } from "../api/types";
import PageHeader from "../components/PageHeader";
import ProgressBar from "../components/runs/ProgressBar";
import StatusBadge from "../components/runs/StatusBadge";
import { formatDuration, formatNumber, formatWhen } from "../lib/format";

type KindFilter = "all" | "train" | "evaluate";

function runPath(run: RunRecord): string {
  return `/runs/${run.kind}/${run.id}`;
}

function StatTile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="panel px-4 py-3">
      <div className="text-[11px] font-medium uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className="mt-1 font-mono text-2xl text-slate-100">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

function RunRow({
  run,
  selected,
  selectable,
  onToggle,
}: {
  run: RunRecord;
  selected: boolean;
  selectable: boolean;
  onToggle: () => void;
}) {
  const navigate = useNavigate();
  const stop = useStopRun();
  const remove = useDeleteRun();
  const active = run.status === "running" || run.status === "queued";
  return (
    <tr
      className="cursor-pointer border-t border-ink-700/70 transition-colors hover:bg-ink-800/50"
      onClick={() => navigate(runPath(run))}
    >
      <td className="w-8 pl-4 pr-1 py-2.5" onClick={(event) => event.stopPropagation()}>
        <input
          type="checkbox"
          className="accent-sky-500"
          checked={selected}
          disabled={!selected && !selectable}
          title={
            !selected && !selectable
              ? "at most 6 runs can be compared at once"
              : "select for comparison"
          }
          onChange={onToggle}
        />
      </td>
      <td className="px-4 py-2.5">
        <div className="text-sm font-medium text-slate-200">{run.display_name}</div>
        <div className="font-mono text-[11px] text-slate-500">{run.id}</div>
      </td>
      <td className="px-3 py-2.5">
        <span
          className={`rounded px-1.5 py-0.5 text-[11px] font-medium uppercase tracking-wide ${
            run.kind === "train"
              ? "bg-indigo-950/70 text-indigo-300"
              : "bg-cyan-950/70 text-cyan-300"
          }`}
        >
          {run.kind}
        </span>
      </td>
      <td className="px-3 py-2.5">
        <StatusBadge status={run.status} />
      </td>
      <td className="w-44 px-3 py-2.5">
        {active || run.status === "succeeded" ? (
          <div className="flex items-center gap-2">
            <ProgressBar value={run.progress} className="w-28" />
            <span className="font-mono text-[11px] text-slate-400">
              {run.progress != null ? `${Math.round(run.progress * 100)}%` : "…"}
            </span>
          </div>
        ) : (
          <span className="text-xs text-slate-600">—</span>
        )}
      </td>
      <td className="px-3 py-2.5 text-right font-mono text-xs text-slate-400">
        {run.kind === "train" && run.last_sps != null ? formatNumber(run.last_sps, 0) : "—"}
      </td>
      <td className="px-3 py-2.5 text-right text-xs text-slate-400">
        {formatWhen(run.created_at)}
      </td>
      <td className="px-3 py-2.5 text-right font-mono text-xs text-slate-400">
        {formatDuration(run.started_at, run.ended_at)}
      </td>
      <td className="px-4 py-2.5 text-right" onClick={(event) => event.stopPropagation()}>
        <div className="flex justify-end gap-1.5">
          {run.kind === "train" && run.status === "succeeded" && (
            <Link
              to={`/evaluate/new?source=${run.id}`}
              className="btn-ghost !px-2 !py-1 !text-xs"
            >
              Evaluate
            </Link>
          )}
          {active ? (
            <button
              className="btn-danger !px-2 !py-1 !text-xs"
              disabled={stop.isPending}
              onClick={() => stop.mutate(run.id)}
            >
              Stop
            </button>
          ) : (
            <button
              className="btn-ghost !px-2 !py-1 !text-xs text-slate-500"
              disabled={remove.isPending}
              onClick={() => {
                if (window.confirm(`Delete run "${run.display_name}" and its artifacts?`))
                  remove.mutate({ runId: run.id, artifacts: true });
              }}
            >
              Delete
            </button>
          )}
        </div>
      </td>
    </tr>
  );
}

export default function DashboardPage() {
  const navigate = useNavigate();
  const { data, isLoading, error } = useRuns();
  const [kind, setKind] = useState<KindFilter>("all");
  // 6 = the metric-chart series palette size.
  const MAX_COMPARE = 6;
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const toggleSelected = (runId: string) =>
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });

  const runs = useMemo(() => {
    const all = data?.runs ?? [];
    return kind === "all" ? all : all.filter((run) => run.kind === kind);
  }, [data, kind]);

  const active = (data?.runs ?? []).filter(
    (run) => run.status === "running" || run.status === "queued",
  ).length;
  const trained = (data?.runs ?? []).filter(
    (run) => run.kind === "train" && run.status === "succeeded",
  ).length;
  const external = data?.external_models ?? [];

  return (
    <div>
      <PageHeader
        title="Dashboard"
        subtitle="All experiment runs"
        actions={
          <>
            <Link to="/evaluate/new" className="btn-ghost">
              New evaluation
            </Link>
            <Link to="/train/new" className="btn-primary">
              New training run
            </Link>
          </>
        }
      />
      <div className="space-y-6 p-8">
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatTile label="Active runs" value={String(active)} />
          <StatTile label="Completed trainings" value={String(trained)} />
          <StatTile
            label="Evaluations"
            value={String((data?.runs ?? []).filter((r) => r.kind === "evaluate").length)}
          />
          <StatTile
            label="External models"
            value={String(external.length)}
            hint={external.length ? "trained outside the UI" : undefined}
          />
        </div>

        <div className="panel overflow-hidden">
          <div className="flex items-center gap-1 border-b border-ink-700 px-3 py-2">
            {(["all", "train", "evaluate"] as KindFilter[]).map((option) => (
              <button
                key={option}
                onClick={() => setKind(option)}
                className={`rounded-md px-3 py-1 text-xs font-medium capitalize transition-colors ${
                  kind === option
                    ? "bg-ink-700 text-slate-100"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                {option === "all" ? "All" : option === "train" ? "Training" : "Evaluations"}
              </button>
            ))}
            {selected.size > 0 && (
              <div className="ml-auto flex items-center gap-1.5">
                <button
                  className="btn-primary !px-3 !py-1 !text-xs"
                  disabled={selected.size < 2}
                  title={selected.size < 2 ? "select at least 2 runs" : undefined}
                  onClick={() => navigate(`/compare?runs=${[...selected].join(",")}`)}
                >
                  Compare ({selected.size})
                </button>
                <button
                  className="btn-ghost !px-2 !py-1 !text-xs text-slate-500"
                  onClick={() => setSelected(new Set())}
                >
                  clear
                </button>
              </div>
            )}
          </div>
          {isLoading ? (
            <div className="px-4 py-12 text-center text-sm text-slate-500">Loading…</div>
          ) : error ? (
            <div className="px-4 py-12 text-center text-sm text-red-400">
              {(error as Error).message}
            </div>
          ) : runs.length === 0 ? (
            <div className="px-4 py-16 text-center">
              <div className="text-sm text-slate-400">No runs yet</div>
              <div className="mt-1 text-xs text-slate-600">
                Start with a training run, then evaluate its checkpoints.
              </div>
              <Link to="/train/new" className="btn-primary mt-4">
                New training run
              </Link>
            </div>
          ) : (
            <table className="w-full text-left">
              <thead>
                <tr className="text-[11px] uppercase tracking-wider text-slate-500">
                  <th className="w-8 pl-4 pr-1 py-2" />
                  <th className="px-4 py-2 font-medium">Run</th>
                  <th className="px-3 py-2 font-medium">Kind</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Progress</th>
                  <th className="px-3 py-2 text-right font-medium">SPS</th>
                  <th className="px-3 py-2 text-right font-medium">Created</th>
                  <th className="px-3 py-2 text-right font-medium">Duration</th>
                  <th className="px-4 py-2" />
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <RunRow
                    key={run.id}
                    run={run}
                    selected={selected.has(run.id)}
                    selectable={selected.size < MAX_COMPARE}
                    onToggle={() => toggleSelected(run.id)}
                  />
                ))}
              </tbody>
            </table>
          )}
        </div>

        {external.length > 0 && (
          <div className="panel">
            <div className="border-b border-ink-700 px-4 py-2.5 text-xs font-medium uppercase tracking-wider text-slate-500">
              External models
            </div>
            <div className="divide-y divide-ink-700/70">
              {external.map((model) => (
                <div key={model.id} className="flex items-center justify-between px-4 py-2.5">
                  <div>
                    <div className="font-mono text-sm text-slate-300">
                      {model.experiment_name}
                    </div>
                    <div className="text-[11px] text-slate-500">
                      agents: {model.agents.join(", ")} · checkpoints:{" "}
                      {model.checkpoints.length}
                    </div>
                  </div>
                  <Link
                    to={`/evaluate/new?experiment=${model.experiment_name}`}
                    className="btn-ghost !px-2 !py-1 !text-xs"
                  >
                    Evaluate
                  </Link>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
