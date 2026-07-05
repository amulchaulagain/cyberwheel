import { useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { useDeleteSweep, useSweep } from "../api/hooks";
import PageHeader from "../components/PageHeader";
import StatusBadge from "../components/runs/StatusBadge";
import { formatNumber } from "../lib/format";

function shortMetric(tag: string): string {
  return tag.split("/").slice(1).join("/").replace(/_episodic_return$/, "") || tag;
}

export default function SweepPage() {
  const { sweepId } = useParams();
  const navigate = useNavigate();
  const { data: sweep } = useSweep(sweepId, true);
  const remove = useDeleteSweep();

  const metricKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const cell of sweep?.cells ?? [])
      for (const key of Object.keys(cell.metrics ?? {})) keys.add(key);
    return [...keys].sort();
  }, [sweep]);

  if (!sweep) return <div className="p-8 text-sm text-slate-500">Loading…</div>;

  const childIds = sweep.cells.map((c) => c.run_id).join(",");

  return (
    <div>
      <PageHeader
        title={
          <span className="flex items-center gap-3">
            {sweep.display_name} {sweep.status && <StatusBadge status={sweep.status} />}
          </span>
        }
        subtitle={
          <span className="font-mono">
            {sweep.id} · {sweep.base_config} · {sweep.cells.length} runs
          </span>
        }
        actions={
          <>
            <button
              className="btn-ghost"
              onClick={() => navigate(`/compare?runs=${childIds}`)}
            >
              Compare runs
            </button>
            <button
              className="btn-danger"
              disabled={sweep.status === "running" || remove.isPending}
              onClick={() => {
                if (window.confirm(`Delete sweep "${sweep.display_name}" and its runs?`))
                  remove.mutate(
                    { sweepId: sweep.id, artifacts: true },
                    { onSuccess: () => navigate("/") },
                  );
              }}
            >
              Delete
            </button>
          </>
        }
      />
      <div className="p-8">
        <div className="panel overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-[11px] uppercase tracking-wider text-slate-500">
                <th className="px-4 py-2 font-medium">Run</th>
                {sweep.varied_keys.map((key) => (
                  <th key={key} className="px-3 py-2 font-medium">{key}</th>
                ))}
                <th className="px-3 py-2 font-medium">Status</th>
                {metricKeys.map((key) => (
                  <th key={key} className="px-3 py-2 text-right font-medium">{shortMetric(key)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sweep.cells.map((cell) => (
                <tr key={cell.run_id} className="border-t border-ink-700/70 hover:bg-ink-800/50">
                  <td className="px-4 py-2.5">
                    <Link to={`/runs/train/${cell.run_id}`} className="font-mono text-xs text-sky-400">
                      {cell.run_id}
                    </Link>
                  </td>
                  {sweep.varied_keys.map((key) => (
                    <td key={key} className="px-3 py-2.5 font-mono text-slate-300">
                      {String(cell.params[key])}
                    </td>
                  ))}
                  <td className="px-3 py-2.5">
                    {cell.status && cell.status !== "missing" ? (
                      <StatusBadge status={cell.status} />
                    ) : (
                      <span className="text-xs text-slate-600">{cell.status ?? "—"}</span>
                    )}
                  </td>
                  {metricKeys.map((key) => {
                    const value = cell.metrics?.[key];
                    return (
                      <td key={key} className="px-3 py-2.5 text-right font-mono text-slate-300">
                        {value === null || value === undefined ? "—" : formatNumber(value, 1)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
