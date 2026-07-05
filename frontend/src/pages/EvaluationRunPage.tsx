import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";

import {
  useActions,
  useEvalSummary,
  useRun,
  useVizEpisode,
  useVizLayout,
  useVizMeta,
} from "../api/hooks";
import type { StatBlock } from "../api/types";
import PageHeader from "../components/PageHeader";
import ProgressBar from "../components/runs/ProgressBar";
import StatusBadge from "../components/runs/StatusBadge";
import Legend from "../components/viz/Legend";
import NetworkCanvas from "../components/viz/NetworkCanvas";
import NodeDetails from "../components/viz/NodeDetails";
import StepScrubber from "../components/viz/StepScrubber";
import { EpisodePlayer, type RenderNode } from "../components/viz/frameState";
import { formatNumber } from "../lib/format";

const METRIC_LABELS: Record<string, string> = {
  total_reward: "Total reward",
  blue_reward: "Blue reward",
  red_reward: "Red reward",
};

function SummaryTile({ label, stat }: { label: string; stat: StatBlock }) {
  const half =
    stat.ci95_hi !== null && stat.ci95_lo !== null
      ? (stat.ci95_hi - stat.ci95_lo) / 2
      : null;
  return (
    <div className="panel px-3 py-2">
      <div className="text-[10px] font-medium uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className="font-mono text-lg text-slate-100">
        {stat.mean === null ? (
          "—"
        ) : (
          <>
            {formatNumber(stat.mean, 1)}
            {half !== null && half > 0 && (
              <span className="text-sm text-slate-400"> ± {formatNumber(half, 1)}</span>
            )}
          </>
        )}
      </div>
      <div className="text-[10px] text-slate-600">n={stat.n}</div>
    </div>
  );
}

function RewardTile({ label, value, tone }: { label: string; value: number; tone?: "up" | "down" }) {
  return (
    <div className="panel px-3 py-2">
      <div className="text-[10px] font-medium uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div
        className={`font-mono text-lg ${
          tone === "up" ? "text-emerald-400" : tone === "down" ? "text-rose-400" : "text-slate-100"
        }`}
      >
        {formatNumber(value, 1)}
      </div>
    </div>
  );
}

export default function EvaluationRunPage() {
  const { runId } = useParams();
  const { data: run } = useRun(runId);
  const active = run?.status === "running" || run?.status === "queued";
  const { data: meta } = useVizMeta(runId, Boolean(active));
  const { data: summary } = useEvalSummary(runId, Boolean(active));
  const hasViz = Boolean(meta?.episodes_written?.length);

  const [episode, setEpisode] = useState(0);
  const [step, setStep] = useState(0);
  const [selected, setSelected] = useState<RenderNode | null>(null);

  const episodesWritten = meta?.episodes_written ?? [];
  useEffect(() => {
    if (episodesWritten.length && !episodesWritten.includes(episode))
      setEpisode(episodesWritten[episodesWritten.length - 1]);
  }, [meta]);

  const { data: layout } = useVizLayout(runId, hasViz);
  const { data: episodeData } = useVizEpisode(runId, episode, hasViz);
  const { data: actions } = useActions(runId, episode);

  const player = useMemo(
    () => (layout && episodeData ? new EpisodePlayer(layout, episodeData) : null),
    [layout, episodeData],
  );
  useEffect(() => setStep(0), [episode]);

  const renderState = useMemo(() => player?.stateAt(step) ?? null, [player, step]);
  const attackedIds = useMemo(() => {
    const ids = new Set<number>();
    if (!episodeData) return ids;
    for (let i = 0; i <= step && i < episodeData.steps.length; i++) {
      const frame = episodeData.steps[i];
      if (frame.da && frame.red.dst !== undefined && frame.red.dst >= 0)
        ids.add(frame.red.dst);
    }
    return ids;
  }, [episodeData, step]);

  const currentFrame = episodeData?.steps[step];
  const rewardTotals = actions?.reward_totals?.[String(episode)];

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title={
          run ? (
            <span className="flex items-center gap-3">
              {run.display_name} <StatusBadge status={run.status} />
            </span>
          ) : (
            "Evaluation"
          )
        }
        subtitle={
          run ? (
            <span className="font-mono">
              {run.id} · model: {run.experiment_name} · checkpoint: {run.checkpoint}
            </span>
          ) : undefined
        }
        actions={
          summary ? (
            <a
              className="btn-ghost"
              href={`/api/runs/${runId}/report`}
              target="_blank"
              rel="noopener"
            >
              Export report
            </a>
          ) : undefined
        }
      />

      {summary && (
        <div className="space-y-2 px-4 pt-3">
          <div className="flex flex-wrap items-baseline gap-2">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">
              Batch summary
            </span>
            <span className="text-[11px] text-slate-600">
              {summary.seeds.length} seed{summary.seeds.length === 1 ? "" : "s"} ·{" "}
              {summary.num_episodes} episode{summary.num_episodes === 1 ? "" : "s"} each ·
              95% CI (Student-t)
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {summary.metrics
              .filter((metric) => metric in summary.overall)
              .map((metric) => (
                <SummaryTile
                  key={metric}
                  label={METRIC_LABELS[metric] ?? metric}
                  stat={summary.overall[metric]}
                />
              ))}
          </div>
          {summary.seeds.length > 1 && (
            <div className="panel overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-ink-700 text-left text-[10px] uppercase tracking-wider text-slate-500">
                    <th className="px-3 py-2 font-medium">Seed</th>
                    <th className="px-3 py-2 font-medium">Episodes</th>
                    {summary.metrics.map((metric) => (
                      <th key={metric} className="px-3 py-2 font-medium">
                        {METRIC_LABELS[metric] ?? metric} (mean)
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {summary.per_seed.map((block, index) => (
                    <tr
                      key={`${block.seed}-${index}`}
                      className="border-b border-ink-800 font-mono text-slate-300 last:border-0"
                    >
                      <td className="px-3 py-1.5">{block.seed}</td>
                      <td className="px-3 py-1.5">{block.episodes}</td>
                      {summary.metrics.map((metric) => {
                        const mean = block.metrics[metric]?.mean;
                        return (
                          <td key={metric} className="px-3 py-1.5">
                            {mean === null || mean === undefined ? "—" : formatNumber(mean, 1)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {active && !hasViz ? (
        <div className="p-8">
          <div className="panel px-6 py-10 text-center">
            <div className="text-sm text-slate-300">Evaluation in progress…</div>
            <ProgressBar value={run?.progress} className="mx-auto mt-4 max-w-md" />
            <div className="mt-2 text-xs text-slate-500">
              The replay appears as soon as the first episode finishes.
            </div>
          </div>
        </div>
      ) : !hasViz ? (
        <div className="p-8">
          <div className="panel px-6 py-10 text-center text-sm text-slate-400">
            {run?.visualize === false
              ? "This evaluation was run without visualization recording."
              : "No visualization artifacts found for this run."}
          </div>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col gap-3 p-4">
          {/* Episode picker + reward strip */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-slate-500">Episode</span>
              {episodesWritten.map((ep) => (
                <button
                  key={ep}
                  onClick={() => setEpisode(ep)}
                  className={`rounded-md px-2.5 py-1 font-mono text-xs transition-colors ${
                    ep === episode
                      ? "bg-ink-600 text-slate-100"
                      : "bg-ink-800 text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {ep}
                </button>
              ))}
              {active && (
                <span className="ml-1 text-[11px] text-slate-600">more running…</span>
              )}
            </div>
            <div className="flex flex-1 flex-wrap justify-end gap-2">
              {rewardTotals && (
                <>
                  <RewardTile label="Total reward" value={rewardTotals.total ?? 0} />
                  {"blue_reward" in rewardTotals && (
                    <RewardTile label="Blue" value={rewardTotals.blue_reward} tone="up" />
                  )}
                  {"red_reward" in rewardTotals && (
                    <RewardTile label="Red" value={rewardTotals.red_reward} tone="down" />
                  )}
                </>
              )}
              {episodeData && (
                <>
                  <RewardTile label="Decoys" value={episodeData.totals.decoys_deployed} />
                  <RewardTile label="Decoy hits" value={episodeData.totals.decoy_attacks} />
                </>
              )}
            </div>
          </div>

          <div className="flex min-h-0 flex-1 gap-3">
            {/* Canvas */}
            <div className="panel flex min-w-0 flex-1 flex-col overflow-hidden">
              <div className="min-h-0 flex-1">
                {layout && renderState ? (
                  <NetworkCanvas
                    layout={layout}
                    state={renderState}
                    attackedIds={attackedIds}
                    selectedId={selected?.id ?? null}
                    onSelect={setSelected}
                  />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-slate-500">
                    Loading network…
                  </div>
                )}
              </div>
              <Legend />
              {player && (
                <StepScrubber step={step} stepCount={player.stepCount} onStep={setStep} />
              )}
            </div>

            {/* Right column: current actions + node details */}
            <div className="flex w-80 shrink-0 flex-col gap-3">
              <div className="panel">
                <div className="border-b border-ink-700 px-4 py-2 text-xs font-medium uppercase tracking-wider text-slate-500">
                  Step {step + 1} actions
                </div>
                {currentFrame ? (
                  <div className="space-y-2 p-4 text-xs">
                    <div className="rounded-md border border-rose-900/40 bg-rose-950/20 px-3 py-2">
                      <div className="mb-0.5 text-[10px] uppercase tracking-wider text-rose-400/80">
                        Red
                      </div>
                      <div className="font-mono text-slate-200">
                        {currentFrame.red.a ?? "—"}
                      </div>
                      <div className="mt-0.5 text-[11px] text-slate-500">
                        {currentFrame.red.ok ? "succeeded" : "failed"}
                        {currentFrame.da && (
                          <span className="ml-1 text-fuchsia-400">· hit decoy</span>
                        )}
                      </div>
                    </div>
                    <div className="rounded-md border border-sky-900/40 bg-sky-950/20 px-3 py-2">
                      <div className="mb-0.5 text-[10px] uppercase tracking-wider text-sky-400/80">
                        Blue
                      </div>
                      <div className="font-mono text-slate-200">
                        {currentFrame.blue.a ?? "—"}
                        {currentFrame.blue.tgt && (
                          <span className="text-slate-500"> → {currentFrame.blue.tgt}</span>
                        )}
                      </div>
                      <div className="mt-0.5 text-[11px] text-slate-500">
                        {currentFrame.blue.ok ? "succeeded" : "no-op"}
                      </div>
                    </div>
                    <div className="flex justify-between pt-1 font-mono text-slate-400">
                      <span className="text-slate-500">step reward</span>
                      <span>{formatNumber(currentFrame.red.r + currentFrame.blue.r, 1)}</span>
                    </div>
                  </div>
                ) : (
                  <div className="px-4 py-6 text-center text-xs text-slate-600">—</div>
                )}
              </div>
              <div className="panel min-h-0 flex-1 overflow-auto">
                <NodeDetails node={selected} onClose={() => setSelected(null)} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
