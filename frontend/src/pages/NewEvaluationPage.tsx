import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import {
  useCheckpoints,
  useEnvConfigDefaults,
  useLaunchEvaluate,
  useOptions,
  useRuns,
} from "../api/hooks";
import PageHeader from "../components/PageHeader";
import {
  Field,
  FormSection,
  NumberField,
  Select,
  TextField,
  Toggle,
} from "../components/forms/fields";
import { formatNumber } from "../lib/format";

/** Keys copied from a source training run's snapshot into the eval form. */
const PREFILL_KEYS = [
  "agents",
  "detector_config",
  "decoy_config",
  "host_config",
  "valid_targets",
  "max_decoys",
  "network_size_compatibility",
  "reward_function",
  "blue_reward_function",
  "red_reward_function",
] as const;

export default function NewEvaluationPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { data: options } = useOptions();
  const { data: runsData } = useRuns();
  const launch = useLaunchEvaluate();

  const [displayName, setDisplayName] = useState("");
  const [baseConfig, setBaseConfig] = useState("");
  const [sourceKey, setSourceKey] = useState(
    searchParams.get("source") ??
      (searchParams.get("experiment") ? `external:${searchParams.get("experiment")}` : ""),
  );
  const [checkpoint, setCheckpoint] = useState("agent");
  const [params, setParams] = useState<Record<string, unknown>>({});
  // Kept outside `params`: the defaults effect below overwrites params wholesale.
  const [seedsText, setSeedsText] = useState("");

  const evalConfigs = options?.env_configs.evaluate ?? [];
  useEffect(() => {
    if (!baseConfig && evalConfigs.length) {
      setBaseConfig(
        evalConfigs.includes("evaluate_rl_red_vs_rl_blue.yaml")
          ? "evaluate_rl_red_vs_rl_blue.yaml"
          : evalConfigs[0],
      );
    }
  }, [evalConfigs, baseConfig]);

  const trainRuns = useMemo(
    () =>
      (runsData?.runs ?? []).filter(
        (run) => run.kind === "train" && run.status !== "queued",
      ),
    [runsData],
  );
  const externalModels = runsData?.external_models ?? [];

  const sourceOptions = [
    ...trainRuns.map((run) => ({
      value: run.id,
      label: `${run.display_name} (${run.status})`,
    })),
    ...externalModels.map((model) => ({
      value: model.id,
      label: `${model.experiment_name} (external)`,
    })),
  ];

  const sourceRun = trainRuns.find((run) => run.id === sourceKey);
  const { data: defaults } = useEnvConfigDefaults(baseConfig || undefined);
  const { data: checkpointInfo } = useCheckpoints(sourceKey || undefined);

  useEffect(() => {
    if (!defaults?.params) return;
    // The replay view is the reason to evaluate from the UI, so default
    // recording on even though the base eval config ships visualize: false.
    const merged: Record<string, unknown> = { ...defaults.params, visualize: true };
    if (sourceRun) {
      for (const key of PREFILL_KEYS) {
        if (sourceRun.params[key] !== undefined) merged[key] = sourceRun.params[key];
      }
      const sourceNetworks = sourceRun.params.network_config;
      if (Array.isArray(sourceNetworks) && sourceNetworks.length)
        merged.network_config = sourceNetworks[0];
      else if (typeof sourceNetworks === "string") merged.network_config = sourceNetworks;
    }
    setParams(merged);
    setSeedsText(String(merged.seed ?? 1));
  }, [defaults, sourceRun?.id]);

  const parsedSeeds = useMemo(() => {
    const parts = seedsText.split(",").map((part) => part.trim()).filter(Boolean);
    if (!parts.length || parts.length > 20) return null;
    const seeds = parts.map(Number);
    if (seeds.some((seed) => !Number.isInteger(seed))) return null;
    if (new Set(seeds).size !== seeds.length) return null;
    return seeds;
  }, [seedsText]);

  useEffect(() => {
    const available = checkpointInfo?.checkpoints ?? [];
    if (available.length && !available.map(String).includes(checkpoint))
      setCheckpoint(String(available[0]));
  }, [checkpointInfo]);

  const set = (key: string) => (value: unknown) =>
    setParams((previous) => ({ ...previous, [key]: value }));
  const setAgent = (agent: "red" | "blue") => (value: string) =>
    setParams((previous) => ({
      ...previous,
      agents: { ...(previous.agents as Record<string, string>), [agent]: value },
    }));
  const num = (key: string, fallback = 0): number => {
    const value = params[key];
    return typeof value === "number" ? value : fallback;
  };
  const str = (key: string, fallback = ""): string => {
    const value = params[key];
    return typeof value === "string" ? value : fallback;
  };
  const bool = (key: string, fallback = false): boolean =>
    params[key] === undefined ? fallback : Boolean(params[key]);
  const agents = (params.agents ?? {}) as { red?: string; blue?: string };

  const canLaunch =
    displayName.trim().length > 0 &&
    baseConfig &&
    sourceKey &&
    parsedSeeds !== null &&
    !launch.isPending;

  const onLaunch = async () => {
    const source = sourceKey.startsWith("external:")
      ? { experiment_name: sourceKey.slice("external:".length) }
      : { run_id: sourceKey };
    const seeds = parsedSeeds ?? [1];
    const run = await launch.mutateAsync({
      display_name: displayName.trim(),
      base_config: baseConfig,
      source,
      checkpoint,
      params: {
        ...params,
        visualize: bool("visualize", true),
        // A single seed keeps the classic body; a list adds the batch param.
        ...(seeds.length > 1 ? { seeds, seed: seeds[0] } : { seed: seeds[0] }),
      },
    });
    navigate(`/runs/evaluate/${run.id}`);
  };

  const totalSteps =
    num("num_episodes") * num("num_steps") * (parsedSeeds?.length ?? 1);

  return (
    <div>
      <PageHeader
        title="New evaluation"
        subtitle="Replay a trained policy and record its behavior"
        actions={
          <button className="btn-primary" disabled={!canLaunch} onClick={onLaunch}>
            {launch.isPending ? "Launching…" : "Launch evaluation"}
          </button>
        }
      />
      <div className="space-y-5 p-8">
        {launch.error && (
          <div className="rounded-md border border-red-900/60 bg-red-950/40 px-4 py-2.5 text-sm text-red-300">
            {(launch.error as Error).message}
          </div>
        )}

        <FormSection title="Model" description="Which trained policy to evaluate">
          <Field label="Display name" span={2}>
            <TextField
              value={displayName}
              onChange={setDisplayName}
              placeholder="e.g. best checkpoint on 200-host network"
            />
          </Field>
          <Field
            label="Source training run"
            span={2}
            hint={sourceRun ? `experiment: ${sourceRun.experiment_name}` : undefined}
          >
            <Select
              value={sourceKey}
              onChange={setSourceKey}
              options={sourceOptions}
              allowEmpty
            />
          </Field>
          <Field
            label="Checkpoint"
            hint={
              checkpointInfo
                ? `agents: ${checkpointInfo.agents.join(", ")}`
                : "select a source run first"
            }
          >
            <Select
              value={checkpoint}
              onChange={setCheckpoint}
              options={(checkpointInfo?.checkpoints ?? []).map((tag) => ({
                value: String(tag),
                label: tag === "agent" ? "latest (agent.pt)" : `step ${formatNumber(Number(tag), 0)}`,
              }))}
            />
          </Field>
          <Field label="Base config">
            <Select value={baseConfig} onChange={setBaseConfig} options={evalConfigs} />
          </Field>
        </FormSection>

        <FormSection title="Scenario" description="Prefilled from the source run's training snapshot">
          <Field label="Network">
            <Select
              value={str("network_config")}
              onChange={set("network_config")}
              options={(options?.network_configs ?? []).map((network) => network.file)}
            />
          </Field>
          <Field label="Red agent">
            <Select
              value={agents.red ?? ""}
              onChange={setAgent("red")}
              options={options?.red_agents ?? []}
            />
          </Field>
          <Field label="Blue agent">
            <Select
              value={agents.blue ?? ""}
              onChange={setAgent("blue")}
              options={options?.blue_agents ?? []}
            />
          </Field>
          <Field label="Detector">
            <Select
              value={str("detector_config")}
              onChange={set("detector_config")}
              options={options?.detector_configs ?? []}
            />
          </Field>
          <Field label="Decoys">
            <Select
              value={str("decoy_config")}
              onChange={set("decoy_config")}
              options={options?.decoy_configs ?? []}
            />
          </Field>
          <Field label="Valid targets">
            <Select
              value={str("valid_targets")}
              onChange={set("valid_targets")}
              options={options?.valid_targets ?? []}
            />
          </Field>
          <Field label="Max decoys">
            <NumberField value={num("max_decoys")} onChange={set("max_decoys")} min={0} step={1} />
          </Field>
        </FormSection>

        <FormSection title="Rollout" description="Length, reproducibility and recording">
          <Field label="Episodes">
            <NumberField value={num("num_episodes")} onChange={set("num_episodes")} min={1} step={1} />
          </Field>
          <Field label="Steps per episode" hint={`${formatNumber(totalSteps, 0)} total steps`}>
            <NumberField value={num("num_steps")} onChange={set("num_steps")} min={1} step={1} />
          </Field>
          <Field
            label="Seeds"
            hint={
              parsedSeeds
                ? parsedSeeds.length > 1
                  ? `${parsedSeeds.length}-seed batch — each seed reseeds its block of episodes`
                  : "single seed"
                : "comma-separated integers, e.g. 1,2,3 (max 20, unique)"
            }
          >
            <TextField value={seedsText} onChange={setSeedsText} placeholder="e.g. 1,2,3" mono />
          </Field>
          <Field
            label="Deterministic"
            hint="governs single-seed runs; multi-seed batches always reseed per seed"
          >
            <Toggle value={bool("deterministic")} onChange={set("deterministic")} />
          </Field>
          <Field
            label="Record visualization"
            hint="Per-step network state for the replay view"
          >
            <Toggle value={bool("visualize", true)} onChange={set("visualize")} />
          </Field>
        </FormSection>
      </div>
    </div>
  );
}
