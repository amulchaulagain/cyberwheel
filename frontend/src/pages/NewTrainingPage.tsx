import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useEnvConfigDefaults, useLaunchTrain, useOptions } from "../api/hooks";
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

export default function NewTrainingPage() {
  const navigate = useNavigate();
  const { data: options } = useOptions();
  const launch = useLaunchTrain();

  const [displayName, setDisplayName] = useState("");
  const [baseConfig, setBaseConfig] = useState("");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [showAdvanced, setShowAdvanced] = useState(false);

  const trainConfigs = options?.env_configs.train ?? [];
  useEffect(() => {
    if (!baseConfig && trainConfigs.length) {
      setBaseConfig(
        trainConfigs.includes("train_rl_red_agent_vs_rl_blue.yaml")
          ? "train_rl_red_agent_vs_rl_blue.yaml"
          : trainConfigs[0],
      );
    }
  }, [trainConfigs, baseConfig]);

  const { data: defaults } = useEnvConfigDefaults(baseConfig || undefined);
  useEffect(() => {
    if (defaults?.params) setParams(defaults.params);
  }, [defaults]);

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
  const bool = (key: string): boolean => Boolean(params[key]);
  const agents = (params.agents ?? {}) as { red?: string; blue?: string };
  const networks: string[] = Array.isArray(params.network_config)
    ? (params.network_config as string[])
    : typeof params.network_config === "string"
      ? [params.network_config as string]
      : [];

  const toggleNetwork = (file: string) => {
    const next = networks.includes(file)
      ? networks.filter((n) => n !== file)
      : [...networks, file];
    set("network_config")(next);
  };

  const estimatedUpdates = useMemo(() => {
    const batch = num("num_envs", 1) * num("num_steps", 1);
    return batch > 0 ? Math.floor(num("total_timesteps") / batch) : 0;
  }, [params]);

  const canLaunch =
    displayName.trim().length > 0 &&
    baseConfig.length > 0 &&
    networks.length > 0 &&
    !launch.isPending;

  const onLaunch = async () => {
    const body = {
      display_name: displayName.trim(),
      base_config: baseConfig,
      params: { ...params, network_config: networks },
    };
    const run = await launch.mutateAsync(body);
    navigate(`/runs/train/${run.id}`);
  };

  return (
    <div>
      <PageHeader
        title="New training run"
        subtitle="All values prefilled from the selected base config"
        actions={
          <button className="btn-primary" disabled={!canLaunch} onClick={onLaunch}>
            {launch.isPending ? "Launching…" : "Launch training"}
          </button>
        }
      />
      <div className="space-y-5 p-8">
        {launch.error && (
          <div className="rounded-md border border-red-900/60 bg-red-950/40 px-4 py-2.5 text-sm text-red-300">
            {(launch.error as Error).message}
          </div>
        )}

        <FormSection title="Run" description="Identity and starting template">
          <Field label="Display name" span={2} hint="Names the run everywhere; the run id derives from it">
            <TextField
              value={displayName}
              onChange={setDisplayName}
              placeholder="e.g. blue vs art-agent, 3-network curriculum"
            />
          </Field>
          <Field label="Base config" span={2} hint="Defaults below come from this file">
            <Select value={baseConfig} onChange={setBaseConfig} options={trainConfigs} />
          </Field>
        </FormSection>

        <FormSection title="Environment" description="Networks, agents, detection and rewards">
          <Field label="Networks (multi-network curriculum)" span={4}>
            <div className="flex flex-wrap gap-1.5">
              {(options?.network_configs ?? []).map((network) => {
                const selected = networks.includes(network.file);
                return (
                  <button
                    key={network.file}
                    type="button"
                    onClick={() => toggleNetwork(network.file)}
                    className={`rounded-md border px-2.5 py-1.5 text-xs transition-colors ${
                      selected
                        ? "border-accent-dim bg-accent-dim/15 text-accent"
                        : "border-ink-600 text-slate-400 hover:border-ink-500 hover:text-slate-300"
                    }`}
                  >
                    <span className="font-mono">{network.file.replace(".yaml", "")}</span>
                    {network.hosts != null && (
                      <span className="ml-1.5 text-slate-500">
                        {formatNumber(network.hosts, 0)}h
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
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
          <Field label="Host definitions">
            <Select
              value={str("host_config")}
              onChange={set("host_config")}
              options={options?.host_configs ?? []}
            />
          </Field>
          <Field label="Valid targets">
            <Select
              value={str("valid_targets")}
              onChange={set("valid_targets")}
              options={options?.valid_targets ?? []}
            />
          </Field>
          <Field label="Blue reward">
            <Select
              value={str("blue_reward_function")}
              onChange={set("blue_reward_function")}
              options={options?.blue_reward_functions ?? []}
            />
          </Field>
          <Field label="Red reward">
            <Select
              value={str("red_reward_function")}
              onChange={set("red_reward_function")}
              options={options?.red_reward_functions ?? []}
            />
          </Field>
          <Field label="Max decoys">
            <NumberField value={num("max_decoys")} onChange={set("max_decoys")} min={0} step={1} />
          </Field>
          <Field label="Obs size compat">
            <Select
              value={str("network_size_compatibility")}
              onChange={set("network_size_compatibility")}
              options={options?.network_size_compatibility ?? []}
            />
          </Field>
        </FormSection>

        <FormSection title="Training" description="Scale, duration and reproducibility">
          <Field label="Total timesteps" hint={`≈ ${formatNumber(estimatedUpdates, 0)} PPO updates`}>
            <NumberField value={num("total_timesteps")} onChange={set("total_timesteps")} min={1} step={1} />
          </Field>
          <Field label="Parallel envs">
            <NumberField value={num("num_envs")} onChange={set("num_envs")} min={1} step={1} />
          </Field>
          <Field label="Steps per rollout">
            <NumberField value={num("num_steps")} onChange={set("num_steps")} min={1} step={1} />
          </Field>
          <Field label="Checkpoints (num saves)">
            <NumberField value={num("num_saves")} onChange={set("num_saves")} min={1} step={1} />
          </Field>
          <Field label="Eval episodes">
            <NumberField value={num("eval_episodes")} onChange={set("eval_episodes")} min={1} step={1} />
          </Field>
          <Field label="Device">
            <Select value={str("device")} onChange={set("device")} options={options?.devices ?? []} />
          </Field>
          <Field label="Seed">
            <NumberField value={num("seed")} onChange={set("seed")} step={1} />
          </Field>
          <Field label="Deterministic">
            <Toggle value={bool("deterministic")} onChange={set("deterministic")} />
          </Field>
          <Field label="Async envs">
            <Toggle value={bool("async_env")} onChange={set("async_env")} />
          </Field>
        </FormSection>

        <FormSection
          title="RL hyperparameters"
          description="PPO optimizer and loss settings"
          collapsed={!showAdvanced}
          onToggle={() => setShowAdvanced((previous) => !previous)}
        >
          <Field label="Actor LR">
            <NumberField value={num("actor_lr")} onChange={set("actor_lr")} />
          </Field>
          <Field label="Critic LR">
            <NumberField value={num("critic_lr")} onChange={set("critic_lr")} />
          </Field>
          <Field label="LR anneal">
            <Select
              value={String(params.anneal_lr ?? "false")}
              onChange={(value) => set("anneal_lr")(value === "false" ? false : value)}
              options={(options?.anneal_lr ?? []).map((option) => String(option))}
            />
          </Field>
          <Field label="Min LR">
            <NumberField value={num("min_lr")} onChange={set("min_lr")} />
          </Field>
          <Field label="Restart T-mult">
            <NumberField value={num("restart_Tmult")} onChange={set("restart_Tmult")} step={1} />
          </Field>
          <Field label="Gamma">
            <NumberField value={num("gamma")} onChange={set("gamma")} />
          </Field>
          <Field label="GAE lambda">
            <NumberField value={num("gae_lambda")} onChange={set("gae_lambda")} />
          </Field>
          <Field label="Minibatches">
            <NumberField value={num("num_minibatches")} onChange={set("num_minibatches")} min={1} step={1} />
          </Field>
          <Field label="Update epochs">
            <NumberField value={num("update_epochs")} onChange={set("update_epochs")} min={1} step={1} />
          </Field>
          <Field label="Clip coef">
            <NumberField value={num("clip_coef")} onChange={set("clip_coef")} />
          </Field>
          <Field label="Entropy coef">
            <NumberField value={num("ent_coef")} onChange={set("ent_coef")} />
          </Field>
          <Field label="Value fn coef">
            <NumberField value={num("vf_coef")} onChange={set("vf_coef")} />
          </Field>
          <Field label="Max grad norm">
            <NumberField value={num("max_grad_norm")} onChange={set("max_grad_norm")} />
          </Field>
          <Field label="Normalize advantage">
            <Toggle value={bool("norm_adv")} onChange={set("norm_adv")} />
          </Field>
          <Field label="Clip value loss">
            <Toggle value={bool("clip_vloss")} onChange={set("clip_vloss")} />
          </Field>
        </FormSection>

        <FormSection title="Tracking" description="Optional Weights & Biases logging (TensorBoard is always on)">
          <Field label="Track to W&B">
            <Toggle value={bool("track")} onChange={set("track")} />
          </Field>
          {bool("track") && (
            <>
              <Field label="W&B project">
                <TextField value={str("wandb_project_name")} onChange={set("wandb_project_name")} mono />
              </Field>
              <Field label="W&B entity">
                <TextField value={str("wandb_entity")} onChange={set("wandb_entity")} mono />
              </Field>
            </>
          )}
        </FormSection>
      </div>
    </div>
  );
}
