import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useEnvConfigDefaults, useLaunchSweep, useOptions } from "../api/hooks";
import PageHeader from "../components/PageHeader";
import { Field, FormSection, NumberField, Select, TextField } from "../components/forms/fields";

interface GridRow {
  param: string;
  values: string;
}

function parseValues(text: string): (string | number)[] {
  return text
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean)
    .map((v) => (v !== "" && !isNaN(Number(v)) ? Number(v) : v));
}

export default function NewSweepPage() {
  const navigate = useNavigate();
  const { data: options } = useOptions();
  const launch = useLaunchSweep();

  const [displayName, setDisplayName] = useState("");
  const [baseConfig, setBaseConfig] = useState("");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [rows, setRows] = useState<GridRow[]>([{ param: "learning_rate", values: "0.0003, 0.001" }]);

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
  const num = (key: string, fallback = 0): number => {
    const value = params[key];
    return typeof value === "number" ? value : fallback;
  };

  const grid = useMemo(() => {
    const out: Record<string, (string | number)[]> = {};
    for (const row of rows) {
      const key = row.param.trim();
      const values = parseValues(row.values);
      if (key && values.length) out[key] = values;
    }
    return out;
  }, [rows]);

  const cellCount = Object.values(grid).reduce((product, values) => product * values.length, 1);
  const gridKeys = Object.keys(grid);
  const canLaunch =
    displayName.trim().length > 0 &&
    baseConfig &&
    gridKeys.length > 0 &&
    cellCount >= 1 &&
    cellCount <= 16 &&
    !launch.isPending;

  const setRow = (index: number, patch: Partial<GridRow>) =>
    setRows((previous) => previous.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  const addRow = () => setRows((previous) => [...previous, { param: "", values: "" }]);
  const removeRow = (index: number) =>
    setRows((previous) => previous.filter((_, i) => i !== index));

  const onLaunch = async () => {
    const sweep = await launch.mutateAsync({
      display_name: displayName.trim(),
      base_config: baseConfig,
      params,
      grid,
    });
    navigate(`/sweeps/${sweep.id}`);
  };

  return (
    <div>
      <PageHeader
        title="New sweep"
        subtitle="Train one run per parameter combination and compare the results"
        actions={
          <button className="btn-primary" disabled={!canLaunch} onClick={onLaunch}>
            {launch.isPending ? "Launching…" : `Launch sweep (${cellCount} run${cellCount === 1 ? "" : "s"})`}
          </button>
        }
      />
      <div className="space-y-5 p-8">
        {launch.error && (
          <div className="rounded-md border border-red-900/60 bg-red-950/40 px-4 py-2.5 text-sm text-red-300">
            {(launch.error as Error).message}
          </div>
        )}

        <FormSection title="Base run" description="The training config every cell starts from">
          <Field label="Display name" span={2}>
            <TextField value={displayName} onChange={setDisplayName} placeholder="e.g. lr vs entropy" />
          </Field>
          <Field label="Base config">
            <Select value={baseConfig} onChange={setBaseConfig} options={trainConfigs} />
          </Field>
          <Field label="Total timesteps">
            <NumberField value={num("total_timesteps")} onChange={set("total_timesteps")} min={1} step={1} />
          </Field>
          <Field label="Steps per rollout">
            <NumberField value={num("num_steps")} onChange={set("num_steps")} min={1} step={1} />
          </Field>
          <Field label="Seed">
            <NumberField value={num("seed")} onChange={set("seed")} step={1} />
          </Field>
        </FormSection>

        <FormSection
          title="Parameter grid"
          description={`One run per combination · ${cellCount} cell${cellCount === 1 ? "" : "s"}${cellCount > 16 ? " (max 16)" : ""}`}
        >
          <div className="col-span-2 space-y-2">
            {rows.map((row, index) => (
              <div key={index} className="flex items-center gap-2">
                <div className="w-56">
                  <TextField
                    value={row.param}
                    onChange={(v) => setRow(index, { param: v })}
                    placeholder="param (e.g. learning_rate)"
                    mono
                  />
                </div>
                <div className="flex-1">
                  <TextField
                    value={row.values}
                    onChange={(v) => setRow(index, { values: v })}
                    placeholder="comma-separated values (e.g. 0.0003, 0.001)"
                    mono
                  />
                </div>
                <button
                  className="btn-ghost !px-2 !py-1 !text-xs text-slate-500"
                  onClick={() => removeRow(index)}
                  disabled={rows.length === 1}
                >
                  ✕
                </button>
              </div>
            ))}
            <button className="btn-ghost !px-2 !py-1 !text-xs" onClick={addRow}>
              + Add parameter
            </button>
          </div>
        </FormSection>
      </div>
    </div>
  );
}
