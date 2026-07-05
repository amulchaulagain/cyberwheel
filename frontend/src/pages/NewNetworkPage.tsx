import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useGenerateNetwork, usePreviewNetwork } from "../api/hooks";
import type { GenerateNetworkParams, VizLayout } from "../api/types";
import PageHeader from "../components/PageHeader";
import { Field, FormSection, NumberField, Select, TextField, Toggle } from "../components/forms/fields";
import NetworkCanvas from "../components/viz/NetworkCanvas";
import { neutralState } from "../components/viz/frameState";
import { formatNumber } from "../lib/format";

const TIER_CAPS: Record<string, number> = { small: 100, medium: 1000, large: 10000 };

export default function NewNetworkPage() {
  const navigate = useNavigate();
  const preview = usePreviewNetwork();
  const generate = useGenerateNetwork();

  const [name, setName] = useState("");
  const [params, setParams] = useState<GenerateNetworkParams>({
    num_hosts: 30,
    num_subnets: 4,
    server_ratio: 0.2,
    vuln_density: 1.0,
    dedicated_server_subnets: true,
    seed: 0,
    size_tier: "small",
  });
  const [layout, setLayout] = useState<VizLayout | null>(null);

  const set = <K extends keyof GenerateNetworkParams>(key: K) => (value: GenerateNetworkParams[K]) =>
    setParams((previous) => ({ ...previous, [key]: value }));

  const numServers = Math.round(params.num_hosts * params.server_ratio);
  const cap = TIER_CAPS[params.size_tier];
  const nameValid = /^[a-zA-Z0-9_-]+$/.test(name.trim());
  const withinTier = params.num_hosts <= cap && params.num_hosts >= 1 && params.num_subnets >= 1;
  const canGenerate = nameValid && withinTier && !generate.isPending;

  const state = useMemo(() => (layout ? neutralState(layout) : null), [layout]);

  const onPreview = async () => {
    const { layout } = await preview.mutateAsync(params);
    setLayout(layout);
  };
  const onGenerate = async () => {
    await generate.mutateAsync({ name: name.trim(), params });
    navigate(`/train/new`);
  };

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="New network"
        subtitle="Procedurally generate a network from security-posture knobs"
        actions={
          <>
            <button className="btn-ghost" disabled={!withinTier || preview.isPending} onClick={onPreview}>
              {preview.isPending ? "Rendering…" : "Preview"}
            </button>
            <button className="btn-primary" disabled={!canGenerate} onClick={onGenerate}>
              {generate.isPending ? "Creating…" : "Create network"}
            </button>
          </>
        }
      />
      <div className="flex min-h-0 flex-1 gap-4 p-6">
        <div className="w-96 shrink-0 space-y-5 overflow-y-auto">
          {(preview.error || generate.error) && (
            <div className="rounded-md border border-red-900/60 bg-red-950/40 px-4 py-2.5 text-sm text-red-300">
              {((preview.error || generate.error) as Error).message}
            </div>
          )}

          <FormSection title="Identity" description="Name and reproducibility">
            <Field label="Network name" span={2} hint={name && !nameValid ? "letters, digits, - or _ only" : undefined}>
              <TextField value={name} onChange={setName} placeholder="e.g. segmented_dmz_60" mono />
            </Field>
            <Field label="Seed" hint="deterministic generation">
              <NumberField value={params.seed} onChange={(v) => set("seed")(v)} step={1} />
            </Field>
            <Field label="Size tier" hint={`cap: ${cap} hosts`}>
              <Select
                value={params.size_tier}
                onChange={(v) => set("size_tier")(v as GenerateNetworkParams["size_tier"])}
                options={["small", "medium", "large"]}
              />
            </Field>
          </FormSection>

          <FormSection title="Topology" description="Size and segmentation">
            <Field label="Hosts" hint={params.num_hosts > cap ? `exceeds ${params.size_tier} cap` : undefined}>
              <NumberField value={params.num_hosts} onChange={(v) => set("num_hosts")(v)} min={1} step={1} />
            </Field>
            <Field label="Subnets">
              <NumberField value={params.num_subnets} onChange={(v) => set("num_subnets")(v)} min={1} step={1} />
            </Field>
            <Field label="Dedicated server subnets" span={2} hint="place crown jewels in their own segments">
              <Toggle
                value={params.dedicated_server_subnets}
                onChange={(v) => set("dedicated_server_subnets")(v)}
              />
            </Field>
          </FormSection>

          <FormSection title="Security posture" description="Crown jewels and vulnerability">
            <Field label="Server ratio" span={2} hint={`${numServers} crown-jewel server(s)`}>
              <NumberField value={params.server_ratio} onChange={(v) => set("server_ratio")(v)} min={0} step="any" />
            </Field>
            <Field label="Vuln density" span={2} hint="fraction of user hosts that are exploitable (rest hardened)">
              <NumberField value={params.vuln_density} onChange={(v) => set("vuln_density")(v)} min={0} step="any" />
            </Field>
          </FormSection>

          <div className="panel px-4 py-3 text-xs text-slate-400">
            <div className="mb-1 font-medium uppercase tracking-wider text-slate-500">Summary</div>
            {formatNumber(params.num_hosts, 0)} hosts across {formatNumber(params.num_subnets, 0)} subnets ·{" "}
            {numServers} servers · {Math.round(params.vuln_density * 100)}% of user hosts vulnerable
          </div>
        </div>

        <div className="panel flex min-w-0 flex-1 flex-col overflow-hidden">
          {layout && state ? (
            <NetworkCanvas
              layout={layout}
              state={state}
              attackedIds={new Set()}
              selectedId={null}
              onSelect={() => {}}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-slate-500">
              Click “Preview” to render the generated topology
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
