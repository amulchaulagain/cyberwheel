export interface Options {
  env_configs: { train: string[]; evaluate: string[]; run: string[] };
  network_configs: { file: string; hosts: number | null; subnets: number | null }[];
  red_agents: string[];
  blue_agents: string[];
  detector_configs: string[];
  decoy_configs: string[];
  host_configs: string[];
  blue_reward_functions: string[];
  red_reward_functions: string[];
  reward_functions: string[];
  environments: string[];
  valid_targets: string[];
  devices: string[];
  anneal_lr: (string | boolean)[];
  network_size_compatibility: string[];
}

export type RunStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "stopped"
  | "orphaned";

export interface RunRecord {
  id: string;
  kind: "train" | "evaluate";
  display_name: string;
  status: RunStatus;
  pid: number | null;
  exit_code: number | null;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  base_config: string;
  generated_config: string;
  params: Record<string, unknown>;
  experiment_name: string;
  // evaluate-only
  source_run_id?: string | null;
  graph_name?: string;
  checkpoint?: string;
  visualize?: boolean;
  // decorations
  progress?: number | null;
  last_global_step?: number;
  last_sps?: number | null;
  artifacts?: Record<string, boolean>;
}

export interface ExternalModel {
  id: string;
  kind: "external_model";
  experiment_name: string;
  agents: string[];
  checkpoints: (string | number)[];
}

export interface RunsResponse {
  runs: RunRecord[];
  external_models: ExternalModel[];
}

export interface MetricsSummary {
  tags: Record<string, string[]>;
  last_step: number;
  total_timesteps?: number;
}

export type ScalarPoint = [step: number, wallTime: number, value: number];

export interface ActionsResponse {
  episodes: number[];
  reward_totals: Record<string, Record<string, number>>;
  rows: Record<string, string>[];
}

export interface GenerateNetworkParams {
  num_hosts: number;
  num_subnets: number;
  server_ratio: number;
  vuln_density: number;
  dedicated_server_subnets: boolean;
  seed: number;
  size_tier: "small" | "medium" | "large";
}

export interface StatBlock {
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  ci95_lo: number | null;
  ci95_hi: number | null;
  n: number;
}

export interface EvalSeedSummary {
  seed: number;
  episodes: number;
  metrics: Record<string, StatBlock>;
}

export interface EvalSummary {
  format_version: number;
  graph_name?: string;
  experiment_name?: string;
  seeds: number[];
  explicit_seeds: boolean;
  deterministic?: boolean;
  num_episodes: number;
  num_steps: number;
  total_episodes: number;
  metrics: string[];
  per_episode: Record<string, number>[];
  per_seed: EvalSeedSummary[];
  overall: Record<string, StatBlock>;
}

export interface VizMeta {
  format_version: number;
  created_at: string;
  counts: { hosts: number; subnets: number; nodes: number };
  episodes_written: number[];
  experiment_name?: string;
  graph_name?: string;
  network_config?: string | string[];
  agents?: { red?: string; blue?: string } | null;
  num_episodes?: number;
  num_steps?: number;
  seed?: number | null;
  seeds?: number[];
}

export interface LayoutNode {
  name: string;
  kind: "router" | "subnet" | "host";
  x: number;
  y: number;
  r?: number;
  subnet?: string;
  type?: string;
  ip?: string | null;
}

export interface DecoySlots {
  cx: number;
  cy: number;
  rot: number;
  spacing: number;
  base: number;
}

export interface VizLayout {
  bounds: { w: number; h: number };
  nodes: LayoutNode[];
  edges: [number, number][];
  decoy_slots: Record<string, DecoySlots>;
}

export interface AgentActionRecord {
  a: string | null;
  src?: number;
  dst?: number;
  tgt?: string | null;
  ok: boolean;
  r: number;
}

export interface VizFrame {
  s: number;
  red: AgentActionRecord;
  blue: AgentActionRecord;
  pos?: number;
  kh?: Record<string, number>;
  flags?: Record<string, { c: boolean; i: boolean; r: boolean }>;
  add?: { id: number; name: string; subnet: string; type: string; slot: number }[];
  rm?: number[];
  eoff?: [number, number][];
  eon?: [number, number][];
  da?: boolean;
}

export interface VizEpisode {
  episode: number;
  initial: { red_position: number };
  steps: VizFrame[];
  totals: {
    reward: number;
    blue: number;
    red: number;
    decoys_deployed: number;
    decoy_attacks: number;
  };
}
