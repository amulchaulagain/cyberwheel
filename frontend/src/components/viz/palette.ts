// Node-state palette. Killchain severity is an ordered ramp (safe -> impacted)
// so progression reads as intensifying threat; decoy and isolated get distinct
// hues since they are categorically different, not "more severe". Tuned for the
// dark canvas; the red-agent position is drawn as a separate ring, not a fill.

export const STATE_NAMES = [
  "safe",
  "sweeped",
  "scanned",
  "discovered",
  "escalated",
  "impacted",
] as const;

// Slate -> amber -> red ramp.
export const STATE_COLORS = [
  "#64748b", // safe
  "#3f7c9e", // sweeped   (subnet-level recon)
  "#4a97b8", // scanned   (host recon)
  "#eab308", // discovered
  "#f97316", // escalated
  "#ef4444", // impacted
];

export const STATE_LABELS = [
  "Safe",
  "Ping-swept",
  "Port-scanned",
  "Discovered",
  "Priv-escalated",
  "Impacted",
];

export const DECOY_COLOR = "#38bdf8";
export const DECOY_ATTACKED_COLOR = "#a855f7";
export const ISOLATED_COLOR = "#94a3b8";
export const COMPROMISED_RING = "#f43f5e";
export const RED_POSITION_RING = "#ff4d4d";
export const ROUTER_COLOR = "#c084fc";
export const SUBNET_COLOR = "#334155";
export const SUBNET_STROKE = "#475569";
export const EDGE_COLOR = "#233046";
export const EDGE_ISOLATED_COLOR = "#7f1d1d";

export function stateColor(state: number): string {
  return STATE_COLORS[Math.max(0, Math.min(STATE_COLORS.length - 1, state))];
}
