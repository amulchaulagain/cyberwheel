import type { VizLayout } from "../../api/types";
import type { RenderNode, RenderState } from "./frameState";
import {
  COMPROMISED_RING,
  DECOY_ATTACKED_COLOR,
  DECOY_COLOR,
  EDGE_COLOR,
  EDGE_ISOLATED_COLOR,
  ISOLATED_COLOR,
  RED_POSITION_RING,
  ROUTER_COLOR,
  stateColor,
  SUBNET_COLOR,
  SUBNET_STROKE,
} from "./palette";

export interface Transform {
  k: number; // scale
  x: number; // translate
  y: number;
}

export interface RenderOptions {
  hoveredId: number | null;
  selectedId: number | null;
  attackedIds: Set<number>;
}

const HOST_RADIUS = 6;
const ROUTER_RADIUS = 11;
// Below this on-screen host radius, hosts collapse into per-subnet aggregate
// glyphs — individual dots would be sub-pixel and illegible on big networks.
const LOD_HOST_SCREEN_RADIUS = 2.4;

export function renderScene(
  ctx: CanvasRenderingContext2D,
  _layout: VizLayout,
  state: RenderState,
  transform: Transform,
  options: RenderOptions,
  width: number,
  height: number,
): void {
  const dpr = window.devicePixelRatio || 1;
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.translate(transform.x, transform.y);
  ctx.scale(transform.k, transform.k);

  const nodes = state.nodes;
  const screenHostRadius = HOST_RADIUS * transform.k;
  const showHosts = screenHostRadius >= LOD_HOST_SCREEN_RADIUS;

  // Edges
  ctx.lineWidth = 1 / transform.k;
  ctx.strokeStyle = EDGE_COLOR;
  ctx.beginPath();
  for (const [a, b] of state.edges) {
    const na = nodes.get(a);
    const nb = nodes.get(b);
    if (!na || !nb) continue;
    if (!showHosts && (na.kind === "host" || nb.kind === "host")) continue;
    ctx.moveTo(na.x, na.y);
    ctx.lineTo(nb.x, nb.y);
  }
  ctx.stroke();

  // Isolated edges: dashed ghost so the topology stays legible
  if (state.isolatedEdges.length) {
    ctx.strokeStyle = EDGE_ISOLATED_COLOR;
    ctx.setLineDash([4 / transform.k, 4 / transform.k]);
    ctx.beginPath();
    for (const [a, b] of state.isolatedEdges) {
      const na = nodes.get(a);
      const nb = nodes.get(b);
      if (!na || !nb) continue;
      ctx.moveTo(na.x, na.y);
      ctx.lineTo(nb.x, nb.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Subnet discs
  for (const node of nodes.values()) {
    if (node.kind !== "subnet") continue;
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.r ?? 40, 0, Math.PI * 2);
    ctx.fillStyle = SUBNET_COLOR + "22";
    ctx.fill();
    ctx.lineWidth = 1 / transform.k;
    ctx.strokeStyle = SUBNET_STROKE + "66";
    ctx.stroke();
  }

  if (showHosts) {
    for (const node of nodes.values()) {
      if (node.kind === "host") drawHost(ctx, node, transform, options);
    }
  } else {
    drawSubnetAggregates(ctx, state, transform);
  }

  // Routers on top
  for (const node of nodes.values()) {
    if (node.kind !== "router") continue;
    ctx.beginPath();
    ctx.arc(node.x, node.y, ROUTER_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = ROUTER_COLOR;
    ctx.fill();
    ctx.lineWidth = 2 / transform.k;
    ctx.strokeStyle = "#1e1b3a";
    ctx.stroke();
  }

  // Red agent position ring
  const redNode = nodes.get(state.redPosition);
  if (redNode && showHosts) {
    ctx.beginPath();
    ctx.arc(redNode.x, redNode.y, HOST_RADIUS + 4, 0, Math.PI * 2);
    ctx.lineWidth = 2.5 / transform.k;
    ctx.strokeStyle = RED_POSITION_RING;
    ctx.stroke();
  }

  // Labels (only when zoomed in enough to be readable)
  if (transform.k > 1.4) {
    ctx.fillStyle = "#64748b";
    ctx.font = `${11 / transform.k}px ui-monospace, monospace`;
    ctx.textAlign = "center";
    for (const node of nodes.values()) {
      if (node.kind === "subnet") {
        ctx.fillStyle = "#94a3b8";
        ctx.fillText(node.name, node.x, node.y - (node.r ?? 40) - 6 / transform.k);
      }
    }
  }

  ctx.restore();
}

function drawHost(
  ctx: CanvasRenderingContext2D,
  node: RenderNode,
  transform: Transform,
  options: RenderOptions,
): void {
  const radius = HOST_RADIUS;
  let fill: string;
  if (node.decoy) {
    fill = options.attackedIds.has(node.id) ? DECOY_ATTACKED_COLOR : DECOY_COLOR;
  } else if (node.isolated) {
    fill = ISOLATED_COLOR;
  } else {
    fill = stateColor(node.state);
  }

  ctx.beginPath();
  if (node.decoy) {
    // Diamond marks decoys apart from real hosts at a glance.
    ctx.moveTo(node.x, node.y - radius);
    ctx.lineTo(node.x + radius, node.y);
    ctx.lineTo(node.x, node.y + radius);
    ctx.lineTo(node.x - radius, node.y);
    ctx.closePath();
  } else {
    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
  }
  ctx.fillStyle = fill;
  ctx.fill();

  const highlighted = node.id === options.hoveredId || node.id === options.selectedId;
  if (node.compromised) {
    ctx.lineWidth = 2 / transform.k;
    ctx.strokeStyle = COMPROMISED_RING;
    ctx.stroke();
  } else if (highlighted) {
    ctx.lineWidth = 2 / transform.k;
    ctx.strokeStyle = "#e2e8f0";
    ctx.stroke();
  } else {
    ctx.lineWidth = 1 / transform.k;
    ctx.strokeStyle = "#0b0e16";
    ctx.stroke();
  }

  if (node.id === options.selectedId) {
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius + 5, 0, Math.PI * 2);
    ctx.lineWidth = 1.5 / transform.k;
    ctx.strokeStyle = "#e2e8f0";
    ctx.stroke();
  }
}

/** Aggregate glyph per subnet at low zoom: sized by host count, colored by the
 *  most-severe host state, badged when it holds compromised/decoy hosts. */
function drawSubnetAggregates(
  ctx: CanvasRenderingContext2D,
  state: RenderState,
  transform: Transform,
): void {
  const bySubnet = new Map<
    string,
    { count: number; maxState: number; compromised: boolean; decoys: number }
  >();
  for (const node of state.nodes.values()) {
    if (node.kind !== "host" || !node.subnet) continue;
    const bucket =
      bySubnet.get(node.subnet) ??
      { count: 0, maxState: 0, compromised: false, decoys: 0 };
    bucket.count += 1;
    bucket.maxState = Math.max(bucket.maxState, node.state);
    bucket.compromised = bucket.compromised || node.compromised;
    if (node.decoy) bucket.decoys += 1;
    bySubnet.set(node.subnet, bucket);
  }
  for (const node of state.nodes.values()) {
    if (node.kind !== "subnet") continue;
    const bucket = bySubnet.get(node.name);
    if (!bucket) continue;
    const radius = Math.max(10, Math.min(node.r ?? 40, 8 + bucket.count * 0.6));
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = stateColor(bucket.maxState);
    ctx.globalAlpha = 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
    if (bucket.compromised) {
      ctx.lineWidth = 2.5 / transform.k;
      ctx.strokeStyle = COMPROMISED_RING;
      ctx.stroke();
    }
    if (bucket.decoys > 0) {
      ctx.beginPath();
      ctx.arc(node.x + radius * 0.7, node.y - radius * 0.7, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = DECOY_COLOR;
      ctx.fill();
    }
    ctx.fillStyle = "#e2e8f0";
    ctx.font = `${Math.max(9, radius * 0.7)}px ui-sans-serif, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(bucket.count), node.x, node.y);
    ctx.textBaseline = "alphabetic";
  }
}

export function hitTest(
  state: RenderState,
  worldX: number,
  worldY: number,
  transform: Transform,
): RenderNode | null {
  const hostReach = (HOST_RADIUS + 3) / transform.k;
  const routerReach = (ROUTER_RADIUS + 2) / transform.k;
  let best: RenderNode | null = null;
  let bestDist = Infinity;
  for (const node of state.nodes.values()) {
    if (node.kind === "subnet") continue;
    const reach = node.kind === "router" ? routerReach : hostReach;
    const dx = node.x - worldX;
    const dy = node.y - worldY;
    const dist = dx * dx + dy * dy;
    if (dist <= reach * reach && dist < bestDist) {
      best = node;
      bestDist = dist;
    }
  }
  return best;
}

export function initialTransform(
  layout: VizLayout,
  width: number,
  height: number,
): Transform {
  const pad = 40;
  const k = Math.min(
    (width - pad * 2) / layout.bounds.w,
    (height - pad * 2) / layout.bounds.h,
  );
  return {
    k,
    x: (width - layout.bounds.w * k) / 2,
    y: (height - layout.bounds.h * k) / 2,
  };
}
