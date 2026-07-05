import type { VizEpisode, VizLayout } from "../../api/types";
import { decoySlotPosition } from "./decoySlots";

export interface RenderNode {
  id: number;
  name: string;
  kind: "router" | "subnet" | "host";
  x: number;
  y: number;
  r?: number;
  subnet?: string;
  type?: string;
  ip?: string | null;
  decoy: boolean;
  // dynamic state
  state: number; // killchain severity code
  compromised: boolean;
  isolated: boolean;
  restored: boolean;
}

export interface RenderState {
  nodes: Map<number, RenderNode>;
  edges: [number, number][];
  isolatedEdges: [number, number][];
  redPosition: number;
  step: number;
}

/** A static, all-safe render state for previewing a topology with no episode. */
export function neutralState(layout: VizLayout): RenderState {
  const nodes = new Map<number, RenderNode>();
  layout.nodes.forEach((node, id) => {
    nodes.set(id, {
      id,
      name: node.name,
      kind: node.kind,
      x: node.x,
      y: node.y,
      r: node.r,
      subnet: node.subnet,
      type: node.type,
      ip: node.ip,
      decoy: false,
      state: 0,
      compromised: false,
      isolated: false,
      restored: false,
    });
  });
  return {
    nodes,
    edges: layout.edges.map((edge) => [...edge] as [number, number]),
    isolatedEdges: [],
    redPosition: -1,
    step: -1,
  };
}

const KEYFRAME_INTERVAL = 25;

/**
 * Precomputes cumulative render state for every step of an episode by
 * prefix-summing the delta frames, storing a keyframe snapshot every
 * KEYFRAME_INTERVAL steps so scrubbing to any step is O(interval), not O(step).
 */
export class EpisodePlayer {
  readonly stepCount: number;
  private layout: VizLayout;
  private episode: VizEpisode;
  private keyframes: Map<number, RenderState> = new Map();

  constructor(layout: VizLayout, episode: VizEpisode) {
    this.layout = layout;
    this.episode = episode;
    this.stepCount = episode.steps.length;
    this.buildKeyframes();
  }

  private baseState(): RenderState {
    const nodes = new Map<number, RenderNode>();
    this.layout.nodes.forEach((node, id) => {
      nodes.set(id, {
        id,
        name: node.name,
        kind: node.kind,
        x: node.x,
        y: node.y,
        r: node.r,
        subnet: node.subnet,
        type: node.type,
        ip: node.ip,
        decoy: false,
        state: 0,
        compromised: false,
        isolated: false,
        restored: false,
      });
    });
    return {
      nodes,
      edges: this.layout.edges.map((edge) => [...edge] as [number, number]),
      isolatedEdges: [],
      redPosition: this.episode.initial.red_position,
      step: -1,
    };
  }

  private clone(state: RenderState): RenderState {
    const nodes = new Map<number, RenderNode>();
    state.nodes.forEach((node, id) => nodes.set(id, { ...node }));
    return {
      nodes,
      edges: state.edges.map((edge) => [...edge] as [number, number]),
      isolatedEdges: state.isolatedEdges.map((edge) => [...edge] as [number, number]),
      redPosition: state.redPosition,
      step: state.step,
    };
  }

  private applyFrame(state: RenderState, frameIndex: number): void {
    const frame = this.episode.steps[frameIndex];
    state.step = frame.s;
    for (const added of frame.add ?? []) {
      const slots = this.layout.decoy_slots[added.subnet];
      const [x, y] = slots
        ? decoySlotPosition(slots, added.slot)
        : [0, 0];
      state.nodes.set(added.id, {
        id: added.id,
        name: added.name,
        kind: "host",
        x,
        y,
        subnet: added.subnet,
        type: added.type,
        ip: null,
        decoy: true,
        state: 0,
        compromised: false,
        isolated: false,
        restored: false,
      });
      const subnetId = this.subnetId(added.subnet);
      if (subnetId !== undefined) state.edges.push([added.id, subnetId]);
    }
    for (const removedId of frame.rm ?? []) {
      state.nodes.delete(removedId);
      state.edges = state.edges.filter(([a, b]) => a !== removedId && b !== removedId);
    }
    for (const [tag, code] of Object.entries(frame.kh ?? {})) {
      const node = state.nodes.get(Number(tag));
      if (node) node.state = code;
    }
    for (const [tag, flags] of Object.entries(frame.flags ?? {})) {
      const node = state.nodes.get(Number(tag));
      if (node) {
        node.compromised = flags.c;
        node.isolated = flags.i;
        node.restored = flags.r;
      }
    }
    for (const edge of frame.eoff ?? []) {
      state.edges = state.edges.filter(
        ([a, b]) => !(a === edge[0] && b === edge[1]) && !(a === edge[1] && b === edge[0]),
      );
      state.isolatedEdges.push([...edge] as [number, number]);
    }
    for (const edge of frame.eon ?? []) {
      state.isolatedEdges = state.isolatedEdges.filter(
        ([a, b]) => !(a === edge[0] && b === edge[1]) && !(a === edge[1] && b === edge[0]),
      );
      state.edges.push([...edge] as [number, number]);
    }
    if (frame.pos !== undefined) state.redPosition = frame.pos;
  }

  private subnetId(subnetName: string): number | undefined {
    const index = this.layout.nodes.findIndex(
      (node) => node.kind === "subnet" && node.name === subnetName,
    );
    return index >= 0 ? index : undefined;
  }

  private buildKeyframes(): void {
    let state = this.baseState();
    this.keyframes.set(-1, this.clone(state));
    for (let i = 0; i < this.stepCount; i++) {
      this.applyFrame(state, i);
      if (i % KEYFRAME_INTERVAL === KEYFRAME_INTERVAL - 1) {
        this.keyframes.set(i, this.clone(state));
      }
    }
  }

  /** Cumulative render state after `step` frames (step -1 = pre-episode baseline). */
  stateAt(step: number): RenderState {
    const target = Math.max(-1, Math.min(step, this.stepCount - 1));
    let anchor = -1;
    for (const key of this.keyframes.keys()) {
      if (key <= target && key > anchor) anchor = key;
    }
    const state = this.clone(this.keyframes.get(anchor)!);
    for (let i = anchor + 1; i <= target; i++) this.applyFrame(state, i);
    return state;
  }
}
