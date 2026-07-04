import type { RenderNode } from "./frameState";
import { STATE_LABELS } from "./palette";

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-3 py-1">
      <span className="text-slate-500">{label}</span>
      <span className="text-right font-mono text-slate-300">{value}</span>
    </div>
  );
}

export default function NodeDetails({
  node,
  onClose,
}: {
  node: RenderNode | null;
  onClose: () => void;
}) {
  if (!node) {
    return (
      <div className="flex h-full items-center justify-center px-4 text-center text-xs text-slate-600">
        Click a node to inspect its state, or use the scrubber to replay the episode.
      </div>
    );
  }
  return (
    <div className="p-4 text-xs">
      <div className="mb-3 flex items-start justify-between">
        <div>
          <div className="font-mono text-sm text-slate-100">{node.name}</div>
          <div className="text-[11px] uppercase tracking-wider text-slate-500">
            {node.decoy ? "decoy host" : node.kind}
          </div>
        </div>
        <button onClick={onClose} className="text-slate-600 hover:text-slate-300">
          ✕
        </button>
      </div>
      <div className="divide-y divide-ink-700/60">
        {node.kind === "host" && (
          <>
            <Row label="Subnet" value={node.subnet ?? "—"} />
            <Row label="Type" value={node.type ?? "—"} />
            {node.ip && <Row label="IP" value={node.ip} />}
            <Row label="Red knowledge" value={STATE_LABELS[node.state] ?? node.state} />
            <Row
              label="Compromised"
              value={
                node.compromised ? <span className="text-rose-400">yes</span> : "no"
              }
            />
            <Row
              label="Isolated"
              value={node.isolated ? <span className="text-slate-300">yes</span> : "no"}
            />
            {node.restored && <Row label="Restored" value="yes" />}
          </>
        )}
        {node.kind === "router" && <Row label="Role" value="Router" />}
      </div>
    </div>
  );
}
