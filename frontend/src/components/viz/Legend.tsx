import {
  DECOY_ATTACKED_COLOR,
  DECOY_COLOR,
  RED_POSITION_RING,
  STATE_COLORS,
  STATE_LABELS,
} from "./palette";

function Swatch({ color, label, shape = "circle" }: { color: string; label: string; shape?: "circle" | "diamond" | "ring" }) {
  return (
    <div className="flex items-center gap-1.5">
      {shape === "diamond" ? (
        <span className="h-2.5 w-2.5 rotate-45" style={{ background: color }} />
      ) : shape === "ring" ? (
        <span
          className="h-2.5 w-2.5 rounded-full border-2"
          style={{ borderColor: color, background: "transparent" }}
        />
      ) : (
        <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
      )}
      <span className="text-[11px] text-slate-400">{label}</span>
    </div>
  );
}

export default function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-4 py-2">
      {STATE_LABELS.map((label, index) => (
        <Swatch key={label} color={STATE_COLORS[index]} label={label} />
      ))}
      <span className="text-ink-600">|</span>
      <Swatch color={DECOY_COLOR} label="Decoy" shape="diamond" />
      <Swatch color={DECOY_ATTACKED_COLOR} label="Decoy hit" shape="diamond" />
      <Swatch color={RED_POSITION_RING} label="Red position" shape="ring" />
      <Swatch color="#f43f5e" label="Compromised" shape="ring" />
    </div>
  );
}
