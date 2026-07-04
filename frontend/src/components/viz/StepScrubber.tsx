import { useEffect, useRef, useState } from "react";

const SPEEDS = [0.5, 1, 2, 4];

export default function StepScrubber({
  step,
  stepCount,
  onStep,
}: {
  step: number;
  stepCount: number;
  onStep: (step: number) => void;
}) {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const timerRef = useRef<number>();
  const stepRef = useRef(step);
  stepRef.current = step;

  useEffect(() => {
    if (!playing) return;
    const interval = 700 / speed;
    timerRef.current = window.setInterval(() => {
      const next = stepRef.current + 1;
      if (next >= stepCount) {
        setPlaying(false);
        return;
      }
      onStep(next);
    }, interval);
    return () => window.clearInterval(timerRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speed, stepCount]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowRight") onStep(Math.min(stepCount - 1, step + 1));
      else if (event.key === "ArrowLeft") onStep(Math.max(0, step - 1));
      else if (event.key === " ") {
        event.preventDefault();
        setPlaying((previous) => !previous);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, stepCount, onStep]);

  return (
    <div className="flex items-center gap-3 border-t border-ink-700 bg-ink-900 px-4 py-2.5">
      <button
        onClick={() => {
          if (step >= stepCount - 1) onStep(0);
          setPlaying((previous) => !previous);
        }}
        className="btn-primary !px-3 !py-1.5"
      >
        {playing ? "❚❚ Pause" : "▶ Play"}
      </button>
      <button
        onClick={() => onStep(Math.max(0, step - 1))}
        className="btn-ghost !px-2 !py-1.5"
        title="Previous step (←)"
      >
        ◀
      </button>
      <input
        type="range"
        min={0}
        max={Math.max(0, stepCount - 1)}
        value={step}
        onChange={(event) => {
          setPlaying(false);
          onStep(Number(event.target.value));
        }}
        className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-ink-700 accent-accent"
      />
      <button
        onClick={() => onStep(Math.min(stepCount - 1, step + 1))}
        className="btn-ghost !px-2 !py-1.5"
        title="Next step (→)"
      >
        ▶
      </button>
      <div className="w-20 text-center font-mono text-sm text-slate-300">
        {step + 1} / {stepCount}
      </div>
      <div className="flex items-center gap-1">
        {SPEEDS.map((option) => (
          <button
            key={option}
            onClick={() => setSpeed(option)}
            className={`rounded px-1.5 py-0.5 text-[11px] transition-colors ${
              speed === option ? "bg-ink-600 text-slate-100" : "text-slate-500 hover:text-slate-300"
            }`}
          >
            {option}×
          </button>
        ))}
      </div>
    </div>
  );
}
