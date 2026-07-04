import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { VizLayout } from "../../api/types";
import type { RenderNode, RenderState } from "./frameState";
import {
  hitTest,
  initialTransform,
  renderScene,
  type Transform,
} from "./renderer";

export default function NetworkCanvas({
  layout,
  state,
  attackedIds,
  selectedId,
  onSelect,
}: {
  layout: VizLayout;
  state: RenderState;
  attackedIds: Set<number>;
  selectedId: number | null;
  onSelect: (node: RenderNode | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });
  const [transform, setTransform] = useState<Transform | null>(null);
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const dragRef = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);
  const layoutKey = useMemo(() => `${layout.bounds.w}x${layout.bounds.h}`, [layout]);

  // Track container size
  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0].contentRect;
      setSize({ width: rect.width, height: rect.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // Fit on first layout / size
  useEffect(() => {
    if (size.width > 0) setTransform(initialTransform(layout, size.width, size.height));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layoutKey, size.width, size.height]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !transform) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    renderScene(ctx, layout, state, transform, { hoveredId, selectedId, attackedIds }, size.width, size.height);
  }, [layout, state, transform, hoveredId, selectedId, attackedIds, size]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.width * dpr;
    canvas.height = size.height * dpr;
    draw();
  }, [size, draw]);

  useEffect(() => {
    draw();
  }, [draw]);

  const toWorld = (clientX: number, clientY: number): [number, number] => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const t = transform!;
    return [(clientX - rect.left - t.x) / t.k, (clientY - rect.top - t.y) / t.k];
  };

  const onWheel = (event: React.WheelEvent) => {
    if (!transform) return;
    event.preventDefault();
    const rect = canvasRef.current!.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const factor = Math.exp(-event.deltaY * 0.0015);
    const k = Math.max(0.05, Math.min(12, transform.k * factor));
    setTransform({
      k,
      x: px - (px - transform.x) * (k / transform.k),
      y: py - (py - transform.y) * (k / transform.k),
    });
  };

  const onPointerDown = (event: React.PointerEvent) => {
    if (!transform) return;
    (event.target as HTMLElement).setPointerCapture(event.pointerId);
    dragRef.current = { x: event.clientX, y: event.clientY, tx: transform.x, ty: transform.y };
  };

  const onPointerMove = (event: React.PointerEvent) => {
    if (!transform) return;
    if (dragRef.current) {
      setTransform({
        k: transform.k,
        x: dragRef.current.tx + (event.clientX - dragRef.current.x),
        y: dragRef.current.ty + (event.clientY - dragRef.current.y),
      });
      return;
    }
    const [wx, wy] = toWorld(event.clientX, event.clientY);
    const node = hitTest(state, wx, wy, transform);
    setHoveredId(node?.id ?? null);
  };

  const endDrag = (event: React.PointerEvent) => {
    const wasDragging = dragRef.current;
    dragRef.current = null;
    if (wasDragging) {
      const moved =
        Math.abs(event.clientX - wasDragging.x) + Math.abs(event.clientY - wasDragging.y);
      if (moved < 4 && transform) {
        const [wx, wy] = toWorld(event.clientX, event.clientY);
        onSelect(hitTest(state, wx, wy, transform));
      }
    }
  };

  const resetView = () =>
    setTransform(initialTransform(layout, size.width, size.height));

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden">
      <canvas
        ref={canvasRef}
        style={{ width: size.width, height: size.height }}
        className={`touch-none ${hoveredId !== null ? "cursor-pointer" : "cursor-grab active:cursor-grabbing"}`}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerLeave={() => {
          dragRef.current = null;
          setHoveredId(null);
        }}
      />
      <div className="absolute right-3 top-3 flex flex-col gap-1">
        <button
          onClick={() => transform && setTransform({ ...transform, k: Math.min(12, transform.k * 1.3) })}
          className="h-7 w-7 rounded-md border border-ink-600 bg-ink-850/90 text-slate-300 hover:border-ink-500"
        >
          +
        </button>
        <button
          onClick={() => transform && setTransform({ ...transform, k: Math.max(0.05, transform.k / 1.3) })}
          className="h-7 w-7 rounded-md border border-ink-600 bg-ink-850/90 text-slate-300 hover:border-ink-500"
        >
          −
        </button>
        <button
          onClick={resetView}
          className="h-7 w-7 rounded-md border border-ink-600 bg-ink-850/90 text-[10px] text-slate-300 hover:border-ink-500"
          title="Fit to view"
        >
          ⤢
        </button>
      </div>
    </div>
  );
}
