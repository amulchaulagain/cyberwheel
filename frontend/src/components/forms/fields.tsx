import type { ReactNode } from "react";

export function FormSection({
  title,
  description,
  children,
  collapsed = false,
  onToggle,
}: {
  title: string;
  description?: string;
  children: ReactNode;
  collapsed?: boolean;
  onToggle?: () => void;
}) {
  return (
    <section className="panel">
      <button
        type="button"
        onClick={onToggle}
        disabled={!onToggle}
        className={`flex w-full items-center justify-between px-5 py-3 text-left ${
          onToggle ? "cursor-pointer hover:bg-ink-850" : "cursor-default"
        }`}
      >
        <div>
          <div className="text-sm font-semibold text-slate-200">{title}</div>
          {description && <div className="text-xs text-slate-500">{description}</div>}
        </div>
        {onToggle && (
          <span className="text-xs text-slate-500">{collapsed ? "show ▾" : "hide ▴"}</span>
        )}
      </button>
      {!collapsed && (
        <div className="grid grid-cols-2 gap-x-5 gap-y-4 border-t border-ink-700 px-5 py-4 md:grid-cols-3 xl:grid-cols-4">
          {children}
        </div>
      )}
    </section>
  );
}

export function Field({
  label,
  hint,
  span = 1,
  children,
}: {
  label: string;
  hint?: string;
  span?: 1 | 2 | 3 | 4;
  children: ReactNode;
}) {
  const spanClass = { 1: "", 2: "col-span-2", 3: "col-span-2 md:col-span-3", 4: "col-span-2 md:col-span-3 xl:col-span-4" }[span];
  return (
    <label className={`block ${spanClass}`}>
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] text-slate-600">{hint}</span>}
    </label>
  );
}

export function Select({
  value,
  onChange,
  options,
  allowEmpty,
}: {
  value: string;
  onChange: (value: string) => void;
  options: (string | { value: string; label: string })[];
  allowEmpty?: boolean;
}) {
  return (
    <select
      className="field-input appearance-none bg-[right_0.5rem_center] pr-8"
      style={{
        backgroundImage:
          "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%2364748b' fill='none' stroke-width='1.5'/%3E%3C/svg%3E\")",
        backgroundRepeat: "no-repeat",
      }}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    >
      {allowEmpty && <option value="">—</option>}
      {options.map((option) => {
        const { value: v, label } =
          typeof option === "string" ? { value: option, label: option } : option;
        return (
          <option key={v} value={v}>
            {label}
          </option>
        );
      })}
    </select>
  );
}

export function NumberField({
  value,
  onChange,
  step,
  min,
}: {
  value: number | string;
  onChange: (value: number) => void;
  step?: number | "any";
  min?: number;
}) {
  return (
    <input
      type="number"
      className="field-input font-mono"
      value={value}
      step={step ?? "any"}
      min={min}
      onChange={(event) => onChange(Number(event.target.value))}
    />
  );
}

export function TextField({
  value,
  onChange,
  placeholder,
  mono,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  mono?: boolean;
}) {
  return (
    <input
      type="text"
      className={`field-input ${mono ? "font-mono" : ""}`}
      value={value}
      placeholder={placeholder}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

export function Toggle({
  value,
  onChange,
  label,
}: {
  value: boolean;
  onChange: (value: boolean) => void;
  label?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={value}
      onClick={() => onChange(!value)}
      className="flex items-center gap-2.5 py-1"
    >
      <span
        className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors ${
          value ? "bg-accent-dim" : "bg-ink-600"
        }`}
      >
        {/* 36px track, 16px knob, symmetric 2px inset: off=2px, on=18px */}
        <span
          className="inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform duration-150"
          style={{ transform: `translateX(${value ? 18 : 2}px)` }}
        />
      </span>
      {/* fixed width so "on"/"off" never shift surrounding layout */}
      <span className="w-6 text-left text-sm text-slate-300">
        {label ?? (value ? "on" : "off")}
      </span>
    </button>
  );
}
