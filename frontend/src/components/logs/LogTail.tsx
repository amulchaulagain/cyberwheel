import { useEffect, useRef } from "react";

export default function LogTail({ content }: { content: string }) {
  const containerRef = useRef<HTMLPreElement>(null);
  const pinnedRef = useRef(true);

  useEffect(() => {
    const element = containerRef.current;
    if (element && pinnedRef.current) element.scrollTop = element.scrollHeight;
  }, [content]);

  return (
    <pre
      ref={containerRef}
      onScroll={(event) => {
        const element = event.currentTarget;
        pinnedRef.current =
          element.scrollHeight - element.scrollTop - element.clientHeight < 40;
      }}
      className="max-h-72 overflow-auto whitespace-pre-wrap break-all px-4 py-3 font-mono text-[11px] leading-relaxed text-slate-400"
    >
      {content || "no output yet"}
    </pre>
  );
}
