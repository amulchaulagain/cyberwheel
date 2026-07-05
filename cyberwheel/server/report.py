"""Self-contained HTML report for an evaluation run.

Renders the run's metadata, the statistical summary (Feature 1), a per-seed /
per-episode breakdown, and action-log highlights into a single inline-styled
HTML document (no external assets), so it can be saved or shared as one file.
Pure json/csv + string formatting — no torch.
"""

from __future__ import annotations

import csv
import html

from cyberwheel.server import actions_log

_STYLE = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem; background: #0b0e14; color: #cbd5e1;
       font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; line-height: 1.5; }
h1 { font-size: 1.5rem; margin: 0 0 0.25rem; color: #f1f5f9; }
h2 { font-size: 1rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b;
     margin: 2rem 0 0.75rem; }
.sub { color: #64748b; font-family: ui-monospace, monospace; font-size: 0.85rem; }
.meta { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 0.75rem; margin-top: 1rem; }
.meta div { background: #131824; border: 1px solid #1f2836; border-radius: 8px; padding: 0.6rem 0.8rem; }
.meta .k { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; }
.meta .v { font-family: ui-monospace, monospace; color: #e2e8f0; }
table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.45rem 0.7rem; border-bottom: 1px solid #1f2836; }
th { color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
td { font-family: ui-monospace, monospace; color: #e2e8f0; }
td.num, th.num { text-align: right; }
.panel { background: #131824; border: 1px solid #1f2836; border-radius: 10px;
         padding: 0.4rem 1rem 1rem; overflow-x: auto; }
footer { margin-top: 2.5rem; color: #475569; font-size: 0.8rem; }
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _num(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _stat_cell(stat: dict) -> str:
    if not stat or stat.get("mean") is None:
        return "—"
    mean = stat["mean"]
    lo, hi = stat.get("ci95_lo"), stat.get("ci95_hi")
    half = (hi - lo) / 2 if lo is not None and hi is not None else None
    text = _num(mean)
    if half:
        text += f" ± {_num(half)}"
    return text


def _action_highlights(graph_name: str) -> dict:
    """Per-agent action counts + success rate, scanned from the action-log CSV."""
    path = actions_log._path(graph_name)
    agents: dict[str, dict[str, dict]] = {}
    if not path.is_file():
        return agents
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        name_cols = [c for c in (reader.fieldnames or []) if c.endswith("_action_name")]
        for row in reader:
            for col in name_cols:
                agent = col[: -len("_action_name")]
                action = row.get(col) or "—"
                bucket = agents.setdefault(agent, {}).setdefault(
                    action, {"count": 0, "success": 0}
                )
                bucket["count"] += 1
                if str(row.get(f"{agent}_action_success", "")).lower() in ("true", "1"):
                    bucket["success"] += 1
    return agents


def build_report_html(record: dict, graph_name: str, generated_at: str) -> str:
    summary = actions_log.summary(graph_name)  # raises not_found if absent
    params = record.get("params", {})
    metrics = summary.get("metrics", [])

    meta_items = [
        ("Run", record.get("id", "")),
        ("Experiment", record.get("experiment_name", "")),
        ("Checkpoint", record.get("checkpoint", "")),
        ("Network", params.get("network_config", "")),
        ("Seeds", ", ".join(str(s) for s in summary.get("seeds", []))),
        ("Episodes × steps", f"{summary.get('num_episodes')} × {summary.get('num_steps')}"),
        ("Total episodes", summary.get("total_episodes", "")),
    ]
    meta_html = "".join(
        f'<div><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>'
        for k, v in meta_items
    )

    # Overall stats.
    overall = summary.get("overall", {})
    overall_rows = "".join(
        f"<tr><td>{_esc(m)}</td><td class='num'>{_esc(_stat_cell(overall.get(m, {})))}</td>"
        f"<td class='num'>{_esc(_num(overall.get(m, {}).get('std')))}</td>"
        f"<td class='num'>{_esc(_num(overall.get(m, {}).get('min')))}</td>"
        f"<td class='num'>{_esc(_num(overall.get(m, {}).get('max')))}</td>"
        f"<td class='num'>{_esc(overall.get(m, {}).get('n', 0))}</td></tr>"
        for m in metrics
    )
    overall_html = (
        "<h2>Overall (mean ± 95% CI)</h2><div class='panel'><table>"
        "<thead><tr><th>Metric</th><th class='num'>Mean ± CI</th><th class='num'>Std</th>"
        "<th class='num'>Min</th><th class='num'>Max</th><th class='num'>n</th></tr></thead>"
        f"<tbody>{overall_rows}</tbody></table></div>"
    )

    # Per-seed table (only for multi-seed batches).
    per_seed_html = ""
    if len(summary.get("seeds", [])) > 1:
        head = "".join(f"<th class='num'>{_esc(m)}</th>" for m in metrics)
        body = "".join(
            "<tr><td>" + _esc(block.get("seed")) + "</td><td class='num'>"
            + _esc(block.get("episodes"))
            + "</td>"
            + "".join(
                f"<td class='num'>{_esc(_stat_cell(block.get('metrics', {}).get(m, {})))}</td>"
                for m in metrics
            )
            + "</tr>"
            for block in summary.get("per_seed", [])
        )
        per_seed_html = (
            "<h2>Per seed</h2><div class='panel'><table><thead><tr><th>Seed</th>"
            f"<th class='num'>Episodes</th>{head}</tr></thead><tbody>{body}</tbody></table></div>"
        )

    # Per-episode table.
    ep_rows = "".join(
        "<tr><td>" + _esc(ep.get("episode")) + "</td><td>" + _esc(ep.get("seed"))
        + "</td><td class='num'>" + _esc(ep.get("steps")) + "</td>"
        + "".join(f"<td class='num'>{_esc(_num(ep.get(m)))}</td>" for m in metrics)
        + "</tr>"
        for ep in summary.get("per_episode", [])
    )
    ep_head = "".join(f"<th class='num'>{_esc(m)}</th>" for m in metrics)
    per_episode_html = (
        "<h2>Per episode</h2><div class='panel'><table><thead><tr><th>Episode</th><th>Seed</th>"
        f"<th class='num'>Steps</th>{ep_head}</tr></thead><tbody>{ep_rows}</tbody></table></div>"
    )

    # Action highlights.
    highlights = _action_highlights(graph_name)
    action_html = ""
    for agent, actions in sorted(highlights.items()):
        rows = "".join(
            f"<tr><td>{_esc(name)}</td><td class='num'>{_esc(info['count'])}</td>"
            f"<td class='num'>{_esc(_num(100 * info['success'] / info['count'], 0))}%</td></tr>"
            for name, info in sorted(actions.items(), key=lambda kv: -kv[1]["count"])
        )
        action_html += (
            f"<h2>{_esc(agent)} actions</h2><div class='panel'><table><thead><tr><th>Action</th>"
            "<th class='num'>Count</th><th class='num'>Success</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Evaluation report — {_esc(record.get('id', ''))}</title>"
        f"<style>{_STYLE}</style></head><body>"
        f"<h1>Evaluation report</h1>"
        f"<div class='sub'>{_esc(record.get('display_name', ''))}</div>"
        f"<div class='meta'>{meta_html}</div>"
        f"{overall_html}{per_seed_html}{per_episode_html}{action_html}"
        f"<footer>Generated {_esc(generated_at)} · Cyberwheel</footer>"
        "</body></html>"
    )
