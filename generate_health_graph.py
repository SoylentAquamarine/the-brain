#!/usr/bin/env python3
"""
generate_health_graph.py — Generate health graphs from stats/health_log.json.

Produces:
  assets/health_graph.png   — latency over time, last 24 hours
  assets/health_uptime.png  — uptime % bar chart, last 24 hours

Also prints a markdown summary table for use in the README.

Run manually:   python generate_health_graph.py
Run via CI:     called after health_check.py in health-check.yml
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — required in CI
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

HEALTH_LOG  = Path(__file__).parent / "stats" / "health_log.json"
GRAPH_PATH  = Path(__file__).parent / "assets" / "health_graph.png"
UPTIME_PATH = Path(__file__).parent / "assets" / "health_uptime.png"

# Colour palette — one per provider, readable on both light and dark backgrounds
PROVIDER_COLORS = {
    "cerebras":    "#FF6B6B",
    "groq":        "#4ECDC4",
    "gemini":      "#45B7D1",
    "mistral":     "#96CEB4",
    "sambanova":   "#FFEAA7",
    "fireworks":   "#DDA0DD",
    "huggingface": "#98D8C8",
    "openai":      "#74B9FF",
    "pollinations":"#FD79A8",
}


def load_log() -> list:
    if not HEALTH_LOG.exists():
        print("No health log found — run health_check.py first.")
        sys.exit(0)
    return json.loads(HEALTH_LOG.read_text(encoding="utf-8"))


def filter_last_n_hours(entries: list, hours: int = 24) -> list:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = []
    for e in entries:
        try:
            ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
            if ts >= cutoff:
                result.append(e)
        except Exception:
            pass
    return result


def build_series(entries: list) -> dict:
    """Return {provider: [(datetime, latency_ms, quality), ...]}"""
    series = defaultdict(list)
    for e in entries:
        if e["status"] == "no_key":
            continue
        try:
            ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
            series[e["provider"]].append((ts, e["latency_ms"], e["quality"]))
        except Exception:
            pass
    return dict(series)


def generate_latency_graph(series: dict) -> None:
    """Line chart — latency over time per provider."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for provider, points in sorted(series.items()):
        if not points:
            continue
        times    = [p[0] for p in points]
        latencies = [p[1] if p[2] > 0 else None for p in points]  # None = failed
        color    = PROVIDER_COLORS.get(provider, "#FFFFFF")

        # Plot line (skip None gaps)
        valid = [(t, l) for t, l in zip(times, latencies) if l is not None]
        if valid:
            vt, vl = zip(*valid)
            ax.plot(vt, vl, color=color, linewidth=1.5, label=provider, alpha=0.85)
            ax.scatter(vt, vl, color=color, s=18, zorder=5)

        # Mark failures as red X
        failed = [(t, 0) for t, q in zip(times, [p[2] for p in points]) if q == 0.0]
        if failed:
            ft, _ = zip(*failed)
            ax.scatter(ft, [0] * len(ft), marker="x", color="#FF0000", s=40, zorder=6)

    ax.set_xlabel("Time (UTC)", color="#cccccc", fontsize=10)
    ax.set_ylabel("Latency (ms)", color="#cccccc", fontsize=10)
    ax.set_title("Provider Latency — Last 24 Hours", color="#ffffff", fontsize=13, pad=12)
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#444444")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=30, ha="right")
    ax.legend(
        loc="upper right", framealpha=0.3, facecolor="#0f0f23",
        edgecolor="#555555", labelcolor="#eeeeee", fontsize=9
    )
    ax.grid(True, color="#2a2a4a", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPH_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved latency graph -> {GRAPH_PATH}")


def generate_uptime_graph(series: dict) -> None:
    """Horizontal bar chart — uptime % per provider."""
    providers = []
    uptimes   = []
    colors    = []

    for provider, points in sorted(series.items()):
        if not points:
            continue
        total  = len(points)
        ok     = sum(1 for p in points if p[2] >= 0.8)
        pct    = (ok / total * 100) if total else 0
        providers.append(provider)
        uptimes.append(pct)
        colors.append(PROVIDER_COLORS.get(provider, "#FFFFFF"))

    if not providers:
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(providers) * 0.7)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.barh(providers, uptimes, color=colors, alpha=0.85, height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Uptime %", color="#cccccc", fontsize=10)
    ax.set_title("Provider Uptime — Last 24 Hours", color="#ffffff", fontsize=13, pad=12)
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#444444")
    ax.grid(True, axis="x", color="#2a2a4a", linewidth=0.5, alpha=0.7)
    ax.axvline(x=100, color="#555555", linewidth=0.5)

    for bar, pct in zip(bars, uptimes):
        ax.text(
            min(pct + 1, 98), bar.get_y() + bar.get_height() / 2,
            f"{pct:.0f}%", va="center", color="#ffffff", fontsize=9
        )

    plt.tight_layout()
    fig.savefig(UPTIME_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved uptime graph  -> {UPTIME_PATH}")


def build_readme_table(entries_24h: list, series: dict) -> str:
    """Return a markdown table for the README health section."""
    now = datetime.now(timezone.utc)
    lines = []
    lines.append(f"*Last check: {now.strftime('%Y-%m-%d %H:%M UTC')} — auto-generated by health-check.yml*")
    lines.append("")
    lines.append("| Provider | Status | Last Latency | Avg Latency (24h) | Uptime (24h) |")
    lines.append("|---|---|---|---|---|")

    # Get last entry per provider
    last = {}
    for e in entries_24h:
        last[e["provider"]] = e

    for provider, points in sorted(series.items()):
        if not points:
            continue
        rec    = last.get(provider, {})
        status = rec.get("status", "?")
        icon   = {"ok": "OK", "degraded": "WARN", "error": "FAIL", "no_key": "SKIP"}.get(status, "?")
        last_ms = f"{rec.get('latency_ms', 0)}ms" if status not in ("error", "no_key") else "—"
        valid   = [p[1] for p in points if p[2] > 0 and p[1] > 0]
        avg_ms  = f"{int(sum(valid)/len(valid))}ms" if valid else "—"
        total   = len(points)
        ok_ct   = sum(1 for p in points if p[2] >= 0.8)
        uptime  = f"{ok_ct/total*100:.0f}%" if total else "—"
        lines.append(f"| **{provider}** | {icon} | {last_ms} | {avg_ms} | {uptime} |")

    return "\n".join(lines)


def update_readme(table: str) -> None:
    readme = Path(__file__).parent / "README.md"
    if not readme.exists():
        print("  README.md not found — skipping update.")
        return

    content  = readme.read_text(encoding="utf-8")
    start_tag = "<!-- HEALTH_START -->"
    end_tag   = "<!-- HEALTH_END -->"

    new_section = (
        f"{start_tag}\n"
        "## Provider Health\n\n"
        f"{table}\n\n"
        "### Latency over time (last 24h)\n\n"
        "![Provider Latency](assets/health_graph.png)\n\n"
        "### Uptime (last 24h)\n\n"
        "![Provider Uptime](assets/health_uptime.png)\n"
        f"{end_tag}"
    )

    if start_tag in content and end_tag in content:
        before = content[:content.index(start_tag)]
        after  = content[content.index(end_tag) + len(end_tag):]
        content = before + new_section + after
    else:
        # Insert before the License section
        content = content.replace(
            "## License",
            new_section + "\n\n---\n\n## License"
        )

    readme.write_text(content, encoding="utf-8")
    print(f"  Updated README.md health section.")


def main() -> None:
    print("\n=== Generating Health Graphs ===\n")

    if not HAS_MATPLOTLIB:
        print("matplotlib not installed — skipping graphs. Run: pip install matplotlib")
        sys.exit(1)

    log         = load_log()
    entries_24h = filter_last_n_hours(log, hours=24)
    series      = build_series(entries_24h)

    if not series:
        print("No data in the last 24 hours yet.")
        sys.exit(0)

    print(f"  Found {len(entries_24h)} entries across {len(series)} providers\n")

    generate_latency_graph(series)
    generate_uptime_graph(series)

    table = build_readme_table(entries_24h, series)
    update_readme(table)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
