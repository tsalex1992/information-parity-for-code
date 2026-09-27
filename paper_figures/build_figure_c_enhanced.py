"""Enhanced Figure C: strip/beeswarm with median spine + blue/orange split.

Run:  uv run --with matplotlib --with scipy python paper/figures/build_figure_c_enhanced.py
"""
from __future__ import annotations

import json, math, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeip.config import REFERENCE_LANG, RESULTS_DIR, discover_models
from codeip.language_aliases import resolve_metric_language
from codeip.loaders import load_benchmarks

OUT = Path(__file__).resolve().parent
ABS_RESULTS_DIR = ROOT / RESULTS_DIR

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})

DEEP_BLUE   = "#1E40AF"
MID_BLUE    = "#3B82F6"
LIGHT_BLUE  = "#93C5FD"
PALE_BLUE   = "#DBEAFE"
DEEP_ORANGE = "#C2410C"
MID_ORANGE  = "#F97316"
LIGHT_ORANGE= "#FDBA74"
PALE_ORANGE = "#FFF7ED"
DARK        = "#1F2937"
GRAY        = "#6B7280"
LIGHT_GRAY  = "#D1D5DB"
ZEBRA       = "#F3F4F6"


def load_all():
    rows = []
    for dn, info in discover_models(ABS_RESULTS_DIR).items():
        p = ABS_RESULTS_DIR / info["metrics_file"]
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        curated = data.get("results", {}).get("curated", {})
        code_ip = {l: m["code_ip"] for l, m in curated.items() if "code_ip" in m}
        rows.append(dict(
            display_name=dn, model_path=info["model_path"],
            code_ip=code_ip, benchmarks=load_benchmarks(info["model_path"]),
        ))
    return rows


def collect_data(rows):
    all_ip, bench_multi, bench_lcb = {}, {}, {}
    for r in rows:
        for lang, val in r["code_ip"].items():
            if lang == REFERENCE_LANG:
                continue
            all_ip.setdefault(lang, []).append(val)
        for bkey, store in [("MultiPL-E (BigCode)", bench_multi), ("Multi-LCB", bench_lcb)]:
            if bkey not in r["benchmarks"]:
                continue
            for blang, score in r["benchmarks"][bkey].items():
                if blang == REFERENCE_LANG:
                    continue
                ml = resolve_metric_language(blang, set(r["code_ip"]))
                if ml:
                    store.setdefault(ml, []).append(float(score))
    return all_ip, bench_multi, bench_lcb


def build(rows):
    all_ip, bench_multi, bench_lcb = collect_data(rows)

    med_ip = {l: np.median(v) for l, v in all_ip.items() if len(v) >= 3}
    shared = sorted(
        set(med_ip) & (set(bench_multi) | set(bench_lcb)),
        key=lambda l: med_ip[l],
    )
    if not shared:
        shared = sorted(med_ip, key=lambda l: med_ip[l])

    n = len(shared)
    ypos = {l: i for i, l in enumerate(shared)}
    rng = np.random.default_rng(42)

    fig, (ax_ip, ax_multi, ax_lcb) = plt.subplots(
        1, 3, figsize=(14, 0.40 * n + 2.0),
        gridspec_kw={"width_ratios": [2.6, 1.5, 1.5], "wspace": 0.06},
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.87, bottom=0.08)

    # ── Zebra striping (Code IP panel only) ──
    for i in range(n):
        if i % 2 == 0:
            ax_ip.axhspan(i - 0.45, i + 0.45, color=ZEBRA, alpha=0.55, zorder=0, linewidth=0)
    # Light horizontal guides across benchmark panels
    for i in range(n):
        for a in [ax_multi, ax_lcb]:
            a.axhline(i, color="#E5E7EB", lw=0.3, zorder=0)

    # ── Strip 1: Code IP beeswarm ──
    for lang in shared:
        y = ypos[lang]
        vals = list(all_ip.get(lang, {}).values()) if isinstance(all_ip.get(lang), dict) else all_ip.get(lang, [])
        if not vals:
            continue
        jitter = rng.normal(0, 0.13, len(vals))
        ax_ip.scatter(vals, y + jitter, c=LIGHT_BLUE, s=10, alpha=0.50,
                      edgecolors="none", zorder=2)
        med = np.median(vals)
        p10, p90 = np.percentile(vals, [10, 90])
        ax_ip.plot([p10, p90], [y, y], color=GRAY, lw=0.7, alpha=0.4, zorder=3)
        ax_ip.scatter([med], [y], color=DEEP_BLUE, s=45, zorder=5,
                      marker="|", linewidths=2.0)

    ax_ip.axvline(1.0, color=LIGHT_GRAY, lw=0.8, ls=":", zorder=1)
    ax_ip.set_yticks(range(n))
    ax_ip.set_yticklabels(shared, fontsize=9, fontweight="bold")
    ax_ip.set_xlabel("Code IP", fontsize=10, color=GRAY)
    ax_ip.set_title("NLL ratio to Python", fontsize=11.5, fontweight="bold", color=DARK, pad=18)
    ax_ip.text(0.5, 1.01, "22-algorithm curated corpus", transform=ax_ip.transAxes,
               ha="center", va="bottom", fontsize=8.5, color=GRAY, style="italic")
    ax_ip.set_ylim(-0.7, n - 0.3)

    # ── Compute Code IP rank order for displacement shading ──
    ip_rank = {l: i for i, l in enumerate(shared)}

    # ── Strips 2-3: benchmark scores with spine + displacement shading ──
    strip_configs = [
        (ax_multi, bench_multi, "MultiPL-E pass@1", "Matched generation",
         LIGHT_BLUE, DEEP_BLUE, DEEP_BLUE, PALE_BLUE),
        (ax_lcb, bench_lcb, "Multi-LCB pass@1", "Whole-program synthesis",
         LIGHT_ORANGE, DEEP_ORANGE, DEEP_ORANGE, PALE_ORANGE),
    ]

    global_bench_max = 0
    for _, bdata, _, _, _, _, _, _ in strip_configs:
        for lang in shared:
            vals = bdata.get(lang, [])
            if vals:
                global_bench_max = max(global_bench_max, max(vals))
    global_bench_max = math.ceil(global_bench_max / 5) * 5 + 5

    for ax, bdata, title, subtitle, dot_color, med_color, spine_color, shading_color in strip_configs:
        # Compute benchmark medians and ranks for displacement shading
        bench_meds = {}
        for lang in shared:
            vals = bdata.get(lang, [])
            if vals:
                bench_meds[lang] = np.median(vals)

        bench_sorted = sorted(bench_meds, key=lambda l: bench_meds[l])
        bench_rank = {l: i for i, l in enumerate(bench_sorted)}

        # Displacement shading -- intensity proportional to how far this
        # language's benchmark rank deviates from its Code IP rank
        for lang in shared:
            y = ypos[lang]
            if lang not in bench_rank:
                # No benchmark data: leave row white
                ax.axhspan(y - 0.45, y + 0.45, color="white", alpha=1.0,
                           zorder=0, linewidth=0)
                continue
            disp = abs(ip_rank[lang] - bench_rank[lang])
            intensity = disp / max(n - 1, 1)
            alpha = min(0.08 + 0.35 * intensity, 0.40)
            ax.axhspan(y - 0.45, y + 0.45, color=shading_color, alpha=alpha,
                       zorder=0, linewidth=0)

        # Dots + medians
        for lang in shared:
            y = ypos[lang]
            vals = bdata.get(lang, [])
            if not vals:
                continue
            jitter = rng.normal(0, 0.12, len(vals))
            ax.scatter(vals, y + jitter, c=dot_color, s=14, alpha=0.55,
                       edgecolors="none", zorder=2)
            med = np.median(vals)
            ax.scatter([med], [y], color=med_color, s=50, zorder=5,
                       marker="|", linewidths=2.0)

        # ── Median spine ──
        spine_xs, spine_ys = [], []
        for lang in shared:
            if lang in bench_meds:
                spine_xs.append(bench_meds[lang])
                spine_ys.append(ypos[lang])
        if spine_xs:
            ax.plot(spine_xs, spine_ys, color=spine_color, lw=2.5, alpha=0.75,
                    zorder=4, solid_capstyle="round")

        # ── Rank-correlation badge ──
        common = [l for l in shared if l in bench_rank]
        if len(common) >= 4:
            ip_r = np.array([ip_rank[l] for l in common])
            b_r = np.array([bench_rank[l] for l in common])
            rho, _ = spearmanr(ip_r, b_r)
            ax.text(0.96, 0.97, rf"$\rho = {rho:+.2f}$",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5, fontweight="bold", color=spine_color,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=spine_color, lw=0.8, alpha=0.92),
                    zorder=10)

        ax.set_yticks(range(n))
        ax.set_yticklabels([])
        ax.set_xlabel("Benchmark score", fontsize=10, color=GRAY)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=DARK, pad=18)
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.5, color=GRAY, style="italic")
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlim(0, global_bench_max)

    fig.subplots_adjust(top=0.93, bottom=0.06)

    fig.savefig(OUT / "figure_c_enhanced.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure_c_enhanced.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved:", OUT / "figure_c_enhanced.pdf")


if __name__ == "__main__":
    rows = load_all()
    print(f"{len(rows)} models loaded")
    build(rows)
