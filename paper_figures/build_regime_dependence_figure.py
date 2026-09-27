"""Build the regime-dependence figure (Panel A task cards + Panel B dot plot).

Self-contained: all values are inlined from
`paper_strengthening_experiments/16_main_table_uncertainty/results.json`
(the Fisher-z pooled Spearman rhos and 95% intervals that back the paper's
main-results table).

Produces `figure.pdf` and `figure.png` in the same folder.

Run:
    uv run --with matplotlib python paper_strengthening_experiments/32_regime_dependence_figure/build_figure.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data (mirrors exp 16 results.json, which mirrors the paper's main table)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    label: str
    rho: float
    ci_lo: float
    ci_hi: float
    n_models: int
    regime: int  # 0..4, index into REGIMES


REGIMES = [
    dict(key="short_gen", name="short-horizon generation", short="short-horizon generation"),
    dict(key="exec_pred", name="execution prediction", short="execution prediction"),
    dict(key="reverse", name="reverse reasoning", short="reverse reasoning"),
    dict(key="constrained", name="constrained generation", short="constrained generation"),
    dict(key="whole_prog", name="whole-program synthesis", short="whole-program synthesis"),
]

# Palette: five distinct but harmonious accent colors. Strength of signal is
# encoded by the x-position in Panel B, not by color — color ties Panel A cards
# to Panel B rows so the reader can trace each regime across panels.
REGIME_COLORS = {
    0: "#1D4ED8",  # deep blue      — short-horizon generation
    1: "#0D9488",  # teal           — execution prediction
    2: "#D97706",  # amber          — reverse reasoning
    3: "#7C3AED",  # violet         — constrained generation
    4: "#64748B",  # slate gray     — whole-program synthesis
}
REGIME_FILLS = {
    0: "#DBEAFE",
    1: "#CCFBF1",
    2: "#FEF3C7",
    3: "#EDE9FE",
    4: "#E2E8F0",
}

CARD_TASKS = [
    # (regime idx, title, input_label, input_lines, output_label,
    #  output_lines, benchmark_caption)
    (
        0,
        "Short-horizon generation",
        "task description + stub",
        [
            "// Are any two numbers",
            "// distance < threshold?",
            "function has_close(",
            "  numbers, threshold){",
        ],
        "model completes body",
        [
            "for (j = i+1; ...) {",
            "  if (...) return true;",
            "}",
            "return false;",
        ],
        "MultiPL-E \u00b7 IFEvalCode \u00b7 FullStack",
    ),
    (
        1,
        "Execution prediction",
        "code + input",
        ["def f(nums):", "  ...", "assert candidate([...])", "== ????"],
        "output value",
        ["[(4, 1), (2, 3)]"],
        "CRUXEval-X output",
    ),
    (
        2,
        "Reverse reasoning",
        "code + output",
        ["def f(nums):", "  ...", "assert candidate(????)", "== [(4, 1), ...]"],
        "input that yields it",
        ["[1, 1, 3, 1, 3, 1]"],
        "CRUXEval-X input",
    ),
    (
        3,
        "Constrained generation",
        "NL task + constraints",
        [
            "Implement fizz_buzz()",
            "Constraints:",
            "1. use lambda",
            "2. no loops",
            "3. recursive",
        ],
        "code that obeys them",
        ["convert = lambda arr:", "  ...", "# recursive; no loops"],
        "IFEvalCode instruction",
    ),
    (
        4,
        "Whole-program synthesis",
        "contest statement",
        [
            "9x9 multiplication table:",
            "sum entries not equal X.",
            "stdin: X",
            "stdout: sum",
        ],
        "complete program",
        ["import sys", "X = int(input())", "print(ans)"],
        "Multi-LCB",
    ),
]

ROWS: list[Row] = [
    Row("MultiPL-E (BigCode)", -0.813, -0.851, -0.767, 27, 0),
    Row("MultiPL-E (Qwen pool)", -0.675, -0.778, -0.538, 28, 0),
    Row("IFEvalCode correctness (EN)", -0.545, -0.678, -0.377, 21, 0),
    Row("IFEvalCode correctness (ZH)", -0.607, -0.725, -0.454, 21, 0),
    Row("FullStack (Adv. Prog.)", -0.567, -0.691, -0.411, 15, 0),
    Row("FullStack (Basic Prog.)", -0.598, -0.707, -0.461, 15, 0),
    Row("CRUXEval-X output", -0.429, -0.521, -0.326, 18, 1),
    Row("CRUXEval-X input", -0.122, -0.237, -0.003, 18, 2),
    Row("IFEvalCode instruction (EN)", -0.142, -0.343, 0.071, 21, 3),
    Row("IFEvalCode instruction (ZH)", -0.041, -0.250, 0.171, 21, 3),
    Row("Multi-LCB", 0.105, -0.068, 0.272, 16, 4),
]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Inter",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "DejaVu Sans",
        ],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "axes.edgecolor": "#6B7280",
        "axes.labelcolor": "#1F2937",
    }
)

INK = "#111827"
SUBINK = "#4B5563"
GRID = "#E5E7EB"
CODE_BG = "#F3F4F6"
CARD_BG = "#FFFFFF"


# ---------------------------------------------------------------------------
# Panel A: task-card schematics
# ---------------------------------------------------------------------------


def draw_card(
    ax, x, y, w, h, regime_idx, title, in_label, in_lines, out_label, out_lines, caption
):
    accent = REGIME_COLORS[regime_idx]
    fill = REGIME_FILLS[regime_idx]

    # Outer card: soft rounded rectangle, white fill, light border.
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.10",
            linewidth=0.8,
            edgecolor="#D1D5DB",
            facecolor=CARD_BG,
            zorder=1,
        )
    )

    # Top accent strip (thin colored band at the very top of the card).
    strip_h = 0.08
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - strip_h),
            w,
            strip_h,
            boxstyle="round,pad=0,rounding_size=0.10",
            linewidth=0,
            facecolor=accent,
            zorder=2,
        )
    )
    # Square off the strip's bottom so only the top corners round.
    ax.add_patch(
        Rectangle(
            (x, y + h - strip_h),
            w,
            strip_h * 0.55,
            linewidth=0,
            facecolor=accent,
            zorder=2,
        )
    )

    # Title: bold, in the accent color, placed BELOW the strip on white.
    title_y = y + h - strip_h - 0.14
    title_fs = 9.6
    if len(title) >= 22:
        title_fs = 8.9
    ax.text(
        x + w / 2,
        title_y,
        title,
        ha="center",
        va="center",
        color=accent,
        fontsize=title_fs,
        fontweight="bold",
        zorder=3,
    )

    # Layout: title | input label + box | arrow | output label + box | caption.
    pad = 0.07
    label_gap = 0.14
    caption_y = y + 0.025
    body_top = title_y - 0.11
    body_bot = caption_y + 0.19
    usable = body_top - body_bot
    arrow_h = 0.16
    box_h = (usable - 2 * label_gap - arrow_h) / 2
    box_w = w - 2 * pad

    # Input: label, then code box.
    in_label_y = body_top
    in_box_top = in_label_y - label_gap * 0.88
    in_box_y = in_box_top - box_h
    ax.text(
        x + pad,
        in_label_y,
        in_label,
        ha="left",
        va="top",
        color=SUBINK,
        fontsize=7.3,
        fontweight="bold",
        zorder=3,
    )
    _code_box(ax, x + pad, in_box_y, box_w, box_h, in_lines, fill)

    # Output: label, then code box.
    out_label_y = in_box_y - arrow_h * 0.95
    out_box_top = out_label_y - label_gap * 0.88
    out_box_y = out_box_top - box_h
    # Arrow between boxes (between input box bottom and output box top).
    ax.add_patch(
        FancyArrowPatch(
            (x + w / 2, in_box_y - 0.01),
            (x + w / 2, out_label_y + 0.035),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.3,
            color=accent,
            zorder=3,
        )
    )
    ax.text(
        x + pad,
        out_label_y,
        out_label,
        ha="left",
        va="top",
        color=SUBINK,
        fontsize=7.3,
        fontweight="bold",
        zorder=3,
    )
    _code_box(ax, x + pad, out_box_y, box_w, box_h, out_lines, fill)

    # Caption: benchmarks this regime covers.
    wrapped_caption = _wrap_caption(caption, width=25)
    ax.text(
        x + w / 2,
        caption_y,
        wrapped_caption,
        ha="center",
        va="bottom",
        color=SUBINK,
        fontsize=6.5,
        style="italic",
        linespacing=1.05,
        zorder=3,
    )


def _code_box(ax, x, y, w, h, lines, tint):
    wrapped_lines = _wrap_code_lines(lines, width=24)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.05",
            linewidth=0.4,
            edgecolor="#E5E7EB",
            facecolor=tint,
            zorder=2,
        )
    )
    # Render lines top-down inside the box, with a small top padding so the
    # first line isn't flush with the border.
    n = max(len(wrapped_lines), 1)
    top_pad = 0.035
    line_h = (h - 2 * top_pad) / n
    font_size = 7.5 if n <= 4 else 6.8
    for i, text in enumerate(wrapped_lines):
        ax.text(
        x + 0.045,
            y + h - top_pad - (i + 0.5) * line_h,
            text,
            ha="left",
            va="center",
            fontsize=font_size,
            color=INK,
            family="monospace",
            zorder=3,
            clip_on=True,
        )


def _wrap_caption(caption: str, width: int) -> str:
    parts = [part.strip() for part in caption.split("\u00b7")]
    if len(parts) == 1:
        return "\n".join(textwrap.wrap(caption, width=width, break_long_words=False))

    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current} \u00b7 {part}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)
    return "\n".join(lines)


def _wrap_code_lines(lines: list[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        parts = textwrap.wrap(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        wrapped.extend(parts if parts else [""])
    return wrapped


def draw_panel_a(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    n = len(CARD_TASKS)
    gap = 0.08
    total_w = 10
    card_w = (total_w - gap * (n - 1)) / n
    card_h = 3.1
    y0 = 0.0

    for i, (
        regime_idx,
        title,
        in_label,
        in_lines,
        out_label,
        out_lines,
        cap,
    ) in enumerate(CARD_TASKS):
        x0 = i * (card_w + gap)
        draw_card(
            ax,
            x0,
            y0,
            card_w,
            card_h,
            regime_idx,
            title,
            in_label,
            in_lines,
            out_label,
            out_lines,
            cap,
        )

    # Panel title omitted to save vertical space in the final paper figure.


# ---------------------------------------------------------------------------
# Panel B: dot + range plot
# ---------------------------------------------------------------------------


def draw_panel_b(ax):
    n = len(ROWS)
    ys = list(range(n, 0, -1))  # first row at top

    # Data range used for dots; the visible axis extends further left so
    # in-chart slice labels fit without overlapping the whiskers.
    x_data_lo, x_data_hi = -0.95, 0.32
    ax.set_xlim(x_data_lo, x_data_hi)
    ax.set_ylim(0.4, n + 0.6)

    # Regime-group shading: one soft horizontal band per regime.
    groups: list[tuple[int, int, int]] = []  # (regime, y_hi, y_lo)
    start = 0
    for i in range(1, n + 1):
        if i == n or ROWS[i].regime != ROWS[start].regime:
            y_hi = ys[start] + 0.5
            y_lo = ys[i - 1] - 0.5
            groups.append((ROWS[start].regime, y_hi, y_lo))
            start = i
    band_width = x_data_hi - x_data_lo
    for regime, y_hi, y_lo in groups:
        ax.add_patch(
            Rectangle(
                (x_data_lo, y_lo),
                band_width,
                y_hi - y_lo,
                linewidth=0,
                facecolor=REGIME_FILLS[regime],
                alpha=0.45,
                zorder=0,
            )
        )

    # Zero reference line.
    ax.axvline(0, color="#9CA3AF", linewidth=0.9, linestyle="--", zorder=1)

    # Vertical grid.
    for xt in [-0.8, -0.6, -0.4, -0.2, 0.2]:
        ax.axvline(xt, color=GRID, linewidth=0.4, zorder=0)

    # Per-row: slice label (left of whisker, in-chart), CI whisker + dot,
    # rho value (right of whisker).
    for row, y in zip(ROWS, ys):
        color = REGIME_COLORS[row.regime]

        # In-chart slice label: above the whisker, left-aligned at xmin.
        ax.text(
            x_data_lo + 0.015,
            y + 0.22,
            row.label,
            ha="left",
            va="bottom",
            fontsize=10.2,
            color=INK,
            fontweight="bold",
            zorder=4,
        )

        # CI whisker
        ax.plot(
            [row.ci_lo, row.ci_hi],
            [y, y],
            color=color,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=2,
        )
        # Whisker caps
        cap_h = 0.14
        for xend in (row.ci_lo, row.ci_hi):
            ax.plot(
                [xend, xend],
                [y - cap_h, y + cap_h],
                color=color,
                linewidth=1.3,
                zorder=2,
            )
        # Dot (white outline for contrast over shaded band)
        ax.plot(
            row.rho,
            y,
            marker="o",
            markersize=7.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.3,
            zorder=3,
        )
        # Numeric rho just past the right whisker end.
        rho_x = row.ci_hi + 0.015
        rho_y = y
        rho_ha = "left"
        rho_va = "center"
        if rho_x > x_data_hi - 0.06:
            rho_x = x_data_hi - 0.012
            rho_y = y + 0.2
            rho_ha = "right"
            rho_va = "bottom"
        ax.text(
            rho_x,
            rho_y,
            f"{row.rho:+.2f}",
            ha=rho_ha,
            va=rho_va,
            fontsize=9.4,
            color=color,
            fontweight="bold",
            zorder=4,
        )

    # Hide y-axis entirely — slice names are drawn inside the chart above.
    ax.set_yticks([])
    for side in ("left",):
        ax.spines[side].set_visible(False)

    # Regime-group labels inside the shaded bands.
    label_x = x_data_hi - 0.015
    custom_y = {
        0: groups[0][1] - 0.24 if len(groups) > 0 else None,
        1: None,
        2: None,
        3: None,
        4: groups[-1][2] + 0.15 if groups else None,
    }
    for regime, y_hi, y_lo in groups:
        y_mid = (y_hi + y_lo) / 2
        label_y = custom_y.get(regime)
        if label_y is None:
            label_y = y_mid
        ax.text(
            label_x,
            label_y,
            REGIMES[regime]["short"],
            ha="right",
            va="center",
            fontsize=10.4,
            fontweight="bold",
            color=REGIME_COLORS[regime],
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=0.14",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )

    # x-axis ticks and label.
    ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2])
    ax.set_xticklabels(
        ["-0.8", "-0.6", "-0.4", "-0.2", "0", "+0.2"], fontsize=9.8
    )
    ax.set_xlabel(
        r"Pooled within-model Spearman $\rho$ correlation   "
        r"(Code IP vs. benchmark score)",
        fontsize=10.8,
        color=INK,
        labelpad=8,
    )

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", length=3, pad=3, color="#9CA3AF")

    # Direction captions below the axis.
    ax.text(
        -0.94,
        -0.11,
        "\u2190 lower Code IP aligns with higher benchmark score",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8.8,
        color=SUBINK,
        style="italic",
    )
    ax.text(
        0.02,
        -0.11,
        "  no alignment  \u2192",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8.8,
        color=SUBINK,
        style="italic",
    )

    # Panel title omitted to save vertical space in the final paper figure.


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


def main():
    fig = plt.figure(figsize=(10.2, 8.8))
    # Panel A on top, Panel B below, with space between.
    # left=0.05 because Panel B carries slice labels inside the chart.
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 2.05],
        hspace=0.14,
        left=0.05,
        right=0.94,
        top=0.97,
        bottom=0.07,
    )

    ax_a = fig.add_subplot(gs[0])
    draw_panel_a(ax_a)

    ax_b = fig.add_subplot(gs[1])
    draw_panel_b(ax_b)

    pdf_path = HERE / "regime_dependence_figure.pdf"
    png_path = HERE / "regime_dependence_figure.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    try:
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"wrote {pdf_path.name} and {png_path.name}")
    except PermissionError:
        print(
            f"wrote {png_path.name} (could not overwrite {pdf_path.name} — "
            f"close any open viewer and rerun)"
        )


if __name__ == "__main__":
    main()
