"""
Generate all figures for the adaptive guardrails paper.
Outputs to figures/ directory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour palette ────────────────────────────────────────────────────────────
NL_COLOR       = "#222222"
BASH_COLOR     = "#C0392B"
PYTHON_COLOR   = "#2980B9"
CURL_COLOR     = "#27AE60"
PS_COLOR       = "#E67E22"
SQL_COLOR      = "#8E44AD"

GENERAL_COLOR  = "#E74C3C"
CODEAWARE_COLOR = "#2ECC71"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Evasion Rate Comparison (5 models, horizontal bar chart)
# ─────────────────────────────────────────────────────────────────────────────
def fig_evasion_comparison():
    models = [
        "all-MiniLM-L6-v2\n(General NLP)",
        "all-mpnet-base-v2\n(General NLP, larger)",
        "Cohere embed-english-v3\n(General NLP, commercial)",
        "CodeBERT\n(Code-aware)",
        "Gemini embedding-001\n(Code-aware, commercial)",
    ]
    evasion = [1.000, 1.000, 1.000, 0.084, 0.012]
    ci_lo   = [0.985, 0.985, 0.985, 0.056, 0.004]
    ci_hi   = [1.000, 1.000, 1.000, 0.125, 0.035]
    colors  = [GENERAL_COLOR, GENERAL_COLOR, GENERAL_COLOR,
               CODEAWARE_COLOR, CODEAWARE_COLOR]

    xerr_lo = [e - l for e, l in zip(evasion, ci_lo)]
    xerr_hi = [h - e for h, e in zip(ci_hi, evasion)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(models))
    bars = ax.barh(y, evasion, color=colors, alpha=0.85, height=0.55,
                   xerr=[xerr_lo, xerr_hi], error_kw={"elinewidth": 1.5,
                   "capsize": 4, "capthick": 1.5, "color": "#444444"})

    ax.axvline(x=0.20, color="#888888", linestyle="--", linewidth=1.2,
               label="Detection threshold (τ = 0.20)")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Evasion Rate", fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_title("Code-Form Evasion Rate by Embedding Model\n(50 threats × 5 code forms, n=250 per model)",
                 fontsize=11, pad=12)

    for i, (e, n) in enumerate(zip(evasion, [250,250,250,21,3])):
        label = f"{e*100:.1f}%  ({n}/250)"
        ax.text(e + 0.02, i, label, va="center", fontsize=8.5, color="#222222")

    gen_patch  = mpatches.Patch(color=GENERAL_COLOR,   alpha=0.85, label="General NLP (NL-only training)")
    code_patch = mpatches.Patch(color=CODEAWARE_COLOR, alpha=0.85, label="Code-aware (code+NL training)")
    ax.legend(handles=[gen_patch, code_patch,
                        plt.Line2D([0],[0], color="#888888", linestyle="--", linewidth=1.2,
                                   label="Detection threshold (τ = 0.20)")],
              fontsize=8.5, loc="lower right")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_evasion_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Per-Code-Form Breakdown (CodeBERT residual evasion)
# ─────────────────────────────────────────────────────────────────────────────
def fig_per_form_codebert():
    forms      = ["bash", "python", "curl", "powershell", "sql/js"]
    evasion_cb = [0.060, 0.100, 0.040, 0.020, 0.200]
    evasion_gm = [0.000, 0.000, 0.060, 0.000, 0.000]
    colors_cb  = [BASH_COLOR, PYTHON_COLOR, CURL_COLOR, PS_COLOR, SQL_COLOR]

    x = np.arange(len(forms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - width/2, evasion_cb, width, color=colors_cb, alpha=0.85,
                label="CodeBERT")
    b2 = ax.bar(x + width/2, evasion_gm, width, color="#95A5A6", alpha=0.85,
                label="Gemini embedding-001")

    ax.axhline(y=0.20, color="#888888", linestyle="--", linewidth=1.2,
               label="Detection threshold (τ = 0.20)")
    ax.set_xticks(x)
    ax.set_xticklabels(forms, fontsize=10)
    ax.set_ylabel("Evasion Rate", fontsize=10)
    ax.set_ylim(0, 0.30)
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_yticklabels(["0%","5%","10%","15%","20%","25%","30%"])
    ax.set_title("Residual Evasion Rate per Code Form\n(Code-aware models only, n=50 per form)",
                 fontsize=11, pad=12)

    for bar in b1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h*100:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h*100:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.legend(fontsize=9)

    # annotation for JS gap
    ax.annotate("JavaScript accounts for\n45/50 sql/js slots;\nJS evasion = 22.2%\nvs SQL = 0%",
                xy=(4 - width/2, 0.200), xytext=(3.1, 0.260),
                arrowprops=dict(arrowstyle="->", color="#555"),
                fontsize=7.5, color="#555")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_per_form_codebert.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Evasion Delta: mean NL→code vs NL→NL baseline
# ─────────────────────────────────────────────────────────────────────────────
def fig_distance_distribution():
    models       = ["MiniLM", "mpnet", "Cohere", "CodeBERT", "Gemini"]
    mean_nl_code = [0.5762,   0.5618,  0.5073,  0.1268,     0.1316]
    mean_nl_nl   = [0.8140,   0.7990,  0.7200,  0.0671,     0.0610]  # from stats output
    threshold    = 0.20

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))

    b1 = ax.bar(x - width/2, mean_nl_code, width, color="#3498DB", alpha=0.85,
                label="Mean NL → code distance")
    b2 = ax.bar(x + width/2, mean_nl_nl, width, color="#BDC3C7", alpha=0.85,
                label="Mean NL → NL baseline (pairwise)")

    ax.axhline(y=threshold, color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"Detection threshold τ = {threshold}")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Cosine Distance", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("NL→Code Distance vs NL→NL Baseline per Embedding Model\n"
                 "(Threshold-raising cannot close the gap for general NLP models)",
                 fontsize=10.5, pad=12)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.legend(fontsize=9, loc="upper right")

    ax.annotate("Raising τ to catch\ncode variants here\nwould catch all\nNL pairs too",
                xy=(0 - width/2, threshold), xytext=(0.8, 0.45),
                arrowprops=dict(arrowstyle="->", color="#E74C3C"),
                fontsize=7.5, color="#E74C3C")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_distance_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — 4-System Benchmark (pipeline performance table as figure)
# ─────────────────────────────────────────────────────────────────────────────
def fig_benchmark_results():
    systems    = ["1. Rules/Regex\nBaseline", "2. LLM-only\n(Claude Haiku)",
                  "3. Memory + LLM", "4. Full System\n(Ours)"]
    precision  = [0.9851, 0.9754, 0.9915, 0.9833]
    recall     = [0.5500, 0.9917, 0.9750, 0.9833]
    f1         = [0.7059, 0.9835, 0.9832, 0.9833]

    x = np.arange(len(systems))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.bar(x - width,   precision, width, color="#2980B9", alpha=0.85, label="Precision")
    ax.bar(x,           recall,    width, color="#E67E22", alpha=0.85, label="Recall")
    ax.bar(x + width,   f1,        width, color="#27AE60", alpha=0.85, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_title("4-System Benchmark — Precision, Recall, F1\n"
                 "(200 human-curated samples, cold-start evaluation)",
                 fontsize=11, pad=12)
    ax.legend(fontsize=9)

    # annotate recall jump
    ax.annotate("", xy=(1, recall[1]), xytext=(0, recall[0]),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    ax.text(0.5, 0.72, "+44.2% recall\nRules → LLM",
            ha="center", fontsize=7.5, color="#555")

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width,   p + 0.005, f"{p:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i,           r + 0.005, f"{r:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + width,   f + 0.005, f"{f:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_benchmark.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


if __name__ == "__main__":
    print("Generating paper figures …")
    fig_evasion_comparison()
    fig_per_form_codebert()
    fig_distance_distribution()
    fig_benchmark_results()
    print("Done. All figures saved to figures/")
