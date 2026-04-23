# figures/tsne_semantic_gap.py
#
# The "money shot" — visualises the semantic gap between natural language
# and code-form attack variants in two embedding spaces:
#
#   MiniLM  → "Geography of Syntax": code forms cluster by language (bash
#              together, Python together) regardless of threat intent.
#              Natural-language descriptions form their own island.
#
#   CodeBERT → "Geography of Intent": NL and code variants for the same
#               threat cluster together. Syntax dissolves; semantics survives.
#
# Two colour dimensions are used simultaneously:
#   Colour  → form type (NL black, bash crimson, python steelblue, ...)
#   Marker  → MITRE ATT&CK category (recon=^, credential=s, persistence=D,
#              exfiltration=v, defense_evasion=P)
#
# Connection lines are drawn for 5 representative threats (one per ATT&CK
# category), connecting the NL point to each of its code-form variants.
# These lines show the gap (MiniLM) or collapse (CodeBERT).
#
# Output:
#   tsne_semantic_gap_minilm.png      — MiniLM only (dpi=200)
#   tsne_semantic_gap_codebert.png    — CodeBERT only (dpi=200)
#   tsne_semantic_gap_comparison.png  — side-by-side (dpi=200, paper figure)
#
# Runtime: ~5 min on CPU (CodeBERT encoding is the bottleneck).
# Cost: $0. No API calls.

import os
import sys
import numpy as np

# Allow importing from the figures/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch

from code_form_analysis import (
    THREATS,
    CODE_FORMS,
    embed_minilm,
    embed_codebert,
    EVASION_THRESHOLD,
)

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Visual encoding
# ---------------------------------------------------------------------------

FORM_COLOURS = {
    "nl":         "#111111",   # black
    "bash":       "#cc0000",   # crimson
    "python":     "#3a78b5",   # steelblue
    "curl":       "#2a8e44",   # forestgreen
    "powershell": "#e07b00",   # darkorange
    "sql":        "#7b52ab",   # mediumpurple  (also covers JavaScript)
}

FORM_SIZES = {
    "nl": 90, "bash": 40, "python": 40,
    "curl": 40, "powershell": 40, "sql": 40,
}

CATEGORY_MARKERS = {
    "recon":            "^",   # triangle up
    "credential_access":"s",   # square
    "persistence":      "D",   # diamond
    "exfiltration":     "v",   # triangle down
    "defense_evasion":  "P",   # plus filled
}

# One representative threat per ATT&CK category for connection lines
HIGHLIGHT_THREATS = {
    "recon":            "T12",   # Scan internal network
    "credential_access":"T01",   # Read env secrets
    "persistence":      "T13",   # Inject into cron job
    "exfiltration":     "T39",   # Upload to cloud storage
    "defense_evasion":  "T45",   # Clear bash history
}

# ---------------------------------------------------------------------------
# Build flat text list and metadata
# ---------------------------------------------------------------------------

def build_texts_and_meta():
    """
    Returns:
        texts   list[str]  — all texts in order: NL then 5 forms per threat
        meta    list[dict] — {form, category, threat_id, form5_lang}
    """
    texts, meta = [], []
    for t in THREATS:
        texts.append(t["natural_language"])
        meta.append({"form": "nl", "category": t["category"],
                     "threat_id": t["id"], "form5_lang": t["form5_lang"]})
        for form in CODE_FORMS:
            texts.append(t[form])
            # For sql slot, record the actual language for labelling
            actual = t["form5_lang"] if form == "sql" else form
            meta.append({"form": form, "category": t["category"],
                         "threat_id": t["id"], "form5_lang": actual,
                         "display_form": actual if form == "sql" else form})
    return texts, meta


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def run_tsne(vecs: np.ndarray, perplexity: int = 30) -> np.ndarray:
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                n_iter=1000, learning_rate="auto", init="pca")
    return tsne.fit_transform(vecs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_ax(ax, coords: np.ndarray, meta: list, model_label: str,
            subtitle: str, highlight_ids: dict):
    """Draw one t-SNE panel onto ax."""
    N = len(THREATS)
    # Points: each threat has 1 NL + 5 code-form points = 6 rows
    # They are laid out sequentially in coords/meta

    # --- Connection lines for highlighted threats ---
    for cat, tid in highlight_ids.items():
        # Find NL index for this threat
        nl_idx = next(i for i, m in enumerate(meta) if m["threat_id"] == tid and m["form"] == "nl")
        code_indices = [i for i, m in enumerate(meta) if m["threat_id"] == tid and m["form"] != "nl"]
        nl_xy = coords[nl_idx]
        for ci in code_indices:
            code_xy = coords[ci]
            ax.plot([nl_xy[0], code_xy[0]], [nl_xy[1], code_xy[1]],
                    color="#bbbbbb", linewidth=0.6, zorder=1, alpha=0.7)

    # --- Scatter points ---
    for i, (xy, m) in enumerate(zip(coords, meta)):
        form = m["form"]
        cat  = m["category"]
        colour = FORM_COLOURS[form]
        size   = FORM_SIZES[form]
        marker = CATEGORY_MARKERS.get(cat, "o")
        edgecolor = "#ffffff" if form == "nl" else "none"
        lw = 0.5 if form == "nl" else 0
        ax.scatter(xy[0], xy[1], c=colour, s=size, marker=marker,
                   edgecolors=edgecolor, linewidths=lw, zorder=3, alpha=0.85)

    # --- Labels for highlighted NL points ---
    for cat, tid in highlight_ids.items():
        nl_idx = next(i for i, m in enumerate(meta) if m["threat_id"] == tid and m["form"] == "nl")
        threat_name = next(t["name"] for t in THREATS if t["id"] == tid)
        xy = coords[nl_idx]
        ax.annotate(threat_name, xy, fontsize=6.5, fontweight="bold",
                    xytext=(4, 4), textcoords="offset points",
                    color="#111111",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"))

    ax.set_title(f"{model_label}\n{subtitle}", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15)


def build_legend_handles():
    """Form colour legend + MITRE category marker legend."""
    form_handles = [
        mlines.Line2D([], [], color=FORM_COLOURS["nl"],         marker="o", linestyle="None",
                      markersize=8, label="Natural Language (NL)"),
        mlines.Line2D([], [], color=FORM_COLOURS["bash"],       marker="o", linestyle="None",
                      markersize=6, label="Bash"),
        mlines.Line2D([], [], color=FORM_COLOURS["python"],     marker="o", linestyle="None",
                      markersize=6, label="Python"),
        mlines.Line2D([], [], color=FORM_COLOURS["curl"],       marker="o", linestyle="None",
                      markersize=6, label="curl"),
        mlines.Line2D([], [], color=FORM_COLOURS["powershell"], marker="o", linestyle="None",
                      markersize=6, label="PowerShell"),
        mlines.Line2D([], [], color=FORM_COLOURS["sql"],        marker="o", linestyle="None",
                      markersize=6, label="SQL / JavaScript"),
    ]
    cat_handles = [
        mlines.Line2D([], [], color="#555555", marker="^", linestyle="None",
                      markersize=7, label="Recon"),
        mlines.Line2D([], [], color="#555555", marker="s", linestyle="None",
                      markersize=7, label="Credential Access"),
        mlines.Line2D([], [], color="#555555", marker="D", linestyle="None",
                      markersize=7, label="Persistence"),
        mlines.Line2D([], [], color="#555555", marker="v", linestyle="None",
                      markersize=7, label="Exfiltration"),
        mlines.Line2D([], [], color="#555555", marker="P", linestyle="None",
                      markersize=7, label="Defense Evasion"),
    ]
    line_handle = mlines.Line2D([], [], color="#bbbbbb", linewidth=1.2,
                                label="NL ↔ code variant (selected threats)")
    return form_handles, cat_handles, line_handle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building text corpus …")
    texts, meta = build_texts_and_meta()
    print(f"  {len(texts)} texts  ({len(THREATS)} threats × 6 forms each)")

    print("\nEmbedding with all-MiniLM-L6-v2 …", flush=True)
    vecs_minilm = embed_minilm(texts)
    print("\nEmbedding with codebert-base …", flush=True)
    vecs_codebert = embed_codebert(texts)

    print("\nRunning t-SNE for MiniLM …", flush=True)
    coords_minilm = run_tsne(vecs_minilm)
    print("Running t-SNE for CodeBERT …", flush=True)
    coords_codebert = run_tsne(vecs_codebert)

    form_handles, cat_handles, line_handle = build_legend_handles()

    # ── Individual plots ──────────────────────────────────────────────────────
    for coords, model_label, subtitle, fname in [
        (coords_minilm,
         "MiniLM — Geography of Syntax",
         "Code forms cluster by language, not intent.\nNL descriptions occupy a separate island.",
         "tsne_semantic_gap_minilm.png"),
        (coords_codebert,
         "CodeBERT — Geography of Intent",
         "NL and code variants for the same threat cluster together.\nSyntax dissolves; semantics survives.",
         "tsne_semantic_gap_codebert.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 8))
        draw_ax(ax, coords, meta, model_label, subtitle, HIGHLIGHT_THREATS)

        # Legends in two columns
        leg1 = ax.legend(handles=form_handles + [line_handle],
                         title="Code Form", title_fontsize=8,
                         fontsize=7.5, loc="upper left",
                         framealpha=0.8, borderpad=0.6)
        ax.add_artist(leg1)
        ax.legend(handles=cat_handles, title="ATT&CK Category",
                  title_fontsize=8, fontsize=7.5, loc="lower left",
                  framealpha=0.8, borderpad=0.6)

        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, fname)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out}")

    # ── Side-by-side comparison (paper figure) ────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    draw_ax(ax1, coords_minilm, meta,
            "all-MiniLM-L6-v2 — Geography of Syntax",
            "Code forms cluster by language, not intent.\n"
            "Gray lines show NL ↔ code-variant distances.",
            HIGHLIGHT_THREATS)
    draw_ax(ax2, coords_codebert, meta,
            "CodeBERT — Geography of Intent",
            "NL and code variants for the same threat cluster together.\n"
            "The semantic gap collapses.",
            HIGHLIGHT_THREATS)

    # Shared legend below both axes
    all_handles = form_handles + [line_handle] + cat_handles
    fig.legend(handles=all_handles, ncol=6, loc="lower center",
               fontsize=8, framealpha=0.9, borderpad=0.6,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Semantic Gap in Retrieval-Based Agent Safety Memory\n"
        f"{len(THREATS)} MITRE ATT&CK threats × 5 code forms — "
        f"Evasion threshold: cosine dist > {EVASION_THRESHOLD}",
        fontsize=13, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "tsne_semantic_gap_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out}")

    print("\nDone. All t-SNE figures written to figures/")


if __name__ == "__main__":
    main()
