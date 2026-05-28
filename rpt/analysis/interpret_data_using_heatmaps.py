import json
import re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import argparse

from rpt.analysis.paths import (
    log_path_for_analysis_dir,
    resolve_analysis_dir,
    resolve_log_path,
    task_model_and_stem_from_analysis_dir,
)

DEFAULT_PAPER_ALIGNMENT_RUNS = [
    {
        "title": "HotpotQA",
        "dir_path": "hotpotqa/gpt-5/log_history_summary_iters_40_k_topics_10_clustering_sample_size_100_cluster_desc_comprehensive_optimizer_non_minimal_pcw_0.0025_pcu_1000",
        "log_path": "logs/hotpotqa/openai/log_history_summary_iters_40_k_topics_10_clustering_sample_size_100_cluster_desc_comprehensive_optimizer_non_minimal_pcw_0.0025_pcu_1000.jsonl",
    },
    {
        "title": "LiveBench-Math",
        "dir_path": "livebench_math/gpt-5/log_history_summary_iters_40_train_123_val_123_test_122_split_seed_0_seed_0_k_topics_10_cluster_desc_comprehensive_optimizer_non_minimal_pcw_0.0025_pcu_1000",
        "log_path": "logs/livebench_math/openai/log_history_summary_iters_40_train_123_val_123_test_122_split_seed_0_seed_0_k_topics_10_cluster_desc_comprehensive_optimizer_non_minimal_pcw_0.0025_pcu_1000.jsonl",
    },
    {
        "title": "Formula",
        "dir_path": "formula/gpt-5/log_history_summary_iters_40_train_500_dev_300_test_200_seed_0_k_topics_20_cluster_desc_comprehensive_optimizer_non_minimal_pcw_0.0025_pcu_1000",
        "log_path": "logs/xbrl_formula/openai/log_history_summary_iters_40_train_500_dev_300_test_200_seed_0_k_topics_20_cluster_desc_comprehensive_optimizer_non_minimal_pcw_0.0025_pcu_1000.jsonl",
    },
]

PAPER_TOPIC_LABEL_OVERRIDES = {
    # HotpotQA failure topics
    "Context Integration and Evidence Use": "Context + Evidence",
    "Multi-hop and Relational Reasoning Failures": "Multi-hop Reasoning",
    "Salience, Proximity, and Heuristic Biases": "Salience Biases",
    "Misinterpretation of Question Cues and Constraints": "Question Constraints",
    "Justification and Answer Consistency": "Answer Consistency",
    "Ambiguity, Alternatives, and Disambiguation Failures": "Disambiguation",
    "Span Extraction and Surface Form Errors": "Span/Surface Errors",
    "Granularity and Specificity Mismatches": "Granularity",
    "Answer Type and Entity Type Mismatches": "Answer/Entity Type",
    "Overconfidence and Calibration Failures": "Overconfidence",
    # HotpotQA patch topics
    "Span Extraction and Minimality": "Span Minimality",
    "Canonical vs. Surface Form Preference": "Canonical Form",
    "Qualifier and Disambiguation Handling": "Qualifiers",
    "Administrative and Legal Designators": "Legal Designators",
    "Answer-Type and Granularity Matching": "Answer Granularity",
    "Handling of Nationality, Country, and Demonyms": "Nationality/Demonyms",
    "Temporal and Numeric Answer Extraction": "Temporal/Numeric",
    "Relation and Multi-hop Query Handling": "Multi-hop Queries",
    "Checklist and Pre-answer Verification": "Pre-answer Checks",
    "Confidence Calibration and Justification": "Calibration",
    # LiveBench Math failure topics
    "Arithmetic and Algebraic Computation Errors": "Arithmetic/Algebra",
    "Geometric and Structural Reasoning Errors": "Geometry/Structure",
    "Misapplication of Mathematical Definitions, Theorems, and Conventions": "Definitions/Theorems",
    "Factorization and Divisibility Confusion": "Factorization",
    "Logical Flow and Structural Misinterpretation": "Logical Flow",
    "Combinatorial and Enumeration Errors": "Combinatorics",
    "Contextual and Semantic Misalignment": "Semantic Alignment",
    "Type, Symbol, and Notation Confusion": "Notation/Symbols",
    "Overconfidence and Verification Failures": "Verification Failure",
    "Speculative Guessing and Incomplete Solution Strategies": "Incomplete Strategy",
    # LiveBench Math patch topics
    "Step-by-Step Solution Protocols": "Step-by-step",
    "Verification and Truthfulness Guards": "Verification Guards",
    "Confidence Calibration and Capping": "Calibration Caps",
    "Mapping, Slot-Filling, and Identifier Invariants": "Identifier Invariants",
    "Arithmetic and Algebraic Reliability Checks": "Algebra Checks",
    "Geometry and Structural Verification": "Geometry Checks",
    "Quantifier and Optimization Audits": "Optimization Audits",
    "Formatting and Output Validation": "Output Validation",
    "Domain-Specific Audits and Guards": "Domain Audits",
    "Narrative and Logical Flow Controls": "Flow Controls",
    # Formula failure topics
    "Exchange Rate and PPP Formula/Direction Errors": "Exchange Rate",
    "Compounding and Discounting Convention Errors": "Compounding",
    "Arithmetic and Numerical Calculation Errors": "Arithmetic",
    "Ignoring or Misusing Provided Inputs": "Provided Inputs",
    "Metric Definition and Formula Selection Errors": "Formula Selection",
    "Cash Flow Timing and Annuity Convention Errors": "Cash-flow Timing",
    "Ambiguity Handling and Overconfidence": "Ambiguity/Confidence",
    "Unit and Scale Conversion Errors": "Unit/Scale",
    "Unjustified or Fabricated Assumptions": "Fabricated Assumptions",
    "Sanity Check and Domain Constraint Neglect": "Sanity Checks",
    # Formula patch topics
    "Unit, Scale, and Format Handling": "Units/Scale/Format",
    "Precision, Rounding, and Display Rules": "Precision/Rounding",
    "Independent Verification and Arithmetic Checks": "Arithmetic Checks",
    "Confidence Calibration and Reporting": "Confidence Reporting",
    "Compounding, Discounting, and Timing Cues": "Timing Cues",
    "Exponentiation and Power Checks": "Power Checks",
    "Black-Scholes and Option Pricing Pipeline": "Option Pricing",
    "Display-Only Safeguards and Reuse Prohibition": "Display Safeguards",
    "Reasoning and Output Field Requirements": "Output Requirements",
    "Special Handling for CDFs, Percent Changes, and Output Formats": "CDFs/Percent Changes",
}

# -------------------------
# Helpers: load files
# -------------------------
def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def parse_transition_key(s):
    # "12->13" -> (12, 13)
    m = re.match(r"^\s*(\d+)\s*->\s*(\d+)\s*$", s)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# -------------------------
# 1) Build per-step metrics from log_all_reports_iters_50.jsonl
# -------------------------
def extract_metrics_from_stats_payload(stats_payload: dict) -> dict:
    """
    stats_payload is the *payload* of an event like train_stats/dev_stats/test_stats.
    In your log, the metrics live under payload["metrics"] with keys:
      task_score, brier, fmt_rate
    """
    if not isinstance(stats_payload, dict):
        return {}

    metrics = stats_payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    # Task-specific primary score:
    # - hotpotqa: exact_match
    # - xbrl_formula: accuracy
    # - livebench_math: task_score
    task_score = metrics.get("task_score", None)
    if task_score is None:
        task_score = metrics.get("accuracy", None)
    if task_score is None:
        task_score = metrics.get("exact_match", None)

    # Map log schema -> our canonical names
    out = {
        "n": metrics.get("n", None),
        "task_score": task_score,
        "brier": metrics.get("brier", None),
        "fmt_rate": metrics.get("format_error_rate", None),
        # "f1_avg": metrics.get("f1", None),
        # "avg_conf": metrics.get("avg_confidence", None),
    }

    # if everything missing, return {}
    if all(v is None for v in [out["task_score"], out["brier"]]):
        return {}
    return out


def parse_step_stats(events):
    """
    Returns:
      step_stats[step]["train_stats"] = {task_score, brier, ...}
      step_stats[step]["val_stats"] = {task_score, brier, ...}
      step_stats[step]["test_stats"] = {...}
    """
    step_stats = defaultdict(dict)

    for e in events:
        if "step" not in e:
            continue
        t = int(e["step"])
        ev = e.get("event")
        payload = e.get("payload", {}) or {}

        if ev == "train_stats":
            m = extract_metrics_from_stats_payload(payload)
            if m:
                step_stats[t]["train_stats"] = m

        if ev in {"val_stats", "dev_stats"}:
            m = extract_metrics_from_stats_payload(payload)
            if m:
                step_stats[t]["val_stats"] = m

        if ev in {"test_stats", "final_test_stats"}:
            m = extract_metrics_from_stats_payload(payload)
            if m:
                step_stats[t]["test_stats"] = m

    return step_stats


def select_metric_split(step_stats, preferred="train_stats"):
    if any(preferred in per_step for per_step in step_stats.values()):
        return preferred
    for fallback in ("val_stats", "test_stats"):
        if any(fallback in per_step for per_step in step_stats.values()):
            return fallback
    return preferred


def build_transition_metrics(step_stats, split="train_stats"):
    """
    One row per transition t->t+1, with deltas on the requested split.
    """
    steps = sorted(step_stats.keys())
    rows = []

    def pick(t):
        return step_stats.get(t, {}).get(split, {})

    for i in range(len(steps) - 1):
        t, t1 = steps[i], steps[i + 1]
        s0, s1 = pick(t), pick(t1)

        def d(k):
            a, b = s0.get(k), s1.get(k)
            if a is None or b is None:
                return None
            return float(b) - float(a)

        rows.append({
            "transition": f"{t}->{t1}",
            "t": t,
            "t1": t1,

            # absolute
            "task_score_t": s0.get("task_score"),
            "task_score_t1": s1.get("task_score"),
            "brier_t": s0.get("brier"),
            "brier_t1": s1.get("brier"),
            # "f1_t": s0.get("f1_avg"),
            # "f1_t1": s1.get("f1_avg"),
            # "avg_conf_t": s0.get("avg_conf"),
            # "avg_conf_t1": s1.get("avg_conf"),

            # deltas
            "d_task_score": d("task_score"),
            "d_brier": d("brier"),
            # "d_f1": d("f1_avg"),
            # "d_avg_conf": d("avg_conf"),
        })

    return pd.DataFrame(rows)

# -------------------------
# 2) Build transition-level error + patch topic presence/intensity
# -------------------------
def build_transition_topic_tables(failure_labeled, patch_labeled):
    """
    failure_labeled: list of {step, topic, ...}
    patch_labeled: list of {transition, topic, ...}
    Returns:
      fail_counts: df indexed by transition, columns=failure_topic, values=count
      patch_counts: df indexed by transition, columns=patch_topic, values=count
    """
    # Failure topics are at step t, which drives transition t->t+1
    fail_by_transition = defaultdict(list)
    for r in failure_labeled:
        t = int(r["step"])
        topic = int(r["topic"])
        if topic == -1:
            continue
        fail_by_transition[f"{t}->{t+1}"].append(topic)

    patch_by_transition = defaultdict(list)
    for r in patch_labeled:
        tr = r.get("transition")
        if not tr:
            continue
        topic = int(r["topic"])
        if topic == -1:
            continue
        patch_by_transition[tr].append(topic)

    transitions = sorted(set(fail_by_transition.keys()) | set(patch_by_transition.keys()),
                         key=lambda s: parse_transition_key(s) or (10**9, 10**9))

    # Build count matrices
    all_fail_topics = sorted({t for ts in fail_by_transition.values() for t in ts})
    all_patch_topics = sorted({t for ts in patch_by_transition.values() for t in ts})

    fail_mat = []
    patch_mat = []

    for tr in transitions:
        fc = Counter(fail_by_transition.get(tr, []))
        pc = Counter(patch_by_transition.get(tr, []))
        fail_mat.append([fc.get(t, 0) for t in all_fail_topics])
        patch_mat.append([pc.get(t, 0) for t in all_patch_topics])

    fail_counts = pd.DataFrame(fail_mat, index=transitions, columns=[f"F{t}" for t in all_fail_topics])
    patch_counts = pd.DataFrame(patch_mat, index=transitions, columns=[f"P{t}" for t in all_patch_topics])

    return fail_counts, patch_counts


# -------------------------
# 3) Pretty labels (optional)
# -------------------------
def load_topic_names(labels_json_path, prefix):
    """
    labels_json_path should be created by your LLM labeling script.
    Expected structure:
      {"topics": {"0": {"label": {"topic_name": ...}}, ...}}  OR  {"0": {"topic_name": ...}, ...}
    """
    if not labels_json_path or not Path(labels_json_path).exists():
        return {}

    data = load_json(labels_json_path)
    out = {}
    if "topics" in data:
        for tid, entry in data["topics"].items():
            label = entry.get("label", {})
            name = label.get("topic_name") or label.get("topic") or None
            if name:
                out[f"{prefix}{tid}"] = name
    else:
        for tid, entry in data.items():
            name = entry.get("topic_name") or entry.get("topic") or None
            if name:
                out[f"{prefix}{tid}"] = name
    return out

def load_topic_name_map(data, prefix):
    """
    Builds a mapping from 'F0'/'P0' style column names -> human-readable topic_name.

    Expects structure:
      {"topics": {"0": {"label": {"topic_name": ...}}, ...}}
    """
    topics = data.get("topics", [])
    out = {}

    for entry in topics:
        tid_str = str(entry.get("topic_id"))
        name = entry.get("name")
        if name:
            out[f"{prefix}{tid_str}"] = name

    return out

# -------------------------
# 4) Heatmap plotting
# -------------------------
# def plot_heatmap(values_df, denoms_df, title, xlabel, ylabel, out_path=None):
#     def _wrap(labels, width=22):
#         return ['\n'.join(textwrap.wrap(str(l), width=width)) for l in labels]

#     A = values_df.values.astype(float)
#     D = denoms_df.values

#     fig_w = max(13, 0.75 * A.shape[1] + 6)
#     fig_h = max(8, 0.45 * A.shape[0] + 4)
#     plt.figure(figsize=(fig_w, fig_h))
#     im = plt.imshow(A, aspect="auto")
#     plt.colorbar(im)

#     xlabels = list(values_df.columns)
#     xlabels = _wrap(xlabels, width=20)
#     ylabels = list(values_df.index)
#     ylabels = _wrap(ylabels, width=18)

#     plt.xticks(range(values_df.shape[1]), xlabels, rotation=35, ha="right")
#     plt.yticks(range(values_df.shape[0]), ylabels)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)

#     # annotate ONLY denominators
#     for i in range(A.shape[0]):
#         for j in range(A.shape[1]):
#             d = D[i, j]
#             if d is None or (isinstance(d, float) and np.isnan(d)):
#                 txt = ""
#             else:
#                 txt = str(int(d))
#             # choose readable text color based on background intensity
#             color = "white" if (not np.isnan(A[i, j]) and A[i, j] > np.nanmean(A)) else "black"
#             plt.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
#     if out_path:
#         plt.savefig(out_path, dpi=200)
#     plt.show()


def plot_heatmap(values_df, numer_df, denom_df=None, *, title="", xlabel="", ylabel="",
                            show="numer", out_path=None):
    """
    show:
      - "numer" => show numerator only
      - "denom" => show denominator only
      - "frac"  => show "numer/denom" (requires denom_df)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    A = values_df.values.astype(float)
    N = numer_df.values
    D = denom_df.values if denom_df is not None else None

    plt.figure(figsize=(max(10, 0.75*A.shape[1] + 6), max(7, 0.45*A.shape[0] + 4)))
    im = plt.imshow(A, aspect="auto")
    plt.colorbar(im)

    plt.xticks(range(values_df.shape[1]), values_df.columns, rotation=35, ha="right")
    plt.yticks(range(values_df.shape[0]), values_df.index)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    mean_val = np.nanmean(A) if np.isfinite(A).any() else 0.0

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if show == "numer":
                txt = "" if (N[i, j] is None or (isinstance(N[i, j], float) and np.isnan(N[i, j]))) else str(int(N[i, j]))
            elif show == "denom":
                txt = "" if (D is None or D[i, j] is None or (isinstance(D[i, j], float) and np.isnan(D[i, j]))) else str(int(D[i, j]))
            else:  # "frac"
                if D is None:
                    raise ValueError("show='frac' requires denom_df")
                txt = f"{int(N[i,j])}/{int(D[i,j])}"

            color = "white" if (np.isfinite(A[i, j]) and A[i, j] > mean_val) else "black"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

    plt.subplots_adjust(left=0.40, bottom=0.35, right=0.98, top=0.90)
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()

def plot_alignment_heatmap_fraction(
    align_df, numer_df, denom_df,
    title="Alignment heatmap: P(patch|failure)",
    xlabel="Patch topic", ylabel="Failure topic",
    out_path=None
):
    A = align_df.values.astype(float)
    N = numer_df.values
    D = denom_df.values

    plt.figure(figsize=(max(10, 0.75*A.shape[1] + 6), max(7, 0.45*A.shape[0] + 4)))
    im = plt.imshow(A, aspect="auto")
    plt.colorbar(im)

    plt.xticks(range(align_df.shape[1]), align_df.columns, rotation=35, ha="right")
    plt.yticks(range(align_df.shape[0]), align_df.index)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    mean_val = np.nanmean(A) if np.isfinite(A).any() else 0.0

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            n = int(N[i, j]) if np.isfinite(N[i, j]) else 0
            d = int(D[i, j]) if np.isfinite(D[i, j]) else 0
            txt = f"{n}/{d}" if d > 0 else "0/0"

            color = "white" if (np.isfinite(A[i, j]) and A[i, j] > mean_val) else "black"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

    plt.subplots_adjust(left=0.40, bottom=0.35, right=0.98, top=0.90)

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def _records_with_reassigned_topics(labeled_records, clusterfusion_payload, *, record_kind, base):
    assignments = clusterfusion_payload.get("assignments", [])
    if len(labeled_records) != len(assignments):
        raise ValueError(
            f"{base}: {record_kind} record count ({len(labeled_records)}) does not match "
            f"cluster-fusion assignment count ({len(assignments)})."
        )

    reassigned = []
    for record, topic in zip(labeled_records, assignments):
        row = dict(record)
        row["topic"] = int(topic)
        reassigned.append(row)
    return reassigned


def build_alignment_fraction_for_analysis_dir(analysis_dir):
    """
    Build P(patch topic present | failure topic present) for one clustering run.

    Returns an align_df whose rows and columns are human-readable topic names.
    """
    base = Path(analysis_dir)
    failure_labeled = load_json(base / "failure_labeled.json")
    failure_labeled_new = load_json(base / "failures_clusterfusion_cosine.json")
    patch_labeled = load_json(base / "patch_labeled.json")
    patch_labeled_new = load_json(base / "patches_clusterfusion_cosine.json")

    failure_labeled = _records_with_reassigned_topics(
        failure_labeled,
        failure_labeled_new,
        record_kind="failure",
        base=base,
    )
    patch_labeled = _records_with_reassigned_topics(
        patch_labeled,
        patch_labeled_new,
        record_kind="patch",
        base=base,
    )

    fail_counts, patch_counts = build_transition_topic_tables(failure_labeled, patch_labeled)
    fail_present = (fail_counts > 0).astype(int)
    patch_present = (patch_counts > 0).astype(int)

    align = []
    for fcol in fail_present.columns:
        idx = fail_present[fcol] == 1
        if idx.sum() == 0:
            align.append([np.nan] * patch_present.shape[1])
        else:
            align.append(list(patch_present[idx].mean(axis=0).values))

    align_df = pd.DataFrame(align, index=fail_present.columns, columns=patch_present.columns)
    failure_name_map = load_topic_name_map(failure_labeled_new, prefix="F")
    patch_name_map = load_topic_name_map(patch_labeled_new, prefix="P")
    return align_df.rename(index=failure_name_map, columns=patch_name_map)


def _wrap_plot_labels(labels, width, label_overrides=None):
    label_overrides = label_overrides or {}
    wrapped = []
    for label in labels:
        display_label = str(label_overrides.get(str(label), str(label))).strip()
        lines = []
        for part in display_label.splitlines():
            part_lines = textwrap.wrap(
                part,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            lines.extend(part_lines if part_lines else [part])
        wrapped.append("\n".join(lines) if lines else str(display_label))
    return wrapped


def plot_paper_alignment_heatmaps(
    run_configs=None,
    *,
    clustering_root="clustering_results",
    out_prefix="vis_results/fig_alignment_all_acl_v1",
    figsize=(9.6, 3.6),
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
    x_label_wrap=28,
    y_label_wrap=28,
    tick_fontsize=6.8,
    title_fontsize=9.5,
    axis_label_fontsize=10.0,
    label_overrides=None,
):
    """
    Create the ACL-style main alignment figure:
      - HotpotQA, LiveBench Math, and Formula in one row by default
      - rows are patch topics and columns are failure topics
      - full topic-name tick labels, manually wrapped
      - no cell text
      - one shared colorbar and shared axis names

    Returns the saved PNG and PDF paths.
    """
    if run_configs is None:
        run_configs = DEFAULT_PAPER_ALIGNMENT_RUNS
    if label_overrides is None:
        label_overrides = PAPER_TOPIC_LABEL_OVERRIDES

    panels = []
    for config in run_configs:
        base = resolve_analysis_dir(config["dir_path"], root=clustering_root)
        if not base.exists():
            raise FileNotFoundError(f"Could not find clustering results directory: {base}")
        panels.append({
            "title": config["title"],
            "align_df": build_alignment_fraction_for_analysis_dir(base).fillna(0.0),
        })

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_prefix.with_suffix(".png")
    out_pdf = out_prefix.with_suffix(".pdf")

    rc = {
        "font.family": "serif",
        "font.size": tick_fontsize,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=figsize,
            constrained_layout=False,
        )
        if len(panels) == 1:
            axes = [axes]

        image = None
        for ax, panel in zip(axes, panels):
            align_df = panel["align_df"].T
            image = ax.imshow(
                align_df.values.astype(float),
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            ax.set_title(panel["title"], fontsize=title_fontsize, pad=5)
            ax.set_xticks(np.arange(align_df.shape[1]))
            ax.set_yticks(np.arange(align_df.shape[0]))
            ax.set_xticklabels(
                _wrap_plot_labels(align_df.columns, x_label_wrap, label_overrides),
                rotation=65,
                ha="right",
                va="top",
                rotation_mode="anchor",
                fontsize=tick_fontsize - 0.2,
            )
            ax.set_yticklabels(
                _wrap_plot_labels(align_df.index, y_label_wrap, label_overrides),
                fontsize=tick_fontsize,
            )
            ax.tick_params(axis="both", which="major", length=0, pad=0.8)

            ax.set_xticks(np.arange(-0.5, align_df.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, align_df.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.35)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        fig.subplots_adjust(left=0.105, right=0.915, bottom=0.30, top=0.86, wspace=0.62)
        fig.canvas.draw()
        leftmost = axes[0].get_position()
        rightmost = axes[-1].get_position()
        heatmap_x_center = (leftmost.x0 + rightmost.x1) / 2
        heatmap_y_center = (leftmost.y0 + leftmost.y1) / 2
        fig.text(
            heatmap_x_center,
            -0.010,
            "Failure topic",
            ha="center",
            va="center",
            fontsize=axis_label_fontsize,
        )
        fig.text(
            -0.030,
            heatmap_y_center + 0.018,
            "Patch topic",
            ha="center",
            va="center",
            rotation=90,
            fontsize=axis_label_fontsize,
        )

        if image is not None:
            cax = fig.add_axes([
                rightmost.x1 + 0.012,
                rightmost.y0,
                0.014,
                rightmost.height,
            ])
            cbar = fig.colorbar(image, cax=cax)
            cbar.set_label("P(patch | failure)", fontsize=axis_label_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize, length=2)

        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

    return out_png, out_pdf


def build_patch_delta_effect_for_analysis_dir(
    analysis_dir,
    events_path,
    *,
    metric_split_preferred="train_stats",
):
    """
    Build mean metric deltas for transitions where each patch topic is present.

    This mirrors the single-run heatmap_patch_to_deltas_v1.png calculation.
    """
    base = Path(analysis_dir)
    failure_labeled = load_json(base / "failure_labeled.json")
    failure_labeled_new = load_json(base / "failures_clusterfusion_cosine.json")
    patch_labeled = load_json(base / "patch_labeled.json")
    patch_labeled_new = load_json(base / "patches_clusterfusion_cosine.json")

    failure_labeled = _records_with_reassigned_topics(
        failure_labeled,
        failure_labeled_new,
        record_kind="failure",
        base=base,
    )
    patch_labeled = _records_with_reassigned_topics(
        patch_labeled,
        patch_labeled_new,
        record_kind="patch",
        base=base,
    )

    events = load_jsonl(events_path)
    step_stats = parse_step_stats(events)
    metric_split = select_metric_split(step_stats, preferred=metric_split_preferred)
    trans_metrics = build_transition_metrics(step_stats, split=metric_split).set_index("transition")

    fail_counts, patch_counts = build_transition_topic_tables(failure_labeled, patch_labeled)
    common = sorted(
        set(trans_metrics.index) & set(fail_counts.index) & set(patch_counts.index),
        key=lambda s: parse_transition_key(s) or (10**9, 10**9),
    )
    if not common:
        raise ValueError(
            f"{base}: no overlapping transitions found between metrics, failure topics, and patch topics."
        )

    trans_metrics = trans_metrics.loc[common]
    patch_counts = patch_counts.loc[common]
    patch_counts = patch_counts.rename(columns=load_topic_name_map(patch_labeled_new, prefix="P"))
    patch_present = (patch_counts > 0).astype(int)

    delta_cols = ["d_task_score", "d_brier"]
    effect = []
    for pcol in patch_present.columns:
        idx = patch_present[pcol] == 1
        if idx.sum() == 0:
            effect.append([np.nan] * len(delta_cols))
        else:
            effect.append([trans_metrics.loc[idx, c].astype(float).mean() for c in delta_cols])

    return pd.DataFrame(
        effect,
        index=patch_present.columns,
        columns=["Δ score", "Δ Brier"],
    )


def plot_paper_patch_delta_heatmaps(
    run_configs=None,
    *,
    clustering_root="clustering_results",
    logs_root="logs",
    out_prefix="vis_results/fig_patch_to_deltas_all_acl_v1",
    figsize=(8.4, 2.55),
    cmap="RdBu_r",
    y_label_wrap=24,
    panel_width_scale=0.70,
    tick_fontsize=6.0,
    title_fontsize=8.8,
    axis_label_fontsize=9.0,
    label_overrides=None,
):
    """
    Create a compact paper figure for patch-topic -> metric-delta heatmaps.
    """
    import matplotlib.colors as mcolors

    if run_configs is None:
        run_configs = DEFAULT_PAPER_ALIGNMENT_RUNS
    if label_overrides is None:
        label_overrides = PAPER_TOPIC_LABEL_OVERRIDES

    panels = []
    for config in run_configs:
        base = resolve_analysis_dir(config["dir_path"], root=clustering_root)
        if not base.exists():
            raise FileNotFoundError(f"Could not find clustering results directory: {base}")

        if config.get("log_path"):
            events_path = Path(config["log_path"])
        else:
            task_name, model_name, _ = task_model_and_stem_from_analysis_dir(base, root=clustering_root)
            events_path = log_path_for_analysis_dir(
                base,
                task_name=task_name,
                model_name=model_name,
                logs_root=logs_root,
                clustering_root=clustering_root,
            )
        if not events_path.exists():
            raise FileNotFoundError(f"Could not find source log for {config['title']}: {events_path}")

        panels.append({
            "title": config["title"],
            "effect_df": build_patch_delta_effect_for_analysis_dir(base, events_path).fillna(0.0),
        })

    all_values = np.concatenate([panel["effect_df"].values.ravel() for panel in panels])
    finite_values = all_values[np.isfinite(all_values)]
    max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
    if max_abs == 0.0:
        max_abs = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_prefix.with_suffix(".png")
    out_pdf = out_prefix.with_suffix(".pdf")

    rc = {
        "font.family": "serif",
        "font.size": tick_fontsize,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=figsize,
            constrained_layout=False,
        )
        if len(panels) == 1:
            axes = [axes]

        image = None
        for ax, panel in zip(axes, panels):
            effect_df = panel["effect_df"]
            image = ax.imshow(
                effect_df.values.astype(float),
                aspect="auto",
                interpolation="nearest",
                norm=norm,
                cmap=cmap,
            )
            ax.set_title(panel["title"], fontsize=title_fontsize, pad=4)
            ax.set_xticks(np.arange(effect_df.shape[1]))
            ax.set_yticks(np.arange(effect_df.shape[0]))
            ax.set_xticklabels(effect_df.columns, fontsize=tick_fontsize + 1.0)
            ax.set_yticklabels(
                _wrap_plot_labels(effect_df.index, y_label_wrap, label_overrides),
                fontsize=tick_fontsize,
            )
            ax.tick_params(axis="both", which="major", length=0, pad=1.2)

            ax.set_xticks(np.arange(-0.5, effect_df.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, effect_df.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.35)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        fig.subplots_adjust(left=0.13, right=0.90, bottom=0.18, top=0.81, wspace=0.45)
        for ax in axes:
            pos = ax.get_position()
            new_width = pos.width * panel_width_scale
            ax.set_position([
                pos.x0 + (pos.width - new_width) / 2,
                pos.y0,
                new_width,
                pos.height,
            ])
        fig.canvas.draw()
        leftmost = axes[0].get_position()
        rightmost = axes[-1].get_position()
        heatmap_x_center = (leftmost.x0 + rightmost.x1) / 2
        heatmap_y_center = (leftmost.y0 + leftmost.y1) / 2
        fig.text(
            heatmap_x_center,
            0.055,
            "Δ metric",
            ha="center",
            va="center",
            fontsize=axis_label_fontsize,
        )
        fig.text(
            0.018,
            heatmap_y_center,
            "Patch topic",
            ha="center",
            va="center",
            rotation=90,
            fontsize=axis_label_fontsize,
        )

        if image is not None:
            cax = fig.add_axes([
                rightmost.x1 + 0.012,
                rightmost.y0,
                0.014,
                rightmost.height,
            ])
            cbar = fig.colorbar(image, cax=cax)
            cbar.set_label("Mean Δ metric", fontsize=axis_label_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize, length=2)

        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

    return out_png, out_pdf


def build_failure_delta_effect_for_analysis_dir(
    analysis_dir,
    events_path,
    *,
    metric_split_preferred="train_stats",
):
    """
    Build mean metric deltas for transitions where each failure topic is present.

    This mirrors the single-run heatmap_failure_to_deltas_v1.png calculation.
    """
    base = Path(analysis_dir)
    failure_labeled = load_json(base / "failure_labeled.json")
    failure_labeled_new = load_json(base / "failures_clusterfusion_cosine.json")
    patch_labeled = load_json(base / "patch_labeled.json")
    patch_labeled_new = load_json(base / "patches_clusterfusion_cosine.json")

    failure_labeled = _records_with_reassigned_topics(
        failure_labeled,
        failure_labeled_new,
        record_kind="failure",
        base=base,
    )
    patch_labeled = _records_with_reassigned_topics(
        patch_labeled,
        patch_labeled_new,
        record_kind="patch",
        base=base,
    )

    events = load_jsonl(events_path)
    step_stats = parse_step_stats(events)
    metric_split = select_metric_split(step_stats, preferred=metric_split_preferred)
    trans_metrics = build_transition_metrics(step_stats, split=metric_split).set_index("transition")

    fail_counts, patch_counts = build_transition_topic_tables(failure_labeled, patch_labeled)
    common = sorted(
        set(trans_metrics.index) & set(fail_counts.index) & set(patch_counts.index),
        key=lambda s: parse_transition_key(s) or (10**9, 10**9),
    )
    if not common:
        raise ValueError(
            f"{base}: no overlapping transitions found between metrics, failure topics, and patch topics."
        )

    trans_metrics = trans_metrics.loc[common]
    fail_counts = fail_counts.loc[common]
    fail_counts = fail_counts.rename(columns=load_topic_name_map(failure_labeled_new, prefix="F"))
    fail_present = (fail_counts > 0).astype(int)

    delta_cols = ["d_task_score", "d_brier"]
    effect = []
    for fcol in fail_present.columns:
        idx = fail_present[fcol] == 1
        if idx.sum() == 0:
            effect.append([np.nan] * len(delta_cols))
        else:
            effect.append([trans_metrics.loc[idx, c].astype(float).mean() for c in delta_cols])

    return pd.DataFrame(
        effect,
        index=fail_present.columns,
        columns=["Δ score", "Δ Brier"],
    )


# -------------------------
# Prompt-length analysis
# -------------------------
PROMPT_TEXT_KEYS = (
    "system",
    "developer",
    "instruction",
    "user",
    "prompt",
    "prefix",
    "suffix",
)

PROMPT_LENGTH_COLOR = "#0f766e"
DEV_PERFORMANCE_COLOR = "#4f46e5"


def _canonical_metric_split(split):
    if split in {"dev", "dev_stats", "validation", "validation_stats"}:
        return "val_stats"
    if split in {"val", "val_stats"}:
        return "val_stats"
    if split in {"train", "train_stats"}:
        return "train_stats"
    if split in {"test", "test_stats"}:
        return "test_stats"
    return split


def _extract_prompt_program(payload):
    if not isinstance(payload, dict):
        return None
    prompt_program = payload.get("prompt_program")
    if isinstance(prompt_program, (dict, str, list)):
        return prompt_program
    if any(key in payload for key in PROMPT_TEXT_KEYS):
        return payload
    messages = payload.get("messages")
    if isinstance(messages, list):
        return messages
    return None


def _prompt_program_to_text(prompt_program):
    """
    Convert the prompt program into the text-bearing prompt content.

    Non-text control flags, such as enforce_json_only=True, are intentionally
    ignored because they are not prompt tokens sent as natural language.
    """
    if isinstance(prompt_program, str):
        return prompt_program.strip()

    if isinstance(prompt_program, list):
        parts = []
        for item in prompt_program:
            item_text = _prompt_program_to_text(item)
            if item_text:
                parts.append(item_text)
        return "\n\n".join(parts).strip()

    if not isinstance(prompt_program, dict):
        return ""

    parts = []
    ordered_keys = [
        key for key in PROMPT_TEXT_KEYS if key in prompt_program
    ] + [
        key
        for key in sorted(prompt_program)
        if key not in PROMPT_TEXT_KEYS and key not in {"enforce_json_only"}
    ]
    for key in ordered_keys:
        value = prompt_program.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"[{key}]\n{value.strip()}")
        elif isinstance(value, (dict, list)):
            nested = _prompt_program_to_text(value)
            if nested:
                parts.append(f"[{key}]\n{nested}")
    return "\n\n".join(parts).strip()


def _get_token_counter(model_name=None, encoding_name=None):
    try:
        import tiktoken
    except ImportError:
        return "regex_token_fallback", lambda text: len(re.findall(r"\w+|[^\w\s]", text))

    if encoding_name:
        try:
            enc = tiktoken.get_encoding(encoding_name)
            return encoding_name, lambda text: len(enc.encode(text))
        except Exception:
            pass

    if model_name:
        try:
            enc = tiktoken.encoding_for_model(model_name)
            return f"tiktoken:{model_name}", lambda text: len(enc.encode(text))
        except Exception:
            pass

    for fallback_name in ("o200k_base", "cl100k_base"):
        try:
            enc = tiktoken.get_encoding(fallback_name)
            return fallback_name, lambda text: len(enc.encode(text))
        except Exception:
            continue

    return "regex_token_fallback", lambda text: len(re.findall(r"\w+|[^\w\s]", text))


def build_prompt_length_per_step(
    events,
    *,
    model_name=None,
    encoding_name=None,
    metric_split_preferred="val_stats",
    performance_metric="task_score",
):
    """
    Return one row per step with prompt token/character length and dev metric.

    The metric split defaults to val/dev because the analysis question is about
    dev performance. Tokens are computed with tiktoken when available, with a
    tokenizer-independent character count always included for auditability.
    """
    prompts_by_step = {}
    for event in events:
        if "step" not in event:
            continue
        try:
            step = int(event["step"])
        except (TypeError, ValueError):
            continue
        payload = event.get("payload", {}) or {}
        prompt_program = None
        if event.get("event") == "iter_prompt":
            prompt_program = _extract_prompt_program(payload)
        elif event.get("event") in {"train_stats", "val_stats", "dev_stats", "test_stats", "final_test_stats"}:
            prompt_program = _extract_prompt_program(payload)
        prompt_text = _prompt_program_to_text(prompt_program) if prompt_program is not None else ""
        if prompt_text and step not in prompts_by_step:
            prompts_by_step[step] = prompt_text

    token_encoding, count_tokens = _get_token_counter(model_name=model_name, encoding_name=encoding_name)
    step_stats = parse_step_stats(events)
    metric_split = select_metric_split(
        step_stats,
        preferred=_canonical_metric_split(metric_split_preferred),
    )

    steps = sorted(set(prompts_by_step) | set(step_stats))
    rows = []
    for step in steps:
        prompt_text = prompts_by_step.get(step, "")
        metrics = step_stats.get(step, {}).get(metric_split, {})
        rows.append({
            "step": step,
            "prompt_tokens": count_tokens(prompt_text) if prompt_text else np.nan,
            "prompt_chars": len(prompt_text) if prompt_text else np.nan,
            "prompt_lines": prompt_text.count("\n") + 1 if prompt_text else np.nan,
            "metric_split": metric_split,
            "performance_metric": performance_metric,
            "dev_performance": metrics.get(performance_metric),
            "task_score": metrics.get("task_score"),
            "brier": metrics.get("brier"),
            "fmt_rate": metrics.get("fmt_rate"),
            "n": metrics.get("n"),
        })

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    df.attrs["token_encoding"] = token_encoding
    df.attrs["metric_split"] = metric_split
    return df


def _plot_prompt_length_side_by_side(
    df,
    *,
    size_col,
    performance_col,
    performance_label,
    title,
    out_path,
):
    plot_df = df.dropna(subset=[size_col, performance_col])
    if plot_df.empty:
        raise ValueError("No steps have both prompt length and dev performance.")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), sharex=True)
    axes[0].plot(plot_df["step"], plot_df[size_col], marker="o", linewidth=1.8, color=PROMPT_LENGTH_COLOR)
    # axes[0].set_title("Prompt size", fontsize=10)
    axes[0].set_xlabel("optimization step")
    axes[0].set_ylabel("Prompt tokens" if size_col == "prompt_tokens" else "Prompt characters")
    axes[0].grid(alpha=0.25, linewidth=0.6)

    axes[1].plot(plot_df["step"], plot_df[performance_col], marker="o", linewidth=1.8, color=DEV_PERFORMANCE_COLOR)
    # axes[1].set_title("Dev performance", fontsize=10)
    axes[1].set_xlabel("optimization step")
    axes[1].set_ylabel(performance_label)
    axes[1].grid(alpha=0.25, linewidth=0.6)

    for ax in axes:
        _set_integer_step_ticks(ax, plot_df["step"])

    # make title bold
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _set_integer_step_ticks(ax, steps, max_ticks=12):
    step_values = sorted({int(step) for step in steps if pd.notna(step)})
    if not step_values:
        return
    min_step = step_values[0]
    max_step = step_values[-1]
    if len(step_values) <= max_ticks:
        ticks = step_values
    else:
        stride = max(1, int(np.ceil((max_step - min_step) / (max_ticks - 1))))
        ticks = list(range(min_step, max_step + 1, stride))
        if ticks[-1] != max_step:
            ticks.append(max_step)
    ax.set_xticks(ticks)
    ax.set_xlim(min_step - 0.25, max_step + 0.25)


def _plot_prompt_length_combined(
    df,
    *,
    size_col,
    performance_col,
    performance_label,
    title,
    out_path,
):
    plot_df = df.dropna(subset=[size_col, performance_col])
    if plot_df.empty:
        raise ValueError("No steps have both prompt length and dev performance.")

    fig, ax_size = plt.subplots(figsize=(6.2, 3.4))
    ax_perf = ax_size.twinx()

    size_line, = ax_size.plot(
        plot_df["step"],
        plot_df[size_col],
        marker="o",
        linewidth=1.8,
        color=PROMPT_LENGTH_COLOR,
        label="Prompt tokens" if size_col == "prompt_tokens" else "Prompt characters",
    )
    perf_line, = ax_perf.plot(
        plot_df["step"],
        plot_df[performance_col],
        marker="s",
        linewidth=1.8,
        color=DEV_PERFORMANCE_COLOR,
        label=performance_label,
    )

    ax_size.set_xlabel("RPT step")
    ax_size.set_ylabel(size_line.get_label(), color=size_line.get_color())
    ax_perf.set_ylabel(perf_line.get_label(), color=perf_line.get_color())
    ax_size.tick_params(axis="y", labelcolor=size_line.get_color())
    ax_perf.tick_params(axis="y", labelcolor=perf_line.get_color())
    ax_size.grid(alpha=0.25, linewidth=0.6)
    ax_size.set_title(title, fontsize=11)
    _set_integer_step_ticks(ax_size, plot_df["step"])
    ax_size.legend([size_line, perf_line], [size_line.get_label(), perf_line.get_label()], loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_prompt_length_analysis(
    events_path,
    *,
    out_prefix=None,
    model_name=None,
    encoding_name=None,
    size_metric="tokens",
    metric_split_preferred="val_stats",
    performance_metric="task_score",
    style="side_by_side",
    dataset_name=None,
):
    events_path = Path(events_path)
    events = load_jsonl(events_path)
    df = build_prompt_length_per_step(
        events,
        model_name=model_name,
        encoding_name=encoding_name,
        metric_split_preferred=metric_split_preferred,
        performance_metric=performance_metric,
    )

    size_col = "prompt_tokens" if size_metric == "tokens" else "prompt_chars"
    performance_col = "dev_performance"
    if out_prefix is None:
        out_prefix = Path("vis_results") / events_path.stem / "prompt_length_vs_dev"
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_prefix.with_name(out_prefix.name + "_per_step.csv")
    df.to_csv(csv_path, index=False)

    metric_split = df.attrs.get("metric_split")
    split_label = "Dev" if metric_split == "val_stats" else str(metric_split).replace("_stats", "").title()
    performance_label = f"{split_label} {performance_metric.replace('_', ' ')}"
    title = f"{dataset_name or 'Dataset'}"
    outputs = {"csv": csv_path}
    if style in {"side_by_side", "both"}:
        side_by_side_path = out_prefix.with_name(out_prefix.name + "_side_by_side.pdf")
        _plot_prompt_length_side_by_side(
            df,
            size_col=size_col,
            performance_col=performance_col,
            performance_label=performance_label,
            title=title,
            out_path=side_by_side_path,
        )
        outputs["side_by_side"] = side_by_side_path
    if style in {"combined", "both"}:
        combined_path = out_prefix.with_name(out_prefix.name + "_combined.pdf")
        _plot_prompt_length_combined(
            df,
            size_col=size_col,
            performance_col=performance_col,
            performance_label=performance_label,
            title=title,
            out_path=combined_path,
        )
        outputs["combined"] = combined_path

    plot_df = df.dropna(subset=[size_col, performance_col])
    if len(plot_df) >= 2:
        outputs["pearson_r"] = float(plot_df[size_col].corr(plot_df[performance_col], method="pearson"))
        outputs["spearman_r"] = float(plot_df[size_col].corr(plot_df[performance_col], method="spearman"))
    else:
        outputs["pearson_r"] = np.nan
        outputs["spearman_r"] = np.nan
    outputs["token_encoding"] = df.attrs.get("token_encoding")
    outputs["metric_split"] = df.attrs.get("metric_split")
    outputs["n_steps_plotted"] = int(len(plot_df))
    return outputs


def plot_paper_prompt_length_all(
    run_configs=None,
    *,
    out_path="vis_results/fig_prompt_length_vs_dev_all_gpt5.pdf",
    model_name="gpt-5",
    encoding_name=None,
    size_metric="tokens",
    metric_split_preferred="val_stats",
    performance_metric="task_score",
    figsize=(9.6, 4.4),
):
    """
    Create a compact multi-dataset prompt-length figure.

    Columns are datasets; rows are prompt size and dev performance.
    """
    if run_configs is None:
        run_configs = DEFAULT_PAPER_ALIGNMENT_RUNS

    panels = []
    for config in run_configs:
        events_path = Path(config["log_path"])
        if not events_path.exists():
            raise FileNotFoundError(f"Could not find source log for {config['title']}: {events_path}")
        events = load_jsonl(events_path)
        df = build_prompt_length_per_step(
            events,
            model_name=model_name,
            encoding_name=encoding_name,
            metric_split_preferred=metric_split_preferred,
            performance_metric=performance_metric,
        )
        panels.append({"title": config["title"], "df": df})

    size_col = "prompt_tokens" if size_metric == "tokens" else "prompt_chars"
    size_label = "Prompt tokens" if size_col == "prompt_tokens" else "Prompt characters"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rc = {
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            2,
            len(panels),
            figsize=figsize,
            constrained_layout=False,
        )
        if len(panels) == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for col, panel in enumerate(panels):
            plot_df = panel["df"].dropna(subset=[size_col, "dev_performance"])
            if plot_df.empty:
                continue

            ax_size = axes[0, col]
            ax_perf = axes[1, col]

            ax_size.plot(
                plot_df["step"],
                plot_df[size_col],
                marker="o",
                markersize=3.4,
                linewidth=1.5,
                color=PROMPT_LENGTH_COLOR,
            )
            ax_size.set_title(panel["title"], fontsize=10, fontweight="bold", pad=5)
            ax_size.grid(alpha=0.22, linewidth=0.5)
            _set_integer_step_ticks(ax_size, plot_df["step"], max_ticks=7)

            ax_perf.plot(
                plot_df["step"],
                plot_df["dev_performance"],
                marker="o",
                markersize=3.4,
                linewidth=1.5,
                color=DEV_PERFORMANCE_COLOR,
            )
            ax_perf.set_xlabel("Optimization step", fontsize=9)
            ax_perf.grid(alpha=0.22, linewidth=0.5)
            _set_integer_step_ticks(ax_perf, plot_df["step"], max_ticks=7)

            for ax in (ax_size, ax_perf):
                ax.tick_params(axis="both", labelsize=8, length=2)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.6)

        axes[0, 0].set_ylabel(size_label, fontsize=9)
        axes[1, 0].set_ylabel(f"Dev {performance_metric.replace('_', ' ')}", fontsize=9)

        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.13, top=0.88, wspace=0.28, hspace=0.34)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    return out_path


def plot_paper_failure_delta_heatmaps(
    run_configs=None,
    *,
    clustering_root="clustering_results",
    logs_root="logs",
    out_prefix="vis_results/fig_failure_to_deltas_all_acl_v1",
    figsize=(8.4, 2.55),
    cmap="RdBu_r",
    y_label_wrap=14,
    tick_fontsize=5.4,
    title_fontsize=8.8,
    axis_label_fontsize=9.0,
    label_overrides=None,
):
    """
    Create a compact paper figure for failure-topic -> metric-delta heatmaps.
    """
    import matplotlib.colors as mcolors

    if run_configs is None:
        run_configs = DEFAULT_PAPER_ALIGNMENT_RUNS
    if label_overrides is None:
        label_overrides = PAPER_TOPIC_LABEL_OVERRIDES

    panels = []
    for config in run_configs:
        base = resolve_analysis_dir(config["dir_path"], root=clustering_root)
        if not base.exists():
            raise FileNotFoundError(f"Could not find clustering results directory: {base}")

        if config.get("log_path"):
            events_path = Path(config["log_path"])
        else:
            task_name, model_name, _ = task_model_and_stem_from_analysis_dir(base, root=clustering_root)
            events_path = log_path_for_analysis_dir(
                base,
                task_name=task_name,
                model_name=model_name,
                logs_root=logs_root,
                clustering_root=clustering_root,
            )
        if not events_path.exists():
            raise FileNotFoundError(f"Could not find source log for {config['title']}: {events_path}")

        panels.append({
            "title": config["title"],
            "effect_df": build_failure_delta_effect_for_analysis_dir(base, events_path).fillna(0.0),
        })

    all_values = np.concatenate([panel["effect_df"].values.ravel() for panel in panels])
    finite_values = all_values[np.isfinite(all_values)]
    max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
    if max_abs == 0.0:
        max_abs = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_prefix.with_suffix(".png")
    out_pdf = out_prefix.with_suffix(".pdf")

    rc = {
        "font.family": "serif",
        "font.size": tick_fontsize,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=figsize,
            constrained_layout=False,
        )
        if len(panels) == 1:
            axes = [axes]

        image = None
        for ax, panel in zip(axes, panels):
            effect_df = panel["effect_df"]
            image = ax.imshow(
                effect_df.values.astype(float),
                aspect="auto",
                interpolation="nearest",
                norm=norm,
                cmap=cmap,
            )
            ax.set_title(panel["title"], fontsize=title_fontsize, pad=4)
            ax.set_xticks(np.arange(effect_df.shape[1]))
            ax.set_yticks(np.arange(effect_df.shape[0]))
            ax.set_xticklabels(effect_df.columns, fontsize=tick_fontsize + 1.0)
            ax.set_yticklabels(
                _wrap_plot_labels(effect_df.index, y_label_wrap, label_overrides),
                fontsize=tick_fontsize,
            )
            ax.tick_params(axis="both", which="major", length=0, pad=1.2)

            ax.set_xticks(np.arange(-0.5, effect_df.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, effect_df.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.35)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        fig.subplots_adjust(left=0.13, right=0.90, bottom=0.18, top=0.81, wspace=0.90)
        fig.canvas.draw()
        leftmost = axes[0].get_position()
        rightmost = axes[-1].get_position()
        heatmap_x_center = (leftmost.x0 + rightmost.x1) / 2
        heatmap_y_center = (leftmost.y0 + leftmost.y1) / 2
        fig.text(
            heatmap_x_center,
            0.055,
            "Δ metric",
            ha="center",
            va="center",
            fontsize=axis_label_fontsize,
        )
        fig.text(
            0.018,
            heatmap_y_center,
            "Failure topic",
            ha="center",
            va="center",
            rotation=90,
            fontsize=axis_label_fontsize,
        )

        if image is not None:
            cax = fig.add_axes([
                rightmost.x1 + 0.012,
                rightmost.y0,
                0.014,
                rightmost.height,
            ])
            cbar = fig.colorbar(image, cax=cax)
            cbar.set_label("Mean Δ metric", fontsize=axis_label_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize, length=2)

        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

    return out_png, out_pdf


def main():
    # make argparse
    argparser = argparse.ArgumentParser(description="Generate heatmaps to interpret failure/patch topics and metric deltas.")
    # argparser.add_argument("--failure_labeled", type=str, default="clustering_results/log_last_reports_iters_20_dev_450/failure_labeled.json", help="Path to failure_labeled.json") #_test_150_seed_42
    # argparser.add_argument("--failure_labeled_new", type=str, default="clustering_results/log_last_reports_iters_20_dev_450/failures_clusterfusion_cosine.json", help="Path to failures_clusterfusion_cosine.json for updated failure topic assignments")
    # argparser.add_argument("--patch_labeled", type=str, default="clustering_results/log_last_reports_iters_20_dev_450/patch_labeled.json", help="Path to patch_labeled.json")
    # argparser.add_argument("--patch_labeled_new", type=str, default="clustering_results/log_last_reports_iters_20_dev_450/patches_clusterfusion_cosine.json", help="Path to patches_clusterfusion_cosine.json for updated patch topic assignments")
    # argparser.add_argument("--events_jsonl", type=str, default="logs/hotpotqa/log_last_reports_iters_20_dev_450.jsonl", help="Path to log_last_reports_iters_20_dev_450.jsonl")
    argparser.add_argument("--log_path", type=str, default=None, help="Path to the source JSONL log. You can omit .jsonl.")
    argparser.add_argument("--dir_path", type=str, default="hotpotqa/gpt-5/example", help="Clustering-results dir, task/log stem, or bare log stem.")
    argparser.add_argument("--seed", type=int, default=0, help="Random seed")
    argparser.add_argument("--task_name", type=str, default=None, help="Task name for clustering_results/<task_name>/<model_name>/<log_stem>.")
    argparser.add_argument("--model_name", type=str, default=None, help="Model name for clustering_results/<task_name>/<model_name>/<log_stem>.")
    argparser.add_argument("--dataset_name", type=str, default=None, help="Dataset name/title for prompt-length plots; legacy alias for --task_name outside prompt-length analysis.")
    argparser.add_argument("--logs_root", type=str, default="logs", help="Root directory containing task log folders.")
    argparser.add_argument("--clustering_root", type=str, default="clustering_results", help="Root directory for clustering outputs.")
    argparser.add_argument("--vis_root", type=str, default="vis_results", help="Root directory for visualizations.")
    argparser.add_argument("--paper_alignment_all", action="store_true", help="Generate the ACL-style 3-panel alignment heatmap figure.")
    argparser.add_argument("--paper_alignment_out", type=str, default=None, help="Output prefix for --paper_alignment_all; writes .png and .pdf.")
    argparser.add_argument("--paper_patch_deltas_all", action="store_true", help="Generate the ACL-style 3-panel patch-to-deltas heatmap figure.")
    argparser.add_argument("--paper_patch_deltas_out", type=str, default=None, help="Output prefix for --paper_patch_deltas_all; writes .png and .pdf.")
    argparser.add_argument("--paper_failure_deltas_all", action="store_true", help="Generate the ACL-style 3-panel failure-to-deltas heatmap figure.")
    argparser.add_argument("--paper_failure_deltas_out", type=str, default=None, help="Output prefix for --paper_failure_deltas_all; writes .png and .pdf.")
    argparser.add_argument("--paper_prompt_length_all", action="store_true", help="Generate the 3-dataset prompt-length vs dev-performance PDF.")
    argparser.add_argument("--paper_prompt_length_out", type=str, default=None, help="Output path for --paper_prompt_length_all; writes a PDF.")
    argparser.add_argument("--prompt_length_analysis", action="store_true", help="Plot per-step prompt size next to dev/val performance.")
    argparser.add_argument("--prompt_length_out", type=str, default=None, help="Output prefix for prompt-length analysis files.")
    argparser.add_argument("--prompt_length_metric", choices=["tokens", "chars"], default="tokens", help="Primary prompt-size unit to plot.")
    argparser.add_argument("--prompt_length_split", type=str, default="val_stats", help="Metric split for dev performance; dev/dev_stats aliases val_stats.")
    argparser.add_argument("--prompt_length_perf_metric", type=str, default="task_score", help="Metric from the dev/val stats payload to plot.")
    argparser.add_argument("--prompt_length_style", choices=["side_by_side", "combined", "both"], default="side_by_side", help="Prompt-length plot style.")
    argparser.add_argument("--token_encoding", type=str, default=None, help="Optional tiktoken encoding name, e.g. o200k_base.")
    args = argparser.parse_args()
    # failure_labeled = load_json("clustering_results/failure_labeled.json")
    # patch_labeled = load_json("clustering_results/patch_labeled.json")
    # events = load_jsonl("log_all_reports_iters_50.jsonl")

    if args.paper_alignment_all:
        out_prefix = args.paper_alignment_out or str(Path(args.vis_root) / "fig_alignment_all_acl_v1")
        out_png, out_pdf = plot_paper_alignment_heatmaps(
            clustering_root=args.clustering_root,
            out_prefix=out_prefix,
        )
        print("Saved paper alignment heatmaps to:")
        print(f" - {out_png}")
        print(f" - {out_pdf}")
        return

    if args.paper_patch_deltas_all:
        out_prefix = args.paper_patch_deltas_out or str(Path(args.vis_root) / "fig_patch_to_deltas_all_acl_v1")
        out_png, out_pdf = plot_paper_patch_delta_heatmaps(
            clustering_root=args.clustering_root,
            logs_root=args.logs_root,
            out_prefix=out_prefix,
        )
        print("Saved paper patch-to-deltas heatmaps to:")
        print(f" - {out_png}")
        print(f" - {out_pdf}")
        return

    if args.paper_failure_deltas_all:
        out_prefix = args.paper_failure_deltas_out or str(Path(args.vis_root) / "fig_failure_to_deltas_all_acl_v1")
        out_png, out_pdf = plot_paper_failure_delta_heatmaps(
            clustering_root=args.clustering_root,
            logs_root=args.logs_root,
            out_prefix=out_prefix,
        )
        print("Saved paper failure-to-deltas heatmaps to:")
        print(f" - {out_png}")
        print(f" - {out_pdf}")
        return

    if args.paper_prompt_length_all:
        out_path = args.paper_prompt_length_out or str(Path(args.vis_root) / "fig_prompt_length_vs_dev_all_gpt5.pdf")
        out_pdf = plot_paper_prompt_length_all(
            out_path=out_path,
            model_name=args.model_name or "gpt-5",
            encoding_name=args.token_encoding,
            size_metric=args.prompt_length_metric,
            metric_split_preferred=args.prompt_length_split,
            performance_metric=args.prompt_length_perf_metric,
        )
        print("Saved paper prompt-length figure to:")
        print(f" - {out_pdf}")
        return

    if args.prompt_length_analysis:
        task_name = args.task_name
        display_dataset_name = args.dataset_name or args.task_name
        model_name = args.model_name
        if args.log_path:
            events_path = resolve_log_path(
                args.log_path,
                task_name=task_name,
                model_name=model_name,
                logs_root=args.logs_root,
            )
        else:
            base = resolve_analysis_dir(
                args.dir_path,
                task_name=task_name,
                model_name=model_name,
                root=args.clustering_root,
                logs_root=args.logs_root,
            )
            inferred_task, inferred_model, _ = task_model_and_stem_from_analysis_dir(
                base,
                root=args.clustering_root,
            )
            task_name = task_name or inferred_task
            model_name = model_name or inferred_model
            display_dataset_name = display_dataset_name or task_name
            events_path = log_path_for_analysis_dir(
                base,
                task_name=task_name,
                model_name=model_name,
                logs_root=args.logs_root,
                clustering_root=args.clustering_root,
            )
        if not events_path.exists():
            raise FileNotFoundError(f"Could not find source log: {events_path}")
        out_prefix = args.prompt_length_out
        if out_prefix is None:
            stem = events_path.stem
            if task_name and model_name:
                out_prefix = str(Path(args.vis_root) / task_name / model_name / stem / "prompt_length_vs_dev")
            elif task_name:
                out_prefix = str(Path(args.vis_root) / task_name / stem / "prompt_length_vs_dev")
            else:
                out_prefix = str(Path(args.vis_root) / stem / "prompt_length_vs_dev")
        outputs = run_prompt_length_analysis(
            events_path,
            out_prefix=out_prefix,
            model_name=model_name,
            encoding_name=args.token_encoding,
            size_metric=args.prompt_length_metric,
            metric_split_preferred=args.prompt_length_split,
            performance_metric=args.prompt_length_perf_metric,
            style=args.prompt_length_style,
            dataset_name=display_dataset_name,
        )
        print("Saved prompt-length analysis to:")
        for key in ("csv", "side_by_side", "combined"):
            if key in outputs:
                print(f" - {outputs[key]}")
        print(f"Metric split: {outputs['metric_split']}")
        print(f"Token encoding: {outputs['token_encoding']}")
        print(f"Steps plotted: {outputs['n_steps_plotted']}")
        print(f"Pearson r: {outputs['pearson_r']:.4f}")
        print(f"Spearman r: {outputs['spearman_r']:.4f}")
        return

    # failures_labeled_path = f"clustering_results/log_last_report_iters_20_dev_150_test_500_seed_{args.seed}/failure_labeled.json"
    # failures_labeled_new_path = f"clustering_results/log_last_report_iters_20_dev_150_test_500_seed_{args.seed}/failures_clusterfusion_cosine.json"
    # patch_labeled_path = f"clustering_results/log_last_report_iters_20_dev_150_test_500_seed_{args.seed}/patch_labeled.json"
    # patch_labeled_new_path = f"clustering_results/log_last_report_iters_20_dev_150_test_500_seed_{args.seed}/patches_clusterfusion_cosine.json"
    # events_path = f"logs/hotpotqa/log_last_report_iters_20_dev_150_test_500_seed_{args.seed}.jsonl"

    task_name = args.task_name or args.dataset_name
    analysis_ref = args.log_path or args.dir_path
    base = resolve_analysis_dir(
        analysis_ref,
        task_name=task_name,
        model_name=args.model_name,
        root=args.clustering_root,
        logs_root=args.logs_root,
    )
    inferred_task, inferred_model, stem = task_model_and_stem_from_analysis_dir(base, root=args.clustering_root)
    task_name = task_name or inferred_task
    model_name = args.model_name or inferred_model
    fallback_bases = []
    if task_name:
        fallback_bases.append(Path(args.clustering_root) / task_name / stem)
    fallback_bases.append(Path(args.clustering_root) / stem)
    if not base.exists():
        for fallback_base in fallback_bases:
            if fallback_base != base and fallback_base.exists():
                print(f"Using legacy clustering dir: {fallback_base}")
                base = fallback_base
                break

    events_path = (
        resolve_log_path(args.log_path, task_name=task_name, model_name=model_name, logs_root=args.logs_root)
        if args.log_path
        else log_path_for_analysis_dir(
            base,
            task_name=task_name,
            model_name=model_name,
            logs_root=args.logs_root,
            clustering_root=args.clustering_root,
        )
    )
    if task_name and model_name:
        vis_dir = Path(args.vis_root) / task_name / model_name / stem
    elif task_name:
        vis_dir = Path(args.vis_root) / task_name / stem
    else:
        vis_dir = Path(args.vis_root) / stem
    vis_dir.mkdir(parents=True, exist_ok=True)

    failures_labeled_path = base / "failure_labeled.json"
    failures_labeled_new_path = base / "failures_clusterfusion_cosine.json"
    patch_labeled_path = base / "patch_labeled.json"
    patch_labeled_new_path = base / "patches_clusterfusion_cosine.json"

    failure_labeled = load_json(failures_labeled_path)
    failure_labeled_new = load_json(failures_labeled_new_path)
    # topic_reassignment for failures based on failures_clusterfusion_cosine.json file
    for i, r in enumerate(failure_labeled):
        r["topic"] = failure_labeled_new["assignments"][i]
    
    print(f"Loaded {len(failure_labeled)} failure-labeled records and {len(failure_labeled_new['assignments'])} new failure topic assignments.")

    patch_labeled = load_json(patch_labeled_path)
    patch_labeled_new = load_json(patch_labeled_new_path)
    for i, r in enumerate(patch_labeled):
        r["topic"] = patch_labeled_new["assignments"][i]

    print(f"Loaded {len(patch_labeled)} patch-labeled records and {len(patch_labeled_new['assignments'])} new patch topic assignments.")

    # all 
    # log_path_1 = "logs/hotpotqa/log_last_report_iters_20_dev_450_test_150_seed_0.jsonl"
    # events_1 = load_jsonl(log_path_1)
    # log_path_2 = "logs/hotpotqa/log_last_report_iters_20_dev_450_test_150_seed_7.jsonl"
    # events_2 = load_jsonl(log_path_2)
    # log_path_3 = "logs/hotpotqa/log_last_report_iters_20_dev_450_test_150_seed_42.jsonl"
    # events_3 = load_jsonl(log_path_3)
    # events = events_1 + events_2 + events_3
    
    # per seed
    events = load_jsonl(events_path)

    # Transition metrics (deltas)
    step_stats = parse_step_stats(events)
    metric_split = select_metric_split(step_stats, preferred="train_stats")
    print(f"Using metric split: {metric_split}")
    trans_metrics  = build_transition_metrics(step_stats, split=metric_split).set_index("transition")

    # Transition-level topic intensity
    fail_counts, patch_counts = build_transition_topic_tables(failure_labeled, patch_labeled)

    # Align indices
    common = sorted(set(trans_metrics.index) & set(fail_counts.index) & set(patch_counts.index),
                    key=lambda s: parse_transition_key(s) or (10**9, 10**9))
    if not common:
        raise ValueError(
            "No overlapping transitions found between metrics, failure topics, and patch topics. "
            "Check whether the chosen metric split contains stats for the same steps as the clustering outputs, "
            "or whether the clustering outputs "
            "were generated from a different log."
        )
    trans_metrics = trans_metrics.loc[common]
    fail_counts = fail_counts.loc[common]
    patch_counts = patch_counts.loc[common]

    # Optional: rename columns with LLM topic labels if you have them
    # (uncomment + set these if you have the label jsons)
    # failure_names = load_topic_names("/mnt/data/failure_topic_labels.json", "F")
    # patch_names = load_topic_names("/mnt/data/patch_topic_labels.json", "P")
    
    failure_name_map = load_topic_name_map(failure_labeled_new, prefix="F")
    patch_name_map   = load_topic_name_map(patch_labeled_new,   prefix="P")
    fail_counts = fail_counts.rename(columns=failure_name_map)
    patch_counts = patch_counts.rename(columns=patch_name_map)

    # -------------------------
    # Heatmap 1: Error topic -> Patch topic alignment
    # -------------------------
    # Use binary presence to avoid domination by “many bullets”
    fail_present = (fail_counts > 0).astype(int)
    patch_present = (patch_counts > 0).astype(int)

    # P(patch present | failure present): for each failure topic, compute mean patch_present among transitions where failure present
    align = []
    for fcol in fail_present.columns:
        idx = fail_present[fcol] == 1
        if idx.sum() == 0:
            align.append([np.nan] * patch_present.shape[1])
        else:
            align.append(list(patch_present[idx].mean(axis=0).values))
    align_df = pd.DataFrame(align, index=fail_present.columns, columns=patch_present.columns)

    # denominator per failure topic = number of transitions where that failure topic is present
    denom_F = fail_present.sum(axis=0)  # Series indexed by failure topics

    # repeat across patch columns to match align_df shape
    align_denoms = np.repeat(denom_F.values[:, None], align_df.shape[1], axis=1)
    align_denoms_df = pd.DataFrame(align_denoms, index=align_df.index, columns=align_df.columns)

    # intersection numerator n11 (F,P)
    numer_align = fail_present.T @ patch_present 
    numer_align_df = pd.DataFrame(numer_align.values, index=fail_present.columns, columns=patch_present.columns)
   
    plot_alignment_heatmap_fraction(
        align_df.fillna(0.0),
        numer_align_df,
        align_denoms_df,
        title="Alignment heatmap: P(patch topic present | failure topic present)",
        xlabel="Patch topic",
        ylabel="Failure topic",
        out_path=vis_dir / "heatmap_alignment_failure_to_patch_v1.png", #_test_150_seed_{args.seed}
    )

    # -------------------------
    # Heatmap 2: Patch topic -> metric deltas
    # -------------------------
    delta_cols = ["d_task_score", "d_brier"]
    effect = []
    for pcol in patch_present.columns:
        idx = patch_present[pcol] == 1
        if idx.sum() == 0:
            effect.append([np.nan] * len(delta_cols))
        else:
            effect.append([trans_metrics.loc[idx, c].astype(float).mean() for c in delta_cols])
    effect_df = pd.DataFrame(effect, index=patch_present.columns, columns=delta_cols)
    print(trans_metrics[delta_cols].describe(include="all"))
    print(trans_metrics[delta_cols].isna().mean())

    denom_P = patch_present.sum(axis=0)  # patch topics

    # numerator 
    numer_patch_to_deltas = np.repeat(denom_P.loc[effect_df.index].values[:, None], effect_df.shape[1], axis=1)
    numer_patch_to_deltas_df = pd.DataFrame(numer_patch_to_deltas, index=effect_df.index, columns=effect_df.columns)

    eff_denoms = np.repeat(denom_P.loc[effect_df.index].values[:, None], effect_df.shape[1], axis=1)
    eff_denoms_df = pd.DataFrame(eff_denoms, index=effect_df.index, columns=effect_df.columns)
    
    plot_heatmap(
        effect_df.fillna(0.0),
        # eff_denoms_df,
        numer_patch_to_deltas_df,
        title="Effect heatmap: mean Δmetrics when patch topic is present",
        xlabel="Δ metric (t→t+1)",
        ylabel="Patch topic",
        out_path=vis_dir / "heatmap_patch_to_deltas_v1.png",
    )

    # -------------------------
    # Heatmap 3: Failure topic -> metric deltas
    # -------------------------
    effect_f = []
    for fcol in fail_present.columns:
        idx = fail_present[fcol] == 1
        if idx.sum() == 0:
            effect_f.append([np.nan] * len(delta_cols))
        else:
            effect_f.append([trans_metrics.loc[idx, c].astype(float).mean() for c in delta_cols])
    effect_f_df = pd.DataFrame(effect_f, index=fail_present.columns, columns=delta_cols)
    denom_F = fail_present.sum(axis=0)  # failure topics

    numer_failure_to_deltas = np.repeat(denom_F.loc[effect_f_df.index].values[:, None], effect_f_df.shape[1], axis=1)
    numer_failure_to_deltas_df = pd.DataFrame(numer_failure_to_deltas, index=effect_f_df.index, columns=effect_f_df.columns)

    pred_denoms = np.repeat(denom_F.loc[effect_f_df.index].values[:, None], effect_f_df.shape[1], axis=1)
    pred_denoms_df = pd.DataFrame(pred_denoms, index=effect_f_df.index, columns=effect_f_df.columns)

    plot_heatmap(
        effect_f_df.fillna(0.0),
        # pred_denoms_df,
        numer_failure_to_deltas_df,
        title="Predictability heatmap: mean Δmetrics after each failure topic",
        xlabel="Δ metric (t→t+1)",
        ylabel="Failure topic",
        out_path=vis_dir / "heatmap_failure_to_deltas_v1.png",
    )

    print("Saved heatmaps to:")
    print(f" - {vis_dir / 'heatmap_alignment_failure_to_patch_v1.png'}")
    print(f" - {vis_dir / 'heatmap_patch_to_deltas_v1.png'}")
    print(f" - {vis_dir / 'heatmap_failure_to_deltas_v1.png'}")


if __name__ == "__main__":
    main()
