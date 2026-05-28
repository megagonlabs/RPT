import json
from pathlib import Path
from collections import defaultdict
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
import textwrap

from rpt.analysis.paths import (
    log_path_for_analysis_dir,
    resolve_analysis_dir,
    resolve_log_path,
    task_model_and_stem_from_analysis_dir,
)

PERSISTENCE_BAR_COLOR = "#0f766e"

DEFAULT_PAPER_PERSISTENCE_RUNS = [
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

PERSISTENCE_TOPIC_LABEL_OVERRIDES = {
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
}

# ---- Parsing helpers (matches your log schema) ----
# expects events like:
# {"event":"val_stats","step":t,"payload":{"metrics":{"exact_match":..., "f1":..., "brier":..., "avg_confidence":...}}}
def _extract_metrics(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}

    out = {
        "n": metrics.get("n"),
        # "exact_match": metrics.get("exact_match"),
        "accuracy": metrics.get("accuracy"),    
        "f1": metrics.get("f1"),
        "brier": metrics.get("brier"),
        "avg_confidence": metrics.get("avg_confidence"),
        "format_error_rate": metrics.get("format_error_rate"),
        # keep raw too if you want:
        # "raw_metrics": metrics,
    }

    # if nothing useful:
    if all(out[k] is None for k in ["accuracy", "f1", "brier"]): #exact_match
        return {}
    return out


def load_step_stats_from_log(jsonl_path: str) -> dict:
    """
    Returns:
      step_stats[step]["val"]  = metrics dict
      step_stats[step]["test"] = metrics dict
    """
    step_stats = defaultdict(dict)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if "step" not in e:
                continue
            t = int(e["step"])
            ev = e.get("event")
            payload = e.get("payload", {}) or {}

            if ev in {"val_stats", "dev_stats"}:
                m = _extract_metrics(payload)
                if m:
                    step_stats[t]["val"] = m
            elif ev in {"test_stats", "final_test_stats"}:
                m = _extract_metrics(payload)
                if m:
                    step_stats[t]["test"] = m

    return dict(step_stats)


def _best_step_for_metric(step_stats: dict, split: str, metric: str, mode: str):
    """
    mode: "max" or "min"
    """
    best_t, best_v = None, None
    for t in sorted(step_stats.keys()):
        m = step_stats.get(t, {}).get(split, {})
        if not m:
            continue
        v = m.get(metric)
        if v is None:
            continue
        v = float(v)
        if best_v is None:
            best_t, best_v = t, v
        else:
            if (mode == "max" and v > best_v) or (mode == "min" and v < best_v):
                best_t, best_v = t, v
    return best_t, best_v


def summarize_best_iterations(jsonl_path: str) -> dict:
    """
    For a log file:
      - "original" = earliest step that has BOTH val and test stats (fallback: earliest val, earliest test)
      - best per split (val/test) for:
          exact_match (max), f1 (max), brier (min)
      - for each best step, return the FULL val+test metrics for that iteration.
    """
    step_stats = load_step_stats_from_log(jsonl_path)

    steps_sorted = sorted(step_stats.keys())
    if not steps_sorted:
        return {"file": jsonl_path, "error": "No step stats found"}

    # original: earliest step with both splits if possible
    original_step = None
    for t in steps_sorted:
        if "val" in step_stats[t] and "test" in step_stats[t]:
            original_step = t
            break
    if original_step is None:
        # fallback: just earliest available
        original_step = steps_sorted[0]

    def full_perf(t: int) -> dict:
        return {
            "step": t,
            "val": step_stats.get(t, {}).get("val", None),
            "test": step_stats.get(t, {}).get("test", None),
        }

    metrics = [
        ("accuracy", "max"),
        # ("exact_match", "max"),
        ("f1", "max"),
        ("brier", "min"),
    ]

    out = {
        "file": jsonl_path,
        "original": full_perf(original_step),
        "best": {
            "val": {},
            "test": {},
        },
    }

    for split in ["val", "test"]:
        for metric, mode in metrics:
            t_best, v_best = _best_step_for_metric(step_stats, split, metric, mode)
            out["best"][split][metric] = {
                "mode": mode,
                "best_value": v_best,
                "best_step": t_best,
                "performance_at_best_step": full_perf(t_best) if t_best is not None else None,
            }

    return out


# ---- Run on one or many log files ----
def summarize_logs(log_paths):
    results = []
    for p in log_paths:
        results.append(summarize_best_iterations(p))
    return results


# if __name__ == "__main__":
#     # Example usage: update these paths as needed
#     # log_files = [
#     #     "logs/hotpotqa/log_all_reports_iters_20_dev_450_test_500_seed_0_mode_all_reports.jsonl",
#     #     "logs/hotpotqa/log_all_reports_iters_20_dev_300_test_500_seed_0_mode_all_reports.jsonl",
#     #     "logs/hotpotqa/log_all_reports_iters_20_dev_150_test_500_seed_0_mode_all_reports.jsonl",
#     #     "logs/hotpotqa/log_all_reports_iters_20_dev_300_test_500_seed_0_mode_all_reports_comp_cluster_desc.jsonl",
#     #     # add more jsonl logs here...
#     # ]
#     log_files = [
#         "logs/hle/log_all_reports_iters_20_dev_100_test_500_seed_0_k_topics_10_cluster_desc_comprehensive_text_question_mode_exact_match_w_exp.jsonl",
#         "logs/hle/log_all_reports_iters_20_dev_200_test_500_seed_0_k_topics_10_cluster_desc_comprehensive_text_question_mode_exact_match.jsonl",
#         "logs/hle/log_all_reports_iters_20_dev_300_test_500_seed_0_k_topics_10_cluster_desc_comprehensive_text_question_mode_exact_match.jsonl",
#         "logs/hle/log_all_reports_iters_20_dev_300_test_500_seed_0_k_topics_10_cluster_desc_comprehensive_text_question_mode_multiple_choice.jsonl",
#         "logs/hle/log_all_reports_iters_20_dev_300_test_500_seed_0_k_topics_20_cluster_desc_comprehensive_text_question_mode_exact_match.jsonl",
#         "logs/hle/log_all_reports_iters_20_dev_400_test_500_seed_0_k_topics_10_cluster_desc_comprehensive_text_question_mode_exact_match.jsonl",
#         "logs/hle/log_last_k_reports_iters_20_dev_200_test_500_seed_0_k_topics_10_cluster_desc_comprehensive_text_question_mode_exact_match.jsonl",
#         "logs/hle/log_last_k_reports_iters_20_dev_300_test_500_seed_0_k_topics_10_cluster_desc_comprehensive_text_question_mode_exact_match.jsonl",
#     ]

#     results = summarize_logs(log_files)

#     # Pretty-print
#     print(json.dumps(results, indent=2))

#     # Optionally save
#     Path("logs/hotpotqa/best_iterations_summary_iters_20_test_500_last_report.json").write_text(
#         json.dumps(results, ensure_ascii=False, indent=2),
#         encoding="utf-8",
#     )
#     # print("Saved: logs/hotpotqa/best_iterations_summary_iters_20_dev_450.json")


# -----------------------------
# 1) Loaders
# -----------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def load_failure_docs(failure_docs_path: str) -> List[str]:
    """
    Supports:
      1) {"docs": [...]}
      2) [...] (plain list)
    """
    obj = load_json(failure_docs_path)
    if isinstance(obj, dict) and "docs" in obj:
        return obj["docs"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unrecognized failure_docs.json schema in {failure_docs_path}")


def load_clusterfusion_assignments(clusterfusion_path: str) -> Tuple[List[int], Dict[int, str]]:
    """
    clusterfusion file contains:
      - topics: [{topic_id, name, ...}, ...]
      - assignments: [topic_id per record]
    Returns:
      assignments, tid2name
    """
    cf = load_json(clusterfusion_path)

    assignments = [int(x) for x in cf["assignments"]]
    tid2name = {int(t["topic_id"]): t["name"] for t in cf["topics"]}

    return assignments, tid2name


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def build_name2tid_from_failure_docs(
    failure_docs: List[str],
    assignments: List[int]) -> Dict[str, int]:
    """
    Map analysis_output failure 'name' -> topic_id by matching it to the
    beginning of a doc in failure_docs (docs start with the name, then details).
    """
    n = min(len(failure_docs), len(assignments))
    docs_norm = [_norm(d) for d in failure_docs[:n]]

    # Pre-index by the "head" (first ~8 words) to speed up matching a bit
    # (optional but helps if you later scale)
    head_index: Dict[str, List[int]] = {}
    for i, d in enumerate(docs_norm):
        head = " ".join(d.split()[:8])
        head_index.setdefault(head, []).append(i)

    def resolve(name: str) -> int:
        nn = _norm(name)
        # Fast path: use head index
        head = " ".join(nn.split()[:8])
        candidates = head_index.get(head, list(range(n)))

        # Prefer startswith
        for i in candidates:
            if docs_norm[i].startswith(nn):
                return int(assignments[i])

        # Fallback: contains (rare, but safe)
        for i in candidates:
            if nn in docs_norm[i]:
                return int(assignments[i])

        return -1

    # return a resolver dict; you can also just return the resolve() function
    return {"__resolver__": resolve}

# -----------------------------
# 3) Extract per-step failure labels from your run log
# -----------------------------
def extract_step_failure_labels(log_rows):
    step2labels = defaultdict(list)
    for r in log_rows:
        if r.get("event") != "failure_mode_clusters": #r.get("event") != "analysis_output" or
            continue
        step = r.get("step")
        if step is None:
            continue
        # failures = (r.get("payload", {}) or {}).get("failure_modes", [])
        failures = (r.get("payload", {}) or {}).get("topics", [])
        for fm in failures:
            if isinstance(fm, dict) and fm.get("name"):
                step2labels[int(step)].append(str(fm["name"]))
            elif isinstance(fm, str):
                step2labels[int(step)].append(fm)
    return step2labels

# -----------------------------
# 4) Convert labels -> topic ids per step
# -----------------------------
def labels_to_topics_per_step(step2labels, name2tid, unknown_tid=-1):
    resolver = name2tid.get("__resolver__", None)
    step2tids = {}
    for step, labels in step2labels.items():
        tids = []
        for lab in labels:
            tid = resolver(lab) if resolver is not None else name2tid.get(lab, unknown_tid)
            tids.append(tid if tid is not None else unknown_tid)
        step2tids[step] = tids
    return step2tids


def step_topic_presence(step2tids: Dict[int, List[int]], drop_unknown: bool = True) -> Dict[int, set]:
    out = {}
    for step, tids in step2tids.items():
        s = set(tids)
        if drop_unknown and (-1 in s):
            s.remove(-1)
        out[step] = s
    return out


# -----------------------------
# 5) Per-transition resolved / repeated / new
# -----------------------------
def transition_stats(step2set: Dict[int, set]) -> List[dict]:
    steps = sorted(step2set.keys())
    rows = []
    for i in range(len(steps) - 1):
        t = steps[i]
        t1 = steps[i + 1]
        A = step2set[t]
        B = step2set[t1]
        rows.append({
            "t": t,
            "t1": t1,
            "repeated": len(A & B),
            "resolved": len(A - B),
            "new": len(B - A),
            "size_t": len(A),
            "size_t1": len(B),
        })
    return rows


# -----------------------------
# 6) Persistence lengths (run lengths)
# -----------------------------
def topic_persistence_lengths(step2set: Dict[int, set]) -> Dict[int, List[int]]:
    steps = sorted(step2set.keys())
    topic_runs: Dict[int, List[int]] = defaultdict(list)

    active_len: Dict[int, int] = defaultdict(int)
    active_now: set = set()

    for step in steps:
        cur = step2set[step]

        # increment those continuing
        for tid in cur:
            active_len[tid] = active_len.get(tid, 0) + 1

        # end runs for those that disappeared
        ended = active_now - cur
        for tid in ended:
            topic_runs[tid].append(active_len.get(tid, 0))
            active_len[tid] = 0

        active_now = cur

    # flush at end
    for tid in list(active_now):
        topic_runs[tid].append(active_len.get(tid, 0))

    return topic_runs


def compute_avg_run_length(topic_runs: Dict[int, List[int]], tid2name: Dict[int, str]) -> pd.DataFrame:
    """
    Returns a dataframe with:
      topic_id, topic_name, run_count, avg_run_len, max_run_len, total_occurrences
    where total_occurrences = sum(run lengths) (steps present).
    """
    rows = []
    for tid, runs in topic_runs.items():
        if not runs:
            continue
        rows.append({
            "topic_id": tid,
            "topic_name": tid2name.get(tid, f"F{tid}"),
            "run_count": len(runs),
            "avg_run_len": float(np.mean(runs)),
            "max_run_len": int(np.max(runs)),
            "total_occurrences": int(np.sum(runs)),
        })
    df = pd.DataFrame(rows).sort_values(["avg_run_len", "total_occurrences"], ascending=False)
    return df

# -----------------------------
# 7) Plotting
# -----------------------------
def plot_transition_bars(trans_rows: List[dict], out_path: Optional[str] = None):
    ts = [r["t"] for r in trans_rows]
    repeated = [r["repeated"] for r in trans_rows]
    resolved = [r["resolved"] for r in trans_rows]
    new = [r["new"] for r in trans_rows]

    x = np.arange(len(ts))

    plt.figure(figsize=(max(10, 0.6 * len(ts) + 4), 5))
    plt.bar(x, repeated, label="repeated (persisted)")
    plt.bar(x, resolved, bottom=repeated, label="resolved (disappeared)")
    plt.bar(x, new, bottom=np.array(repeated) + np.array(resolved), label="new (appeared)")

    plt.xticks(x, [f"{t}->{t+1}" for t in ts], rotation=45, ha="right")
    plt.ylabel("# failure topics")
    plt.title("Per-transition failure-topic dynamics")
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


# def plot_persistence_scatter(topic_runs: Dict[int, List[int]], tid2name: Dict[int, str], top_k: int = 10, out_path: Optional[str] = None):
#     scores = [(tid, sum(runs)) for tid, runs in topic_runs.items()]
#     scores.sort(key=lambda x: x[1], reverse=True)
#     chosen = [tid for tid, _ in scores[:top_k]]

#     plt.figure(figsize=(max(10, 0.8 * len(chosen) + 6), 5))
#     for tid in chosen:
#         runs = topic_runs.get(tid, [])
#         xs = [tid] * len(runs)
#         plt.scatter(xs, runs)

#     plt.xticks(chosen, [tid2name.get(t, f"F{t}") for t in chosen], rotation=35, ha="right")
#     plt.ylabel("consecutive steps active (run length)")
#     plt.title("Persistence run lengths for top topics")
#     plt.tight_layout()

#     if out_path:
#         plt.savefig(out_path, dpi=200, bbox_inches="tight")
#     plt.show()
def topic_frequency_and_runs(step2set: Dict[int, set], topic_runs: Dict[int, List[int]]) -> Dict[int, Dict[str, int]]:
    """
    Returns per-topic:
      - occurrences: number of steps where topic is present
      - run_count: number of contiguous runs
      - total_run_steps: sum(run_lengths) (should equal occurrences)
      - max_run: maximum run length
    """
    # occurrences: count steps containing topic
    occ = defaultdict(int)
    for _, s in step2set.items():
        for tid in s:
            occ[tid] += 1

    stats = {}
    for tid, runs in topic_runs.items():
        stats[tid] = {
            "occurrences": int(occ.get(tid, 0)),
            "run_count": int(len(runs)),
            "total_run_steps": int(sum(runs)),
            "max_run": int(max(runs)) if runs else 0,
        }
    # include topics that appear but might not be in topic_runs for some reason
    for tid, c in occ.items():
        if tid not in stats:
            stats[tid] = {"occurrences": int(c), "run_count": 0, "total_run_steps": 0, "max_run": 0}
    return stats

def plot_persistence_scatter_with_freq(
    topic_runs: Dict[int, List[int]],
    tid2name: Dict[int, str],
    stats: Dict[int, Dict[str, int]],
    top_k: int = 10,
    out_path: Optional[str] = None,):
    # Choose top topics by occurrences (frequency), not by sum(runs) (same but clearer)
    scored = [(tid, stats.get(tid, {}).get("occurrences", 0)) for tid in stats.keys()]
    scored.sort(key=lambda x: x[1], reverse=True)
    chosen = [tid for tid, _ in scored[:top_k]]

    plt.figure(figsize=(max(12, 1.0 * len(chosen) + 8), 5))

    # jitter to avoid overplotting when many runs have same length (e.g., many 1s)
    rng = np.random.default_rng(0)

    for x_idx, tid in enumerate(chosen):
        runs = topic_runs.get(tid, [])
        if not runs:
            continue
        xs = np.full(len(runs), x_idx, dtype=float)
        xs += rng.uniform(-0.08, 0.08, size=len(runs))  # small horizontal jitter
        plt.scatter(xs, runs)

    # label each topic with occurrences and run_count
    xticklabels = []
    for tid in chosen:
        name = tid2name.get(tid, f"F{tid}")
        occ = stats.get(tid, {}).get("occurrences", 0)
        rc = stats.get(tid, {}).get("run_count", 0)
        mx = stats.get(tid, {}).get("max_run", 0)
        xticklabels.append(f"{name}\nocc={occ}, runs={rc}, max={mx}")

    plt.xticks(range(len(chosen)), xticklabels, rotation=25, ha="right")
    plt.ylabel("consecutive steps active (run length)")
    plt.title("Persistence run lengths (dots=runs) + frequency (occ) per topic")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()

def _wrapped_topic_labels(labels, width=34):
    wrapped = []
    for label in labels:
        label = PERSISTENCE_TOPIC_LABEL_OVERRIDES.get(str(label), str(label))
        wrapped.append(
            "\n".join(
                textwrap.wrap(
                    label,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        )
    return wrapped


def _integer_ticks_with_padding(max_x, target_ticks=6, right_pad_frac=0.18):
    max_x = max(1.0, float(max_x))
    axis_max = max_x * (1.0 + right_pad_frac)
    max_tick = max(1, int(np.ceil(axis_max)))
    tick_step = max(1, int(np.ceil(max_tick / max(1, target_ticks - 1))))
    ticks = list(range(0, max_tick + tick_step, tick_step))
    return ticks, max(ticks[-1], axis_max)


def _annotate_horizontal_bars(ax, values, y_positions, axis_max, *, fontsize=8):
    offset = max(axis_max * 0.018, 0.12)
    for value, y in zip(values, y_positions):
        value = float(value)
        if value + (3.7 * offset) > axis_max:
            ax.text(
                max(value - offset, axis_max * 0.05),
                y,
                f"{value:.1f}",
                va="center",
                ha="right",
                fontsize=fontsize,
                color="white",
                fontweight="bold",
                clip_on=True,
            )
        else:
            ax.text(
                value + offset,
                y,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=fontsize,
                color="black",
                clip_on=True,
            )


def plot_avg_run_length(avg_df, top_k=15, out_path=None):
    df = avg_df.head(top_k).iloc[::-1]  # reverse for nicer horizontal bars
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.2, max(4.0, 0.55 * len(df) + 1.2)))
    y = np.arange(len(df))
    bars = ax.barh(y, df["avg_run_len"], color=PERSISTENCE_BAR_COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels(_wrapped_topic_labels(df["topic_name"]), fontsize=9.5)
    ax.set_xlabel("Average consecutive optimization steps active", fontsize=10)
    ax.set_title("Average persistence of failure topics", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    max_x = float(df["avg_run_len"].max())
    ticks, axis_max = _integer_ticks_with_padding(max_x, target_ticks=8, right_pad_frac=0.20)
    ax.set_xticks(ticks)
    ax.set_xlim(0, axis_max)

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0, pad=1.2)
    _annotate_horizontal_bars(ax, df["avg_run_len"], y, axis_max, fontsize=10)
    fig.subplots_adjust(left=0.36, right=0.96, bottom=0.16, top=0.90)

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_avg_run_length_for_paths(analysis_dir, log_path):
    base = Path(analysis_dir)
    failure_docs = load_failure_docs(base / "failure_docs.json")
    assignments, tid2name = load_clusterfusion_assignments(base / "failures_clusterfusion_cosine.json")
    name2tid = build_name2tid_from_failure_docs(failure_docs, assignments)

    log_rows = load_jsonl(log_path)
    step2labels = extract_step_failure_labels(log_rows)
    step2tids = labels_to_topics_per_step(step2labels, name2tid, unknown_tid=-1)
    step2set = step_topic_presence(step2tids, drop_unknown=True)

    runs = topic_persistence_lengths(step2set)
    return compute_avg_run_length(runs, tid2name)


def plot_paper_avg_run_length_all(
    run_configs=None,
    *,
    clustering_root="clustering_results",
    out_path="vis_results/fig_avg_run_length_all_gpt5.pdf",
    top_k=10,
    figsize=(15.0, 5.9),
):
    if run_configs is None:
        run_configs = DEFAULT_PAPER_PERSISTENCE_RUNS

    panels = []
    for config in run_configs:
        base = resolve_analysis_dir(config["dir_path"], root=clustering_root)
        if not base.exists():
            raise FileNotFoundError(f"Could not find clustering results directory: {base}")
        log_path = Path(config["log_path"])
        if not log_path.exists():
            raise FileNotFoundError(f"Could not find source log for {config['title']}: {log_path}")
        panels.append({
            "title": config["title"],
            "avg_df": build_avg_run_length_for_paths(base, log_path),
        })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rc = {
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(panels), figsize=figsize, constrained_layout=False)
        if len(panels) == 1:
            axes = [axes]

        for ax, panel in zip(axes, panels):
            df = panel["avg_df"].head(top_k).iloc[::-1]
            y = np.arange(len(df))
            bars = ax.barh(y, df["avg_run_len"], color=PERSISTENCE_BAR_COLOR)
            ax.set_title(panel["title"], fontsize=12.0, fontweight="bold", pad=6)
            ax.set_yticks(y)
            ax.set_yticklabels(_wrapped_topic_labels(df["topic_name"], width=22), fontsize=8.4)
            ax.grid(axis="x", alpha=0.25, linewidth=0.5)
            ax.set_axisbelow(True)

            max_x = float(df["avg_run_len"].max()) if not df.empty else 1.0
            ticks, axis_max = _integer_ticks_with_padding(max_x, target_ticks=6, right_pad_frac=0.22)
            ax.set_xticks(ticks)
            ax.set_xlim(0, axis_max)
            ax.tick_params(axis="x", labelsize=9.0, length=2)
            ax.tick_params(axis="y", length=0, pad=1.2)
            _annotate_horizontal_bars(ax, df["avg_run_len"], y, axis_max, fontsize=9.2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

        fig.suptitle("Average persistence of failure topics", fontsize=14, fontweight="bold", y=0.985)
        fig.text(
            0.53,
            0.030,
            "Average consecutive optimization steps active",
            ha="center",
            va="center",
            fontsize=11,
        )
        fig.subplots_adjust(left=0.070, right=0.990, bottom=0.13, top=0.87, wspace=0.68)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    return out_path
    
# -----------------------------
# 8) MAIN
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Path to the source JSONL log. You can omit .jsonl.",
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="hotpotqa/gpt-5/example",
        help="Existing clustering-results dir, task/log stem, or bare log stem.",
    )
    parser.add_argument("--strict", action="store_true", help="Require exact length match between docs and assignments.")
    parser.add_argument("--task_name", type=str, default=None, help="Task name for clustering_results/<task_name>/<model_name>/<log_stem>.")
    parser.add_argument("--model_name", type=str, default=None, help="Model name for clustering_results/<task_name>/<model_name>/<log_stem>.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Deprecated alias for --task_name.")
    parser.add_argument("--logs_root", type=str, default="logs", help="Root directory containing task log folders.")
    parser.add_argument("--clustering_root", type=str, default="clustering_results", help="Root directory for clustering outputs.")
    parser.add_argument("--vis_root", type=str, default="vis_results", help="Root directory for visualizations.")
    parser.add_argument("--paper_avg_run_length_all", action="store_true", help="Generate a 3-dataset side-by-side average persistence PDF.")
    parser.add_argument("--paper_avg_run_length_out", type=str, default=None, help="Output path for --paper_avg_run_length_all.")
    args = parser.parse_args()

    if args.paper_avg_run_length_all:
        out_path = args.paper_avg_run_length_out or str(Path(args.vis_root) / "fig_avg_run_length_all_gpt5.pdf")
        out_pdf = plot_paper_avg_run_length_all(
            clustering_root=args.clustering_root,
            out_path=out_path,
        )
        print("Saved paper average-persistence figure to:")
        print(f" - {out_pdf}")
        raise SystemExit(0)

    task_name = args.task_name or args.dataset_name
    analysis_ref = args.log_path or args.docs_dir
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

    log_path = (
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

    failure_docs_path = base / "failure_docs.json"
    failure_clusterfusion_path = base / "failures_clusterfusion_cosine.json"

    # Load docs + clusterfusion assignments
    failure_docs = load_failure_docs(failure_docs_path)
    assignments, tid2name = load_clusterfusion_assignments(failure_clusterfusion_path)

    # Build name->tid resolver based on prefix match
    name2tid = build_name2tid_from_failure_docs(failure_docs, assignments)

    log_rows = load_jsonl(log_path)

    step2labels = extract_step_failure_labels(log_rows)
    step2tids = labels_to_topics_per_step(step2labels, name2tid, unknown_tid=-1)
    step2set = step_topic_presence(step2tids, drop_unknown=True)

    print("steps:", sorted(step2set.keys()))
    print("topic counts per step:", {k: len(v) for k, v in step2set.items()})

    # Compute + plot
    trans_rows = transition_stats(step2set)
    # Save visualizations under vis_results/<task_name>/<log_stem>/.
    os.makedirs(vis_dir, exist_ok=True)
    plot_transition_bars(trans_rows, out_path=vis_dir / "failure_transition_dynamics.png")

    runs = topic_persistence_lengths(step2set)
    stats = topic_frequency_and_runs(step2set, runs)
    plot_persistence_scatter_with_freq(runs, tid2name, stats, top_k=10, out_path=vis_dir / "failure_persistence_runs_with_freq.png")

    avg_df = compute_avg_run_length(runs, tid2name)
    print(avg_df)
    plot_avg_run_length(avg_df, top_k=10, out_path=vis_dir / "avg_run_length.png")
    # plot_persistence_scatter(runs, tid2name, top_k=10, out_path=f"vis_results/{args.dataset_name}/{args.docs_dir}/failure_persistence_runs.png")

    print("Saved:")
    print(" - failure_transition_dynamics.png")
    print(" - failure_persistence_runs.png")
