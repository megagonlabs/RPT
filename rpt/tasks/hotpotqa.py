"""
Reflective Prompt Tuning (RPT) via Function Calling
- Dataset: HotpotQA (validation split), sample 100-200
- Target LLM: gpt-4.1 (answers each instance + self-justification + confidence)
- Generator/Evaluator: configurable optimizer model (reflects, calls tools, proposes prompt patch)

Requirements:
  pip install openai datasets pydantic

Env:
  export OPENAI_API_KEY="..."

Run:
  python -m rpt.tasks.hotpotqa --n 150 --iters 6 --seed 7
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import random
import re
import os
import string
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field
from datasets import load_dataset
from rpt.analysis.cluster_fusion import run_clusterfusion, ClusterFusionConfig, PartitionConfig
from rpt.common import JsonlLogger
from rpt.paths import HOTPOTQA_DATA_DIR

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ----------------------------
# Models / schemas (Structured Outputs)
# ----------------------------

class TargetAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0, description="Model's confidence that answer is correct.")
    justification: str = Field(description="Brief justification of the decision (1-4 sentences).")


class PromptProgram(BaseModel):
    system: str
    instruction: str
    # “Context engineering” knobs you’ll tune
    enforce_json_only: bool = True
    max_justification_sentences: int = 3

    def render(self, question: str, context: str) -> Tuple[str, str]:
        sys = self.system
        user = (
            f"{self.instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n" 
        )
        return sys, user


class EvalMetrics(BaseModel):
    n: int
    exact_match: float
    f1: float
    precision: float
    recall: float
    avg_confidence: float
    brier: float
    format_error_rate: float


class EvalItemTrace(BaseModel):
    idx: int
    question: str
    context: Optional[str]
    gold: str
    pred: Optional[str]
    correct_em: bool
    f1: float
    precision: float
    recall: float
    confidence: Optional[float]
    justification: Optional[str]
    # raw_text: str
    error_type: str  # ok | wrong | non_json | other


class EvalReport(BaseModel):
    iteration: int = 0
    prompt_program: Dict[str, Any] = Field(description="The PromptProgram (system/instruction) used to evaluate a target model and produce this report.")
    metrics: EvalMetrics
    # error_summary: Dict[str, int]
    # hard_examples: List[EvalItemTrace]
    insights: Optional[TraceInsights] = None   # NEW


class PromptPatch(BaseModel):
    # Minimal, structured edits only
    system: Optional[str] = None
    instruction: Optional[str] = None
    enforce_json_only: Optional[bool] = None
    max_justification_sentences: Optional[int] = None
    rationale: str


class LoopDecision(BaseModel):
    action: str = Field(description="Either 'patch' or 'stop'.")
    patch: Optional[PromptPatch] = None
    stop_reason: Optional[str] = None


# class FailureMode(BaseModel):
#     name: str = Field(description="Short label, e.g., 'entity mismatch', 'context ignored', 'overconfident wrong'")
#     description: str
#     prevalence_estimate: float = Field(ge=0.0, le=1.0, description="Rough fraction of errors this explains")
#     example_indices: List[int] = Field(default_factory=list)

class FailureModeItem(BaseModel):
    label: str = Field(description="2–6 words, consistent across similar errors")
    definition: str = Field(description="Explanation of the failure mode")
    why: str = Field(description="Brief explanation for THIS example")
    basis: str = Field(description="Cite what in the trace or justification shows this")

class FailureCritique(BaseModel):
    failure_modes: List[FailureModeItem] = Field(default_factory=list, description="1–3 induced failure modes")

class FailureModeTopic(BaseModel):
    name: str
    definition: str
    # prevalence: float = Field(ge=0.0, le=1.0, description="Fraction of failure labels in this cluster")
    examples: List[str] = Field(default_factory=list)

class PromptAction(BaseModel):
    action: str = Field(description="Concrete prompt edit suggestion")
    rationale: str
    expected_effect: str = Field(description="What metric/behavior this should improve")
    risk: str = Field(description="Downside, e.g. verbosity, worse recall, format regressions")
    priority: int = Field(ge=1, le=5)


# class TraceInsights(BaseModel):
#     strengths: List[str] = Field(default_factory=list)
#     failure_modes: List[FailureMode] = Field(default_factory=list)
#     # prompt_actions: List[PromptAction] = Field(default_factory=list)
#     # notes: List[str] = Field(default_factory=list)

class TraceInsights(BaseModel): # based on the topics from cluster fusion + failure modes from the critic
    failure_modes: List[FailureModeTopic] = Field(default_factory=list)

# {
#     "topic_id": i,
#     "name": name,
#     "definition": desc,
#     "examples": [str(x) for x in exs][:8],
# }

# ----------------------------
# HotpotQA utils
# ----------------------------

def load_hotpotqa_val(sample_n: int, seed: int, excluding_ids: List[int] = None) -> List[Dict[str, Any]]:
    # HotpotQA on HF typically provides "distractor" config with context paragraphs.
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    idxs = list(range(len(ds)))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    excluding_ids = set(excluding_ids or [])
    items = []
    for i in idxs:
        row = ds[int(i)]
        if row.get("id") not in excluding_ids:
            items.append(row)
            if len(items) >= sample_n:
                break
    return items[:sample_n]


def load_jsonl_items(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items


def write_jsonl_items(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def repair_cached_hotpot_split(
    path: str,
    items: List[Dict[str, Any]],
    sample_n: int,
    seed: int,
    excluding_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    excluding_ids = set(excluding_ids or [])
    seen_ids = set()
    kept_items: List[Dict[str, Any]] = []
    removed_overlap_ids: List[str] = []
    removed_duplicate_ids: List[str] = []

    for item in items:
        item_id = item.get("id")
        if item_id in excluding_ids:
            removed_overlap_ids.append(item_id)
            continue
        if item_id in seen_ids:
            removed_duplicate_ids.append(item_id)
            continue
        kept_items.append(item)
        seen_ids.add(item_id)

    changed = (
        len(kept_items) != len(items)
        or len(kept_items) != sample_n
    )
    if not changed:
        return kept_items

    needed = sample_n - len(kept_items)
    if needed < 0:
        kept_items = kept_items[:sample_n]
        needed = 0

    if needed > 0:
        refill_exclusions = list(excluding_ids | seen_ids)
        refill_items = load_hotpotqa_val(
            sample_n=needed,
            seed=seed,
            excluding_ids=refill_exclusions,
        )
        kept_items.extend(refill_items)

    if len(kept_items) != sample_n:
        raise ValueError(
            f"Unable to repair split at {path}: expected {sample_n} items, got {len(kept_items)}."
        )

    backup_path = f"{path}.bak"
    if not os.path.exists(backup_path):
        write_jsonl_items(backup_path, items)

    write_jsonl_items(path, kept_items)
    print(
        f"Repaired cached split {path}: kept {len(items) - len(removed_overlap_ids) - len(removed_duplicate_ids)}/{len(items)} items, "
        f"removed {len(removed_overlap_ids)} overlaps and {len(removed_duplicate_ids)} duplicates, "
        f"added {needed} replacements. Backup: {backup_path}"
    )
    return kept_items


def load_or_create_hotpot_split(
    path: str,
    sample_n: int,
    seed: int,
    excluding_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
    if os.path.exists(path):
        items = load_jsonl_items(path)
        return repair_cached_hotpot_split(
            path=path,
            items=items,
            sample_n=sample_n,
            seed=seed,
            excluding_ids=excluding_ids,
        )

    items = load_hotpotqa_val(sample_n=sample_n, seed=seed, excluding_ids=excluding_ids)
    write_jsonl_items(path, items)
    return items


def ensure_disjoint_hotpot_splits(named_splits: Dict[str, List[Dict[str, Any]]]) -> None:
    split_ids = {
        name: {item.get("id") for item in items if item.get("id") is not None}
        for name, items in named_splits.items()
    }
    split_names = list(split_ids.keys())
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise ValueError(
                    f"HotpotQA splits '{left}' and '{right}' overlap on {len(overlap)} ids."
                )


#NOTE: should format context also be automated in the future versions? 
def format_context(row: Dict[str, Any], max_chars: int = 9000) -> str:
    """
    Context is a list of [title, [sent1, sent2, ...]].
    We'll serialize with titles and sentence indices for grounded referencing.
    """
    ctx = row.get("context") or {}
    titles = ctx.get("title") or []
    sentences_list = ctx.get("sentences") or []
    
    chunks = []
    for title, sents in zip(titles, sentences_list):
        para = " ".join(sents)
        chunks.append(f"### {title}\n{para}")
    text = "\n\n".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED]"
    return text


def get_cached_context(row: Dict[str, Any], max_chars: int = 90000) -> str:
    cached = row.get("_formatted_context")
    cached_max_chars = row.get("_formatted_context_max_chars")
    if isinstance(cached, str) and cached_max_chars == max_chars:
        return cached

    text = format_context(row, max_chars=max_chars)
    row["_formatted_context"] = text
    row["_formatted_context_max_chars"] = max_chars
    return text


def prime_context_cache(items: List[Dict[str, Any]], max_chars: int = 90000) -> None:
    for row in items:
        get_cached_context(row, max_chars=max_chars)


# ----------------------------
# QA metrics (HotpotQA official answer evaluation)
# ----------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_precision_recall(pred: str, gold: str) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(pred)
    normalized_ground_truth = normalize_answer(gold)
    zero_metric = (0.0, 0.0, 0.0)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def f1_score(pred: str, gold: str) -> float:
    f1, _, _ = f1_precision_recall(pred, gold)
    return f1



# ----------------------------
# LLM to analyze traces (new)
# ----------------------------
def critique_one_trace_with_gpt5(
    client: OpenAI,
    trace: Dict[str, Any],
    model: str = "gpt-5",
    temperature: float = 0.0,
    ) -> FailureCritique:
    
    """
    Assumes the instance is wrong (since you only pass wrong answers).
    Returns: {"failure_modes":[{"label","why","basis"}, ...]}
    """

    CRITIC_SYSTEM = f"""You are a strict evaluation critic for QA failures.
        You are given ONE QA trace:
            - question
            - context (titles + snippets)
            - gold answer
            - predicted answer
            - model confidence
            - model justification

        Your goal is to diagnose WHY the target model produced the wrong answer.
        
        Instructions:
        1) Produce 1–3 failure_modes with:
            - label: 2–6 words, consistent across similar errors
            - definition: comprehensive explanation of the failure mode
            - why: brief explanation for THIS example
            - basis: cite what in the trace or justification shows this
        2) Make labels concrete and clusterable:
            - Prefer labels like 'wrong bridge entity' over long sentences.
            - Do not include entity names, dates, or example-specific details in labels.
        3) If you cannot identify a clear failure mode, return an empty list.
        4) Output ONLY valid JSON matching the schema (no extra text).
    """

    # Keep the user payload compact + stable.
    user_payload = {
        "question": trace.get("question"),
        "context": trace.get("context", None),  # optional; include if you pass it
        "gold_answer": trace.get("gold"),
        "predicted_answer": trace.get("pred"),
        "confidence": trace.get("confidence"),
        "justification": trace.get("justification"),
    }

    # how to set cluster description_mode in FailureCritique before sending to resp? 

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        text_format=FailureCritique,
        # temperature=temperature,
    )

    crit: FailureCritique = resp.output_parsed
    return crit
    # return json.loads(crit.model_dump_json())


# def run_per_item_critic(
#     client: OpenAI,
#     traces: List[Dict[str, Any]],
#     max_items: Optional[int] = None,
#     show_progress: bool = True,
#     ) -> List[Dict[str, Any]]:
#     """
#     Runs the critic on each trace and returns a list with per-item critiques.
#     Each entry: {"idx": int, "critique": FailureCritique-json, "trace": slim-trace}
#     """
#     items = traces[:max_items] if max_items is not None else traces

#     out = []
#     it = tqdm(items, desc="GPT-5 critic per-sample") if show_progress else items
#     for t in it:
#         if t.get("error_type") == "ok":
#             continue  # skip correct answers; focus on failures
#         out.append({
#             "idx": int(t.get("idx")),
#             "critique": crit,
#             # store a slim copy for later inspection
#             "trace": {
#                 "question": t.get("question"),
#                 "context": t.get("context"),
#                 "gold": t.get("gold"),
#                 "pred": t.get("pred"),
#                 "confidence": t.get("confidence"),
#                 "justification": t.get("justification"),
#                 "error_type": t.get("error_type"),
#             }
#         })
#     return out


# def analyze_traces_with_optimizer(
#     client: OpenAI,
#     *,
#     prompt_program: PromptProgram,
#     traces: List[Dict[str, Any]],
#     metrics: Dict[str, Any],
#     task_name: str = "current_task",
#     max_traces: int = 30) -> Dict[str, Any]:
#     """
#     Post-hoc analysis step:
#     GPT-5 reasons over successes/failures and returns structured insights
#     the generator can use to propose the next prompt patch.
#     """
#     # Sample to control token cost: keep a mix of correct/incorrect/format errors
#     correct = [t for t in traces if t.get("correct_em") is True or t.get("error_type") == "ok"]
#     wrong = [t for t in traces if t.get("error_type") in ("wrong", "wrong_answer")]
#     # fmt = [t for t in traces if t.get("error_type") in ("non_json", "missing_key")] not implemented yet.

#     sampled = []
#     # sampled.extend(fmt[: min(len(fmt), max_traces // 5)])
#     sampled.extend(wrong[: min(len(wrong), (max_traces * 4) // 5)])
#     sampled.extend(correct[: max(0, max_traces - len(sampled))])
#     random.shuffle(sampled)

#     system = (
#         "You are an evaluation analyst. "
#         "You will be given: (1) a prompt program, (2) task metrics, and (3) sampled per-instance traces "
#         "from a target model. Your job is to extract actionable insights to improve the prompt. "
#         "Focus on systematic error patterns, calibration issues, and instruction/format weaknesses to identify failure modes. " #NOTE: we can change this and add more well-defined categories of errors.
#         "Also, highlight strengths of the current prompt/program. "
#         "Provide strengths and major failure modes with prevalence estimates. "
#         "Return ONLY JSON that matches the required schema."
#     )

#     user = json.dumps(
#         {
#             "task_name": task_name,
#             "prompt_program": prompt_program.model_dump_json(),
#             "metrics": metrics,
#             "sampled_traces": sampled,
#             # "request": (
#             #     "Identify strengths, major failure modes with prevalence estimates, "
#             #     "and concrete prompt edit actions (prioritized)."
#             # ),
#         },
#         ensure_ascii=False,
#         indent=2,
#     )

#     # Structured output parse with Pydantic model
#     resp = client.responses.parse(
#         model="gpt-5",
#         input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
#         text_format=TraceInsights,
#     )
#     insights: TraceInsights = resp.output_parsed
#     return json.loads(insights.model_dump_json())


# ----------------------------
# OpenAI calls
# ----------------------------

def call_target_gpt41(client: OpenAI, prompt: PromptProgram, question: str, context: str) -> Tuple[Optional[TargetAnswer], str, str]:
    """
    Calls gpt-4.1 with Structured Outputs via responses.parse (Pydantic).
    """
    sys, user = prompt.render(question, context)

    # We keep “format discipline” mostly via Structured Outputs rather than prompt threats.
    # See Structured Outputs guide. (Responses API + parse helper) :contentReference[oaicite:4]{index=4}
    try:
        resp = client.responses.parse(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            text_format=TargetAnswer,
            temperature=0.0,
        )
        parsed: TargetAnswer = resp.output_parsed
        return parsed, resp.output_text, "ok"
    except Exception as e:
        return None, f"[PARSE_ERROR] {e}", "non_json"

def call_gpt5_structured(
    client: OpenAI,
    system: str,
    user: str,
    schema_model: Any,
    model: str = "gpt-5",
) -> Any:
    """
    Structured output helper for optimizer-side models using responses.parse.
    """
    return client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=schema_model,
    )


def evaluate_single_item(
    client: OpenAI,
    prompt: PromptProgram,
    row: Dict[str, Any],
    idx: int,
    max_context_chars: int,
    ) -> Dict[str, Any]:
    question = row["question"]
    gold = row["answer"]
    context = get_cached_context(row, max_chars=max_context_chars)

    parsed, raw_text, err = call_target_gpt41(client, prompt, question, context)

    if parsed is None:
        trace = EvalItemTrace(
            idx=idx,
            question=question,
            context=context,
            gold=gold,
            pred=None,
            correct_em=False,
            f1=0.0,
            precision=0.0,
            recall=0.0,
            confidence=None,
            justification=None,
            error_type=err,
        )
    else:
        pred = parsed.answer
        f1_value, precision, recall = f1_precision_recall(pred, gold)
        trace = EvalItemTrace(
            idx=idx,
            question=question,
            context=context,
            gold=gold,
            pred=pred,
            correct_em=exact_match(pred, gold),
            f1=f1_value,
            precision=precision,
            recall=recall,
            confidence=float(parsed.confidence),
            justification=parsed.justification,
            error_type=err,
        )

    return {
        "idx": idx,
        "row": row,
        "question": question,
        "gold": gold,
        "context": context,
        "parsed": parsed,
        "raw_text": raw_text,
        "trace": trace,
        "err": err,
    }


def critique_single_trace(
    client: OpenAI,
    row: Dict[str, Any],
    trace: EvalItemTrace,
    optimizer_name: str,
    ) -> Dict[str, Any]:
    crit = critique_one_trace_with_gpt5(
        client,
        trace.model_dump(),
        model=optimizer_name,
    )
    return {
        "idx": trace.idx,
        "row_id": row.get("id"),
        "question": row.get("question", ""),
        "critique": crit,
        "impact": 1.0 - float(trace.f1),
    }

# ----------------------------
# Tool: evaluate_prompt (called by the optimizer model)
# ----------------------------

def evaluate_prompt_tool(
    client: OpenAI,
    # prompt_dict: Dict[str, Any],
    prompt: PromptProgram,
    hotpot_items: List[Dict[str, Any]],
    logger: JsonlLogger,
    max_context_chars: int = 90000,
    step: int = 0,
    mode : str = "train",
    k_topics: int = 10,
    clustering_sample_size: int = 100,
    optimizer_name: str = "gpt-5",
    seed: int = 0,
    max_workers: int = 20,
    enable_critiques: bool = True) -> Dict[str, Any]:
    # prompt = PromptProgram(**prompt_dict)

    traces: List[EvalItemTrace] = []
    failure_labels = []  # for collecting failure examples for the critic step
    fmt_errors = 0
    confs = []
    brier_terms = []
    precisions = []
    recalls = []
    incorrect_results: List[Dict[str, Any]] = []
    max_workers = max(1, min(max_workers, len(hotpot_items) if hotpot_items else 1))
    eval_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_single_item,
                client,
                prompt,
                row,
                idx,
                max_context_chars,
            )
            for idx, row in enumerate(hotpot_items)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {mode} items"):
            eval_results.append(future.result())

    eval_results.sort(key=lambda item: item["idx"])

    for result in eval_results:
        idx = result["idx"]
        row = result["row"]
        parsed = result["parsed"]
        trace = result["trace"]

        logger.log("target_trace", step=step, payload={
            "idx": idx,
            "row_id": row.get("id"),
            "question": result["question"],
            "gold": result["gold"],
            "pred": parsed.answer if parsed else None,
            "confidence": float(parsed.confidence) if parsed else None,
            "justification": parsed.justification if parsed else None,
            "error": result["err"],
            "em": trace.correct_em,
            "f1": trace.f1,
            "precision": trace.precision,
            "recall": trace.recall,
        })

        traces.append(trace)
        if parsed is None:
            fmt_errors += 1
            continue

        conf = float(parsed.confidence)
        confs.append(conf)
        precisions.append(trace.precision)
        recalls.append(trace.recall)
        y = 1.0 if trace.correct_em else 0.0
        brier_terms.append((conf - y) ** 2)

        if mode == "train" and not trace.correct_em:
            incorrect_results.append(result)

    # performance aggregates
    n = len(traces)
    em_rate = sum(1 for t in traces if t.correct_em) / max(1, n)
    f1_avg = sum(t.f1 for t in traces) / max(1, n)
    precision_avg = sum(precisions) / max(1, len(precisions))
    recall_avg = sum(recalls) / max(1, len(recalls))
    avg_conf = sum(confs) / max(1, len(confs))
    brier = sum(brier_terms) / max(1, len(brier_terms))
    fmt_rate = fmt_errors / max(1, n)

    # simple error summary
    err_sum: Dict[str, int] = {}
    for t in traces:
        err_sum[t.error_type] = err_sum.get(t.error_type, 0) + 1

    report = EvalReport(
        iteration=step,
        prompt_program=prompt.model_dump(),
        metrics=EvalMetrics(
            n=n,
            exact_match=em_rate,
            f1=f1_avg,
            precision=precision_avg,
            recall=recall_avg,
            avg_confidence=avg_conf,
            brier=brier,
            format_error_rate=fmt_rate,
        ),
        # error_summary=err_sum, # this version only adds erronous examples
        # hard_examples=hard,
        insights = None
    )
    
    if mode in {"dev", "test"}:
        return json.loads(report.model_dump_json())

    if incorrect_results:
        critique_results: List[Dict[str, Any]] = []
        critique_workers = max(1, min(max_workers, len(incorrect_results)))
        with ThreadPoolExecutor(max_workers=critique_workers) as executor:
            futures = [
                executor.submit(
                    critique_single_trace,
                    client,
                    result["row"],
                    result["trace"],
                    optimizer_name,
                )
                for result in incorrect_results
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Critiquing errors"):
                critique_results.append(future.result())

        critique_results.sort(key=lambda item: item["idx"])
        for critique_result in critique_results:
            crit = critique_result["critique"]
            failure_id_start = len(failure_labels)
            for fm in crit.failure_modes:
                failure_labels.append({
                    "id": failure_id_start,
                    "text": f"{str(fm.label)}:{str(fm.definition)}",
                    "impact": critique_result["impact"],
                })
                failure_id_start += 1

            logger.log("item_critique", step=step, payload={
                "idx": critique_result["idx"],
                "row_id": critique_result["row_id"],
                "question": critique_result["question"],
                "critique": json.loads(crit.model_dump_json()) if crit else None,
            })

    # clustering failure modes with GPT-5 critic outputs
    if not failure_labels:
        return json.loads(report.model_dump_json())

    domain_guidance = (
    "You will receive short failure-mode labels produced by an iterative prompt optimization method. "
    "Each record is describing a failure mode in the model’s behavior. " # is a label describing ...
    # "The labels are intended to guide the prompt optimizer in understanding and addressing target model's weaknesses."
    )
    feature_context = "failure modes"
    clustering_cfg = ClusterFusionConfig(
        k_topics=k_topics,
        partition=PartitionConfig(num_groups=2*k_topics, sample_size=clustering_sample_size, seed=seed, cosine_order=True),
        domain_guidance=domain_guidance,
        feature_context=feature_context,
        text_field="text",
        topic_desc_mode="comprehensive",
    )
    topics = run_clusterfusion(failure_labels, clustering_cfg, get_topics=True)  # returns topics with names and definitions, which we can use to add more structured insights to the report in the future versions.
    logger.log("failure_mode_clusters", step=step, payload={"topics": topics})

    # only choose topics with at least 10% prevalence among the failure labels # opt: and sort by prevalence.
    selected_topics = [t for t in topics if t.get("prevalence", 0) >= 0.10]
    if len(selected_topics) < 1:
        selected_topics = topics[: min(len(topics), 3)]  # if no topic meets the threshold, take top 3 anyway (if available)

    # remove prevalence from the topics before adding to the report, so all failure classes get the same weight in the insights and the generator can focus on the qualitative description of the failure modes rather than the quantitative prevalence (which may be noisy and less actionable for prompt edits).
    for t in selected_topics:
        t.pop("prevalence", None)
        t.pop("topic_id", None)  # remove topic_id as well since it's not needed for the insights and may not be stable across iterations
    # Add topics to the report insights
    report.insights = TraceInsights(
        failure_modes=[FailureModeTopic(**t) for t in selected_topics]
    )

    return json.loads(report.model_dump_json())


# ----------------------------
# Patch application
# ----------------------------

def apply_patch(prompt: PromptProgram, patch: PromptPatch) -> PromptProgram:
    data = prompt.model_dump()
    p = patch.model_dump(exclude_none=True)
    p.pop("rationale", None)
    data.update(p)
    return PromptProgram(**data)


# ----------------------------
# Main loop: optimizer model calls evaluate_prompt(), then returns patch/stop (Structured)
# ----------------------------
def prompt_complexity_chars(prompt_program: Optional[Dict[str, Any]]) -> int:
    if not prompt_program:
        return 0
    system_text = str(prompt_program.get("system", "") or "")
    instruction_text = str(prompt_program.get("instruction", "") or "")
    return len(system_text) + len(instruction_text)


def score_report(
    report: Dict[str, Any],
    w_em: float = 0.5,
    w_f1: float = 1.0,
    w_brier: float = 0.05,
    prompt_complexity_weight: float = 0.0,
    prompt_complexity_unit: float = 1000.0,
) -> float:
    m = report["metrics"]
    em = float(m["exact_match"])
    f1 = float(m["f1"])
    brier = float(m["brier"])
    prompt_complexity_penalty = 0.0
    if prompt_complexity_weight > 0.0:
        prompt_complexity_penalty = (
            prompt_complexity_weight
            * prompt_complexity_chars(report.get("prompt_program"))
            / prompt_complexity_unit
        )
    # weighted average (stable) — equivalent to sum if w's sum to 1
    return w_f1 * f1 + w_em * em - w_brier * brier - prompt_complexity_penalty

def better_score(a: float, b: float, eps: float = 1e-12) -> bool:
    return a > b + eps


def build_current_summary(
    history_reports: List[Dict[str, Any]],
    best_report: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
    current_report = history_reports[-1]
    current_metrics = current_report["metrics"]
    prev_metrics = history_reports[-2]["metrics"] if len(history_reports) >= 2 else None
    best_metrics = best_report["metrics"] if best_report else None

    current_score = score_report(current_report, w_em=0.5, w_f1=0.5)
    prev_score = score_report(history_reports[-2], w_em=0.5, w_f1=0.5) if len(history_reports) >= 2 else None
    best_score = score_report(best_report, w_em=0.5, w_f1=0.5) if best_report else None

    non_improving_streak = 0
    for report in reversed(history_reports):
        if best_report is not None and report["iteration"] == best_report["iteration"]:
            break
        non_improving_streak += 1

    summary = {
        "iteration": current_report["iteration"],
        "metrics": current_metrics,
        "insights": current_report.get("insights"),
        "best_so_far": {
            "iteration": best_report["iteration"] if best_report else None,
            "metrics": best_metrics,
            "score": best_score,
        },
        "delta_vs_previous": None,
        "delta_vs_best": None,
        "non_improving_streak": non_improving_streak,
        "did_last_patch_improve_train": None,
    }

    if prev_metrics is not None and prev_score is not None:
        summary["delta_vs_previous"] = {
            "exact_match": float(current_metrics["exact_match"] - prev_metrics["exact_match"]),
            "f1": float(current_metrics["f1"] - prev_metrics["f1"]),
            "precision": float(current_metrics["precision"] - prev_metrics["precision"]),
            "recall": float(current_metrics["recall"] - prev_metrics["recall"]),
            "brier": float(current_metrics["brier"] - prev_metrics["brier"]),
            "format_error_rate": float(current_metrics["format_error_rate"] - prev_metrics["format_error_rate"]),
            "score": float(current_score - prev_score),
        }
        summary["did_last_patch_improve_train"] = bool(current_score > prev_score + 1e-12)

    if best_metrics is not None and best_score is not None:
        summary["delta_vs_best"] = {
            "exact_match": float(current_metrics["exact_match"] - best_metrics["exact_match"]),
            "f1": float(current_metrics["f1"] - best_metrics["f1"]),
            "precision": float(current_metrics["precision"] - best_metrics["precision"]),
            "recall": float(current_metrics["recall"] - best_metrics["recall"]),
            "brier": float(current_metrics["brier"] - best_metrics["brier"]),
            "format_error_rate": float(current_metrics["format_error_rate"] - best_metrics["format_error_rate"]),
            "score": float(current_score - best_score),
        }

    return summary


OPTIMIZER_INSTRUCTIONS = (
        "You are the Reflective Prompt Tuning (RPT) controller.\n\n"
        "Your goal is to iteratively improve a PromptProgram for a QA task.\n\n"
        "At each iteration you must:\n"
        "  (1) Call `evaluate_prompt` exactly once on the CURRENT PromptProgram.\n"
        "  (2) Read the returned evaluation report with insights.\n"
        "  (3) Output either a PATCH or STOP.\n\n"
        "Optimization target:\n"
        "  - Primary: improve Exact Match (EM) and F1 score on the training set.\n"
        "  - Secondary: improve calibration (lower Brier / reduce overconfidence) without hurting EM/F1.\n\n"
        "Decision guidance:\n"
        "  - When current_summary is provided, use it as the primary decision signal, especially current_summary.metrics and any deltas vs previous/best.\n"
        "  - Use history only to detect trajectory, regressions, and previously ineffective edits.\n"
        "Patch constraints:\n"
        "  - Patches should be minimal, targeted to the failure mode, and should address the underlying issue with concrete actionable guidance.\n"
        "  - Avoid vague guidance that only restates the failure; specify what the model should check, compare, extract, or verify.\n"
        "  - Prefer revising, merging, deleting, or reorganizing existing instructions over adding new broad rules.\n"
        "  - Keep the output contract stable (JSON schema and required fields).\n"
        "  - Avoid adding redundant rules; consolidate or prioritize if conflicts arise.\n"
        "Stop condition:\n"
        "  - Output STOP if train-set performance has plateaued or further edits are unlikely to help.\n\n"
        "Hard rule:\n"
        "  - Do NOT propose a PATCH or STOP decision before calling `evaluate_prompt` and receiving its result."
)


def run_rpt(
    client: OpenAI,
    hotpot_train_items: List[Dict[str, Any]],
    hotpot_dev_items: List[Dict[str, Any]],
    hotpot_test_items: List[Dict[str, Any]],
    seed_prompt: PromptProgram,
    logger: JsonlLogger,
    iters: int = 5,
    mode: str = "last_report", # "last_report" | "all_reports"
    test_every: int = 5,
    clustering_sample_size: int = 100,
    k_topics: int = 10,
    k_reports: int = 5, 
    seed: int = 0,
    eval_workers: int = 20,
    optimizer_name: str = "gpt-5",
    prompt_complexity_weight: float = 0.0,
    prompt_complexity_unit: float = 1000.0,
    ) -> PromptProgram:

    """
    Reflective Prompt Tuning loop.

    - Each iteration:
      1) Evaluate CURRENT prompt on TEST (logged only; no optimizer access required).
      2) Ask the optimizer controller to call evaluate_prompt tool on TRAIN.
      3) Execute evaluate_prompt_tool locally on TRAIN and return tool output to the optimizer.
      4) Evaluate the CURRENT prompt on held-out DEV outside the tool flow and use it for model selection.
      5) Ask the optimizer for a LoopDecision (PATCH or STOP), using either:
         - last_report: only the current iteration report
         - history_summary: past history plus a current summary
         - last_k_reports: recent history window
         - all_reports: full history of reports
    """

    # Correct tool schema for Responses API: {"type":"function","function":{...}}
    tools = [
        {
            "type": "function",
            "name": "evaluate_prompt",
            "description": (
                "Evaluate a PromptProgram on the HotpotQA train set using the target model (gpt-4.1). "
                "Returns an evaluation report (metrics + analysis insights)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt_program": {
                        "type": "object",
                        "description": (
                            "The FULL PromptProgram to evaluate on the target model. "
                            "Copy the current PromptProgram from the user message.\n"
                            "Fields: system, instruction, enforce_json_only, max_justification_sentences."
                        ),
                        "properties": {
                            "system": {"type": "string", "description": "System message for the target model."},
                            "instruction": {"type": "string", "description": "Task instruction shown to the target model."},
                            "enforce_json_only": {"type": "boolean", "description": "If true, target must output only JSON."},
                            "max_justification_sentences": {
                                "type": "integer",
                                "description": "Max number of sentences in the target model justification.",
                                "minimum": 1,
                                "maximum": 10
                            },
                            # "required": [
                            #     "system",
                            #     "instruction",
                            #     "enforce_json_only",
                            #     "max_justification_sentences",
                            # ],
                        },
                    },
                    # "max_context_chars": {"type": "integer", "default": 9000},
                },
                "required": ["prompt_program"],
            },
        }
    ]

    # Use fixed optimizer instructions consistently across tool-call and decision steps.
    optimizer_instructions = OPTIMIZER_INSTRUCTIONS


    prompt = seed_prompt
    train_history_reports: List[Dict[str, Any]] = []  # optimizer-facing train reports
    best_train_report = None

    best_prompt = prompt
    best_dev_report = None
    best_dev_score = float("-inf")
    for t in range(iters):
        step = t + 1
        # ------------------------------------------------------------
        # (A) TEST evaluation BEFORE update (no analysis/critic)
        # ------------------------------------------------------------
        if (step % test_every) == 0 or step == 1:  # step == 1 is the baseline evaluation before any updates
            test_report_json = evaluate_prompt_tool(
                client,
                prompt,
                hotpot_test_items,
                logger=logger,
                step=step,
                mode="test",
                max_workers=eval_workers,
            )
            logger.log("test_stats", step=step, payload=test_report_json)

        # log current prompt
        logger.log("iter_prompt", step=step, payload=prompt.model_dump())
        print(f"\n=== RPT Iteration {step}/{iters} (mode={mode}) ===")

        # ------------------------------------------------------------
        # (B) TOOL-CALL STEP (per OpenAI function calling flow)
        # 1) Create a running input list
        # 2) Call model with tools
        # 3) Append response.output to input_list
        # 4) Execute function calls, append function_call_output to input_list
        # ------------------------------------------------------------
        input_list: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Iteration {t+1}/{iters}\n\n"
                    "Call `evaluate_prompt` on the CURRENT PromptProgram below.\n"
                    "Use this JSON as evaluate_prompt.prompt_program:\n\n"
                    f"{prompt.model_dump_json(indent=2)}\n"                ),
            },
        ]

        # Tool-call step
        response = client.responses.parse(
            model=optimizer_name,
            input=input_list,
            instructions=optimizer_instructions,
            tools=tools,
        )

        # Save model outputs into input_list (required by docs flow for reasoning/tool calls)
        input_list += response.output
        
        # Execute any tool calls
        got_tool_call = False
        tool_outputs: List[Dict[str, Any]] = []
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            if item.name != "evaluate_prompt":
                continue

            got_tool_call = True
            args = json.loads(item.arguments)

            # You can choose to trust args["prompt_program"] or always use current prompt to avoid drift.
            # Here we use the current prompt (recommended).
            report = evaluate_prompt_tool(
                client,
                prompt,
                hotpot_train_items,
                logger=logger,
                step=step,
                k_topics=k_topics,
                clustering_sample_size=clustering_sample_size,
                optimizer_name=optimizer_name,
                seed=seed,
                mode="train",
                max_workers=eval_workers,
            )
            logger.log("train_stats", step=step, payload=report)
            train_history_reports.append(report)

            tool_output = {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(report, ensure_ascii=False),
            }
            tool_outputs.append(tool_output)
            input_list.append(tool_output)

            train_score = score_report(
                report,
                prompt_complexity_weight=prompt_complexity_weight,
                prompt_complexity_unit=prompt_complexity_unit,
            )
            if best_train_report is None or better_score(
                train_score,
                score_report(
                    best_train_report,
                    prompt_complexity_weight=prompt_complexity_weight,
                    prompt_complexity_unit=prompt_complexity_unit,
                ),
            ):
                best_train_report = report

            dev_report = evaluate_prompt_tool(
                client,
                prompt,
                hotpot_dev_items,
                logger=logger,
                step=step,
                mode="dev",
                max_workers=eval_workers,
                enable_critiques=False,
            )
            logger.log("dev_stats", step=step, payload=dev_report)

            curr_dev_score = score_report(
                dev_report,
                prompt_complexity_weight=prompt_complexity_weight,
                prompt_complexity_unit=prompt_complexity_unit,
            )
            if better_score(curr_dev_score, best_dev_score):
                best_dev_score = curr_dev_score
                best_dev_report = dev_report
                best_prompt = prompt
                logger.log("best_update", step=step, payload={
                    "selection_split": "dev",
                    "score": best_dev_score,
                    "dev_metrics": dev_report["metrics"],
                    "train_metrics": report["metrics"],
                    "prompt_program": report.get("prompt_program", prompt.model_dump()),
                })

        if not got_tool_call:
            raise RuntimeError("Optimizer model did not call evaluate_prompt. Fix the controller prompt or tool schema.")

        # ------------------------------------------------------------
        # (C) DECISION STEP
        # Provide either the last report or full history, clearly.
        # ------------------------------------------------------------
        current_summary = build_current_summary(train_history_reports, best_train_report)
        if mode == "last_report":
            decision_payload = {
                "mode": "last_report",
                "current_prompt_program": prompt.model_dump(),
                "current_summary": current_summary,
            }
            mode_hint = (
                "You are only given the CURRENT iteration summary.\n"
                "Do not assume access to earlier reports.\n"
            )
        elif mode == "all_reports":
            decision_payload = {
                "mode": "all_reports",
                "current_prompt_program": prompt.model_dump(),
                "history": train_history_reports,
            }
            mode_hint = (
                "You are given the FULL HISTORY of evaluation reports up to the current iteration.\n"
                "Use history to avoid oscillations and avoid reintroducing previously fixed failures.\n"
            )
        elif mode == "history_summary":
            decision_payload = {
                "mode": "history_summary",
                "current_prompt_program": prompt.model_dump(),
                "history": train_history_reports[:-1],
                "current_summary": current_summary,
            }
            mode_hint = (
                "You are given PAST report history, plus a separate current_summary for the current iteration.\n"
                "Use history for trajectory and current_summary for the decision now.\n"
            )
        elif mode == "last_k_reports":
            decision_payload = {
                "mode": "last_k_reports",
                "current_prompt_program": prompt.model_dump(),
                "history": train_history_reports[-k_reports:],
            }
            mode_hint = (
                f"You are given the LAST {k_reports} evaluation reports up to the current iteration.\n"
                "Use history to avoid oscillations and avoid reintroducing previously fixed failures.\n"
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # decision_payload = {
        #     "mode": mode,
        #     "current_prompt_program": prompt.model_dump(),
        #     "history": history_reports if mode == "all_reports" else history_reports[-1],
        # }

        # Append a final user message to the *same* input_list, then call responses.create again
        input_list.append(
            {
                "role": "user",
                "content": (
                    f"{mode_hint}\n"
                    "Using the most recent `function_call_output` evaluation report above, now decide whether to STOP or output a PATCH.\n"
                    "Return a LoopDecision JSON with fields:\n"
                    "  - action: 'patch' or 'stop'\n"
                    "  - patch (only if action='patch')\n"
                    "  - stop_reason (only if action='stop')\n\n"
                    f"{json.dumps(decision_payload, ensure_ascii=False, indent=2)}"
                ),
            }
        )

        decision_resp = client.responses.parse(
            model=optimizer_name,
            input=input_list,
            instructions=optimizer_instructions,
            text_format=LoopDecision,
        )
        decision: LoopDecision = decision_resp.output_parsed
        logger.log("decision", step=step, payload=decision.model_dump())

        if decision.action == "stop":
            print(f"[STOP] {decision.stop_reason}")
            break

        if decision.action == "patch" and decision.patch is not None:
            prompt = apply_patch(prompt, decision.patch)
            continue

        raise RuntimeError("Invalid LoopDecision from optimizer model.")

    print("\n=== FINAL PROMPTPROGRAM ===")
    print(best_prompt.model_dump_json(indent=2))
    print("Best dev score:", best_dev_score, "Best dev metrics:", best_dev_report["metrics"] if best_dev_report else None)
    # run best prompt on test set and log final test performance
    final_test_report_json = evaluate_prompt_tool(
        client,
        best_prompt,
        hotpot_test_items,
        logger=logger,
        step=iters,
        mode="test",
        max_workers=eval_workers,
    )
    logger.log("final_test_stats", step=iters, payload=final_test_report_json)
    return best_prompt


# ----------------------------
# Seed prompt (context engineering baseline)
# ----------------------------

def make_seed_prompt() -> PromptProgram:
    return PromptProgram(
        system="You are tasked with answering questions using only the provided context.",
        instruction=(
            "**Instructions:**\n"
            "- Reason step by step using only the provided context.\n"
            "- Be concise but thorough in your justification.\n"
            "- Before answering, verify that your answer is supported by the context.\n\n"
            "Your output should be a JSON with fields:\n"
            "- justification: a context-grounded explanation of how you reached the answer.\n"
            "- answer: your concise final answer.\n"
            "- confidence: a number in [0,1] representing your confidence in the final answer.\n"
        ),
        enforce_json_only=True,
        max_justification_sentences=3,
    )


def run_seed_prompt_evaluation(
    client: OpenAI,
    hotpot_train_items: List[Dict[str, Any]],
    hotpot_dev_items: List[Dict[str, Any]],
    hotpot_test_items: List[Dict[str, Any]],
    logger: JsonlLogger,
    eval_workers: int,
    ) -> Dict[str, Dict[str, Any]]:
    seed_prompt = make_seed_prompt()
    logger.log("iter_prompt", step=1, payload=seed_prompt.model_dump())

    train_report = evaluate_prompt_tool(
        client,
        seed_prompt,
        hotpot_train_items,
        logger=logger,
        step=1,
        mode="train",
        max_workers=eval_workers,
        enable_critiques=False,
    )
    logger.log("train_stats", step=1, payload=train_report)

    dev_report = evaluate_prompt_tool(
        client,
        seed_prompt,
        hotpot_dev_items,
        logger=logger,
        step=1,
        mode="dev",
        max_workers=eval_workers,
        enable_critiques=False,
    )
    logger.log("dev_stats", step=1, payload=dev_report)

    test_report = evaluate_prompt_tool(
        client,
        seed_prompt,
        hotpot_test_items,
        logger=logger,
        step=1,
        mode="test",
        max_workers=eval_workers,
    )
    logger.log("test_stats", step=1, payload=test_report)

    return {"train": train_report, "dev": dev_report, "test": test_report}


def main():
    # save the prompt with the best performance on the dev set across iterations, and also the final prompt after iteration 20. (later: add early stopping and save that one too)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=300, help="HotpotQA train sample size used for optimizer feedback.")
    ap.add_argument("--n_dev", type=int, default=300, help="HotpotQA val sample size (100-200 recommended).")
    ap.add_argument("--n_test", type=int, default=500, help="HotpotQA test sample size (100-200 recommended).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling and clustering.")
    ap.add_argument("--iters", type=int, default=20) #latest: 20
    ap.add_argument("--mode", type=str, default="all_reports", choices=["all_reports", "history_summary", "last_report", "last_k_reports"], help="`history_summary` provides past history plus a separate current summary for the decision step.")
    ap.add_argument("--k_topics", type=int, default=10, help="Number of failure mode clusters for the optimizer-side critic.")
    ap.add_argument("--k_reports", type=int, default=5, help="Number of recent reports to provide in 'last_k_reports' mode.")
    ap.add_argument("--clustering_sample_size", type=int, default=100, help="Number of failure examples to sample for clustering.")
    ap.add_argument("--test_every", type=int, default=5, help="test every N iterations (including iteration 1 for the initial prompt)")
    ap.add_argument("--eval_workers", type=int, default=20, help="Number of worker threads for dev/test evaluation and critique calls.")
    ap.add_argument(
        "--optimizer_name",
        type=str,
        default="gpt-5",
        help="Model name used for optimizer/controller and critique calls.",
    )
    ap.add_argument(
        "--prompt_complexity_weight",
        type=float,
        default=0.0,
        help=(
            "Penalty weight subtracted from selection score as "
            "weight * prompt_chars / prompt_complexity_unit."
        ),
    )
    ap.add_argument(
        "--prompt_complexity_unit",
        type=float,
        default=1000.0,
        help="Normalization unit for prompt complexity penalty, in characters.",
    )
    ap.add_argument("--evaluate_only", action="store_true", help="Evaluate the seed prompt on the cached dev/test sets and exit.")
    args = ap.parse_args()

    random.seed(args.seed)

    client = OpenAI()

    train_path = str(HOTPOTQA_DATA_DIR / "train.jsonl")
    dev_path = str(HOTPOTQA_DATA_DIR / "dev.jsonl")
    test_path = str(HOTPOTQA_DATA_DIR / "test.jsonl")


    hotpot_train_items = load_or_create_hotpot_split(
        path=train_path,
        sample_n=args.n_train,
        seed=args.seed,
    )

    train_ids = [item.get("id") for item in hotpot_train_items]

    hotpot_test_items = load_or_create_hotpot_split(
        path=test_path,
        sample_n=args.n_test,
        seed=args.seed + 1,
        excluding_ids=train_ids,
    )
    test_ids = [item.get("id") for item in hotpot_test_items]

    hotpot_dev_items = load_or_create_hotpot_split(
        path=dev_path,
        sample_n=args.n_dev,
        seed=args.seed + 2,
        excluding_ids=train_ids + test_ids,
    )

    ensure_disjoint_hotpot_splits({
        "train": hotpot_train_items,
        "dev": hotpot_dev_items,
        "test": hotpot_test_items,
    })

    print(
        f"Loaded {len(hotpot_train_items)} HotpotQA train items, "
        f"{len(hotpot_dev_items)} dev items, and {len(hotpot_test_items)} test items."
    )
    prime_context_cache(hotpot_train_items)
    prime_context_cache(hotpot_dev_items)
    prime_context_cache(hotpot_test_items)
    if args.evaluate_only:
        logger = JsonlLogger(
            f"logs/hotpotqa/openai/log_evaluate_only_mode_{args.mode}_iters_{args.iters}_k_topics_{args.k_topics}_clustering_sample_size_{args.clustering_sample_size}_cluster_desc_comprehensive_seed_prompt.jsonl"
        )
        reports = run_seed_prompt_evaluation(
            client,
            hotpot_train_items,
            hotpot_dev_items,
            hotpot_test_items,
            logger=logger,
            eval_workers=args.eval_workers,
        )
        print("\n=== SEED PROMPT TRAIN METRICS ===")
        print(json.dumps(reports["train"]["metrics"], indent=2))
        print("\n=== SEED PROMPT DEV METRICS ===")
        print(json.dumps(reports["dev"]["metrics"], indent=2))
        print("\n=== SEED PROMPT TEST METRICS ===")
        print(json.dumps(reports["test"]["metrics"], indent=2))
        return

    logger = JsonlLogger(
        f"logs/hotpotqa/openai/{args.optimizer_name}/log_{args.mode}_iters_{args.iters}_k_topics_{args.k_topics}_clustering_sample_size_{args.clustering_sample_size}_cluster_desc_comprehensive_optimizer_non_minimal_pcw_{args.prompt_complexity_weight:g}_pcu_{args.prompt_complexity_unit:g}.jsonl"
    )
    seed_prompt = make_seed_prompt()
    best_prompt = run_rpt(
        client,
        hotpot_train_items,
        hotpot_dev_items,
        hotpot_test_items,
        seed_prompt,
        logger=logger,
        iters=args.iters,
        mode=args.mode,
        test_every=args.test_every,
        k_topics=args.k_topics,
        clustering_sample_size=args.clustering_sample_size,
        seed=args.seed,
        k_reports=args.k_reports,
        eval_workers=args.eval_workers,
        optimizer_name=args.optimizer_name,
        prompt_complexity_weight=args.prompt_complexity_weight,
        prompt_complexity_unit=args.prompt_complexity_unit,
    )

    print("\n=== FINAL PROMPTPROGRAM ===")
    print(best_prompt.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
