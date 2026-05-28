from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from rpt.analysis.cluster_fusion import run_clusterfusion, ClusterFusionConfig, PartitionConfig
from rpt.common import JsonlLogger
from rpt.paths import XBRL_FORMULA_DATA_DIR


class TargetAnswer(BaseModel):
    reasoning: str = Field(default="", description="Chain of thought / reasoning / thinking process, detailed analysis and calculations.")
    answer: str = Field(description="Final numeric answer only.")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability the numeric answer is correct.")


class PromptProgram(BaseModel):
    system: str
    instruction: str
    enforce_json_only: bool = True
    # max_reasoning_sentences: int = 5

    def render(self, question: str, context: str = "") -> Tuple[str, str]:
        sys = self.system
        if context.strip():
            user = f"{self.instruction}\n\nContext:\n{context}\n\nQuestion:\n{question}\n"
        else:
            user = f"{self.instruction}\n\nQuestion:\n{question}\n"
        return sys, user


class EvalMetrics(BaseModel):
    n: int
    accuracy: float
    avg_confidence: float
    brier: float
    format_error_rate: float


class EvalItemTrace(BaseModel):
    idx: int
    question: str
    context: str
    original_context: str
    gold: str
    pred: Optional[str]
    correct: bool
    confidence: Optional[float]
    reasoning: Optional[str]
    error_type: str = ""


class FailureModeItem(BaseModel):
    label: str = Field(description="2–6 words, consistent across similar errors")
    definition: str = Field(description="Explanation of the failure mode")
    why: str = Field(description="Brief explanation for THIS example")
    basis: str = Field(description="Cite what in reasoning/evidence shows this")


class FailureCritique(BaseModel):
    failure_modes: List[FailureModeItem] = Field(default_factory=list)


class FailureModeTopic(BaseModel):
    name: str
    definition: str
    examples: List[str] = Field(default_factory=list)


class TraceInsights(BaseModel):
    failure_modes: List[FailureModeTopic] = Field(default_factory=list)


class EvalReport(BaseModel):
    iteration: int = 0
    prompt_program: Dict[str, Any]
    metrics: EvalMetrics
    # error_summary: Dict[str, int]
    insights: Optional[TraceInsights] = None


class PromptPatch(BaseModel):
    system: Optional[str] = Field(
        default=None,
        description="Complete replacement system message. Omit to leave unchanged.",
    )
    instruction: Optional[str] = Field(
        default=None,
        description=(
            "Complete replacement instruction text for the target model. "
            "This will not be merged; the target model will see this text verbatim, "
            "so it must be clean target-facing prompt text rather than optimizer-facing "
            "changelog text. Do not include meta-edit language like 'revised', "
            "'targeted', 'this patch', 'observed failures', 'merge these edits', "
            "'apply this patch', 'preserve all other behavior', or "
            "'replace the current rule'."
        ),
    )
    enforce_json_only: Optional[bool] = Field(default=None, description=("JSON-only output contract flag. Omit to leave unchanged."))
    # max_reasoning_sentences: Optional[int] = Field(
    #     default=None,
    #     description="Complete replacement value for max_reasoning_sentences. Omit to leave unchanged.",
    # )
    rationale: str = Field(
        description=(
            "Explanation for the optimizer log only. Put meta-edit explanations here, "
            "not inside system or instruction."
        )
    )


class LoopDecision(BaseModel):
    action: str = Field(description="Either 'patch' or 'stop'.")
    patch: Optional[PromptPatch] = None
    stop_reason: Optional[str] = None


PATCH_META_MARKERS = (
    "apply this patch",
    "apply these targeted",
    "apply the following",
    "merge/override",
    "current promptprogram",
    "keep all other invariants unchanged",
    "keep all other behavior unchanged",
    "preserve all other behavior",
    "insert start",
    "insert end",
    "replace the current rule",
    "observed failures",
    "existing instruction text",
    "[revised",
    "targeted fixes",
    "merge these edits",
)

PATCH_DECISION_MAX_RETRIES = 5
PARSE_COLLAPSE_FORMAT_ERROR_RATE_THRESHOLD = 0.5
PARSE_COLLAPSE_FORMAT_ERROR_RATE_DELTA_THRESHOLD = 0.25


# ----------------------------
# ACE-style formula preprocessing and evaluation
# ----------------------------

def load_jsonl(path: str, sample_n: Optional[int] = None, seed: int = 0) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if sample_n is not None and 0 < sample_n < len(items):
        rnd = random.Random(seed)
        rnd.shuffle(items)
        items = items[:sample_n]
    return items


def parse_context_and_question_formula(all_context: str) -> Tuple[str, str]:
    """Mirror ACE eval/finance/data_processor.py for the formula task."""
    if "Question: " in all_context and ". Answer:" in all_context:
        parts = all_context.split("Question: ", 1)
        question_part = parts[1]
        question_text = question_part.split(". Answer:")[0].strip()
        if question_text.startswith('"') and question_text.endswith('"'):
            question_text = question_text[1:-1]
        question_text += (
            " Your answer should be a plain floating point number, round to the nearest hundredth "
            "if necessary. Do the necessary conversions, for example 5 million should be 5000000.0."
        )
        return "", question_text
    return "", all_context


def process_formula_data(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    for item in raw_data:
        target = str(item.get("target", "")).strip()
        if "question" in item:
            context_text = str(item.get("context", "")).strip()
            question = str(item.get("question", "")).strip()
            others = dict(item.get("others") or {})
            others.setdefault("original_context", context_text)
            others.setdefault("task", "formula")
            others.setdefault("data_source", "finlora")
        else:
            original_context = str(item.get("context", ""))
            context_text, question = parse_context_and_question_formula(original_context)
            others = {
                "original_context": original_context,
                "task": "formula",
                "data_source": "finlora",
            }
        processed.append(
            {
                "context": context_text,
                "question": question,
                "target": target,
                "others": others,
            }
        )
    return processed


def formula_answer_is_correct(predicted: str, ground_truth: str) -> bool:
    try:
        predicted = predicted.replace(",", "")
        ground_truth = ground_truth.replace(",", "")
        return float(predicted) == float(ground_truth)
    except Exception:
        return predicted == ground_truth


# ----------------------------
# Critic
# ----------------------------

def critique_one_trace_with_gpt5(
    client: OpenAI,
    trace: Dict[str, Any],
    model: str = "gpt-5",
) -> FailureCritique:
    critic_system = f"""You are a strict evaluation critic for XBRL formula-construction failures.
You are given ONE QA trace with:
- question
- gold answer
- predicted answer
- model confidence
- model reasoning

Your job is to diagnose why the model produced the wrong answer.

Instructions:
1) Produce 1–3 failure_modes with:
   - label: 2–6 words, consistent across similar errors
   - definition: comprehensive explanation of the failure mode
    - why: brief, self-contained explanation for THIS example, e.g. "The question asked for the city's location relative to Rome, but the model returned the city name instead."
   - basis: cite what in reasoning/evidence shows this
2) Focus on actionable failure modes.
3) If you cannot identify a clear failure mode, return an empty list.
4) Output ONLY valid JSON matching the schema.
"""

    user_payload = {
        "question": trace.get("question"),
        "gold_answer": trace.get("gold"),
        "predicted_answer": trace.get("pred"),
        "confidence": trace.get("confidence"),
        "reasoning": trace.get("reasoning"),
    }

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": critic_system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        text_format=FailureCritique,
    )
    return resp.output_parsed


# ----------------------------
# OpenAI calls
# ----------------------------

def call_target_model(
    client: OpenAI,
    prompt: PromptProgram,
    question: str,
    context: str = "",
    target_model: str = "gpt-4.1",
) -> Tuple[Optional[TargetAnswer], str, str]:
    sys, user = prompt.render(question=question, context=context)
    try:
        resp = client.responses.parse(
            model=target_model,
            input=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            text_format=TargetAnswer,
            temperature=0.0,
        )
        return resp.output_parsed, resp.output_text, "ok"
    except Exception as e:
        return None, f"[PARSE_ERROR] {e}", "non_json"


def evaluate_single_item(
    client: OpenAI,
    prompt: PromptProgram,
    row: Dict[str, Any],
    idx: int,
    target_model: str,
) -> Dict[str, Any]:
    question = str(row.get("question", "")).strip()
    context = str(row.get("context", "")).strip()
    original_context = str(row.get("others", {}).get("original_context", "")).strip()
    gold = str(row.get("target", "")).strip()

    parsed, raw_text, err = call_target_model(
        client,
        prompt,
        question=question,
        context=context,
        target_model=target_model,
    )

    pred = str(parsed.answer).strip() if parsed else None
    confidence = float(parsed.confidence) if parsed else None
    reasoning = parsed.reasoning if parsed else None
    is_correct = formula_answer_is_correct(pred, gold) if pred is not None else False

    trace = EvalItemTrace(
        idx=idx,
        question=question,
        context=context,
        original_context=original_context,
        gold=gold,
        pred=pred,
        correct=is_correct,
        confidence=confidence,
        reasoning=reasoning,
        error_type="" if parsed and is_correct else ("wrong_answer" if parsed else err),
    )

    return {
        "idx": idx,
        "question": question,
        "context": context,
        "original_context": original_context,
        "gold": gold,
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
    }


# ----------------------------
# Tool: evaluate_prompt
# ----------------------------

def evaluate_prompt_tool(
    client: OpenAI,
    prompt: PromptProgram,
    items: List[Dict[str, Any]],
    logger: JsonlLogger,
    hard_k: int = 8, # should remove this later
    step: int = 0,
    mode: str = "dev",
    k_topics: int = 10,
    clustering_sample_size: int = 100,
    optimizer_name: str = "gpt-5",
    seed: int = 0,
    target_model: str = "gpt-4.1",
    max_workers: int = 20,
    ) -> Dict[str, Any]:
    
    traces: List[EvalItemTrace] = []
    failure_labels: List[Dict[str, Any]] = []
    fmt_errors = 0
    confs: List[float] = []
    brier_terms: List[float] = []
    correct_count = 0

    max_workers = max(1, min(max_workers, len(items) if items else 1))
    eval_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_single_item,
                client,
                prompt,
                row,
                idx,
                target_model,
            )
            for idx, row in enumerate(items)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {mode}"):
            eval_results.append(future.result())

    eval_results.sort(key=lambda item: item["idx"])
    incorrect_eval_results: List[Dict[str, Any]] = []

    for result in eval_results:
        idx = result["idx"]
        parsed = result["parsed"]
        trace = result["trace"]

        logger.log(
            "target_trace",
            step=step,
            payload={
                "idx": idx,
                "question": result["question"],
                "context": result["context"],
                "original_context": result["original_context"],
                "gold": result["gold"],
                "pred": parsed.answer if parsed else None,
                "confidence": float(parsed.confidence) if parsed else None,
                "reasoning": parsed.reasoning if parsed else None,
                "raw_text": result["raw_text"],
                "mode": mode,
            },
        )

        traces.append(trace)
        if parsed is None:
            fmt_errors += 1
            continue

        conf = float(parsed.confidence)
        correct_count += int(trace.correct)
        confs.append(conf)
        brier_terms.append((conf - (1.0 if trace.correct else 0.0)) ** 2)

        if mode == "train" and not trace.correct:
            incorrect_eval_results.append(result)

    if mode == "train" and incorrect_eval_results:
        critique_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    critique_single_trace,
                    client,
                    items[result["idx"]],
                    result["trace"],
                    optimizer_name,
                )
                for result in incorrect_eval_results
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Critiquing train errors"):
                critique_results.append(future.result())

        critique_results.sort(key=lambda item: item["idx"])
        for critique_result in critique_results:
            crit = critique_result["critique"]
            if crit is not None:
                failure_id_start = len(failure_labels)
                for fm in crit.failure_modes:
                    failure_labels.append(
                        {
                            "id": failure_id_start,
                            "text": f"{str(fm.label)}:{str(fm.definition)}",
                            "example": fm.why,
                            "impact": 1.0,
                        }
                    )
                    failure_id_start += 1

            logger.log(
                "item_critique",
                step=step,
                payload={
                    "idx": critique_result["idx"],
                    "row_id": critique_result["row_id"],
                    "question": critique_result["question"],
                    "critique": json.loads(crit.model_dump_json()) if crit else None,
                },
            )

    metrics = EvalMetrics(
        n=len(items),
        accuracy=(correct_count / len(items)) if items else 0.0,
        avg_confidence=(sum(confs) / len(confs)) if confs else 0.0,
        brier=(sum(brier_terms) / len(brier_terms)) if brier_terms else 1.0,
        format_error_rate=(fmt_errors / len(items)) if items else 0.0,
    )

    insights = None
    if failure_labels:
        domain_guidance = (
            "You will receive short failure-mode labels produced by an iterative prompt optimization method. "
            "Each record describes a recurring failure pattern in model behavior."
        )
        clustering_cfg = ClusterFusionConfig(
            k_topics=k_topics,
            partition=PartitionConfig(num_groups=max(2, 2 * k_topics), sample_size=clustering_sample_size, seed=seed, cosine_order=True),
            domain_guidance=domain_guidance,
            feature_context="failure modes",
            text_field="text",
            topic_desc_mode="comprehensive",
        )
        topics = run_clusterfusion(failure_labels, clustering_cfg, get_topics=True)
        logger.log("failure_mode_clusters", step, {"topics": topics})
        selected_topics = [t for t in topics if t.get("prevalence", 0.0) >= 0.10] or topics[: min(len(topics), 3)]
        for t in selected_topics:
            t.pop("prevalence", None)
            t.pop("topic_id", None)
       
        insights = TraceInsights(failure_modes=[FailureModeTopic(**t) for t in selected_topics])

    report = EvalReport(
        iteration=step,
        prompt_program=prompt.model_dump(),
        metrics=metrics,
        insights=insights,
    )
    return report.model_dump()


# ----------------------------
# RPT loop
# ----------------------------

def apply_patch(prompt: PromptProgram, patch: PromptPatch) -> PromptProgram:
    data = prompt.model_dump()
    ignored_patch_fields = {"rationale", "enforce_json_only"}
    for key, value in patch.model_dump(exclude_none=True).items():
        if key not in ignored_patch_fields:
            data[key] = value
    return PromptProgram(**data)


def prompt_complexity_chars(prompt_program: Optional[Dict[str, Any]]) -> int:
    if not prompt_program:
        return 0
    system_text = str(prompt_program.get("system", "") or "")
    instruction_text = str(prompt_program.get("instruction", "") or "")
    return len(system_text) + len(instruction_text)


def score_report(
    report: Dict[str, Any],
    w_correct: float = 0.5,
    w_brier: float = 0.05,
    prompt_complexity_weight: float = 0.0,
    prompt_complexity_unit: float = 1000.0,
) -> float:
    m = report["metrics"]
    prompt_complexity_penalty = 0.0
    if prompt_complexity_weight > 0.0:
        prompt_complexity_penalty = (
            prompt_complexity_weight
            * prompt_complexity_chars(report.get("prompt_program"))
            / prompt_complexity_unit
        )
    return (
        w_correct * float(m["accuracy"])
        - w_brier * float(m["brier"])
        - prompt_complexity_penalty
    )


def better_score(a: float, b: float, eps: float = 1e-12) -> bool:
    return a > b + eps


def detect_parse_collapse(
    train_report: Dict[str, Any],
    previous_train_report: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    current_format_error_rate = float(train_report["metrics"].get("format_error_rate", 0.0))
    previous_format_error_rate = (
        float(previous_train_report["metrics"].get("format_error_rate", 0.0))
        if previous_train_report is not None
        else None
    )
    reasons = []
    if current_format_error_rate >= PARSE_COLLAPSE_FORMAT_ERROR_RATE_THRESHOLD:
        reasons.append(
            f"format_error_rate {current_format_error_rate:.3f} >= "
            f"{PARSE_COLLAPSE_FORMAT_ERROR_RATE_THRESHOLD:.3f}"
        )
    if previous_format_error_rate is not None:
        delta = current_format_error_rate - previous_format_error_rate
        if delta >= PARSE_COLLAPSE_FORMAT_ERROR_RATE_DELTA_THRESHOLD:
            reasons.append(
                f"format_error_rate increased by {delta:.3f} from "
                f"{previous_format_error_rate:.3f} to {current_format_error_rate:.3f}"
            )
    if not reasons:
        return None
    return {
        "current_format_error_rate": current_format_error_rate,
        "previous_format_error_rate": previous_format_error_rate,
        "reason": "; ".join(reasons),
    }


def validate_prompt_patch(patch: PromptPatch) -> Tuple[bool, Optional[str]]:
    for field_name in ("system", "instruction"):
        value = getattr(patch, field_name)
        if not value:
            continue
        lowered = value.lower()
        for marker in PATCH_META_MARKERS:
            if marker in lowered:
                return (
                    False,
                    f"{field_name} contains optimizer-facing/meta-edit text marker: {marker!r}. "
                    "Return clean target-facing prompt text only.",
                )
    return True, None


def build_current_summary(
    history_reports: List[Dict[str, Any]],
    best_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    current_report = history_reports[-1]
    current_metrics = current_report["metrics"]
    current_score = score_report(current_report)
    prev_metrics = history_reports[-2]["metrics"] if len(history_reports) >= 2 else None
    prev_score = score_report(history_reports[-2]) if len(history_reports) >= 2 else None
    best_metrics = best_report["metrics"] if best_report else None
    best_score = score_report(best_report) if best_report else None

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
        "score": current_score,
    }

    if prev_metrics is not None and prev_score is not None:
        summary["delta_vs_previous"] = {
            "accuracy": float(current_metrics["accuracy"] - prev_metrics["accuracy"]),
            "brier": float(current_metrics["brier"] - prev_metrics["brier"]),
            "format_error_rate": float(current_metrics["format_error_rate"] - prev_metrics["format_error_rate"]),
            "score": float(current_score - prev_score),
        }
        summary["did_last_patch_improve_train"] = bool(
            current_score > prev_score + 1e-12
        )

    if best_metrics is not None and best_score is not None:
        summary["delta_vs_best"] = {
            "accuracy": float(current_metrics["accuracy"] - best_metrics["accuracy"]),
            "brier": float(current_metrics["brier"] - best_metrics["brier"]),
            "format_error_rate": float(current_metrics["format_error_rate"] - best_metrics["format_error_rate"]),
            "score": float(current_score - best_score),
        }

    return summary


OPTIMIZER_INSTRUCTIONS = (
        "You are the Reflective Prompt Tuning (RPT) controller.\n\n"
        "Your goal is to iteratively improve a PromptProgram for a finance formula QA task.\n\n"
        "At each iteration you must:\n"
        "  (1) Call `evaluate_prompt` exactly once on the CURRENT PromptProgram.\n"
        "  (2) Read the returned evaluation report with insights.\n"
        "  (3) Output either a PATCH or STOP.\n\n"
        "Optimization target:\n"
        "  - Primary: improve numeric accuracy on the training split.\n"
        "  - Secondary: improve calibration (lower Brier / reduce overconfidence) without hurting accuracy.\n\n"
        "Decision guidance:\n"
        "  - When current_summary is provided, use it as the primary decision signal, especially current_summary.metrics and any deltas vs previous/best.\n"
        "  - Use history only to detect trajectory, regressions, and previously ineffective edits.\n"
        "Patch constraints:\n"
        "  - A patch directly edits one or more PromptProgram fields for the next iteration; for system/instruction, write the revised prompt text to use next, not how to edit it.\n"
        "  - Edits should be targeted to the failure modes, and designed to address their underlying issues with concrete guidance.\n"
        "  - Do not reduce failure modes to a short generic instruction; provide actionable steps.\n"
        "  - Prefer revising, merging, deleting, or reorganizing existing instructions over adding new broad rules.\n"
        "  - If the prompt becomes long, conflicting, or brittle, prefer a compact stable replacement that preserves the essential output contract, calculation discipline, final-answer formatting, and confidence rule.\n"
        "  - If detailed diagnostics would make target outputs less parseable, use a concise fallback instruction instead of layering more audit text.\n"
        "  - Keep the output contract stable (JSON schema and required fields).\n"
        "  - Avoid redundant or conflicting rules; consolidate instructions when possible.\n"
        "Stop condition:\n"
        "  - Output STOP if training-set performance has plateaued or further edits are unlikely to help.\n\n"
        "Hard rule:\n"
        "  - Do NOT propose a PATCH or STOP decision before calling `evaluate_prompt` and receiving its result."
)

def run_rpt(
    client: OpenAI,
    train_items: List[Dict[str, Any]],
    dev_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    seed_prompt: PromptProgram,
    logger: JsonlLogger,
    iters: int = 5,
    mode: str = "last_report",
    test_every: int = 5,
    clustering_sample_size: int = 100,
    k_topics: int = 10,
    k_reports: int = 5,
    seed: int = 0,
    eval_workers: int = 20,
    optimizer_name: str = "gpt-5",
    target_model: str = "gpt-4.1",
    prompt_complexity_weight: float = 0.0,
    prompt_complexity_unit: float = 1000.0,
    ) -> PromptProgram:
    tools = [{
        "type": "function",
        "name": "evaluate_prompt",
        "description": (
            "Evaluate a PromptProgram on the XBRL formula-construction training split using the target model. "
            "Returns an evaluation report with performance metrics and analysis insights."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt_program": {
                    "type": "object",
                    "properties": {
                        "system": {"type": "string"},
                        "instruction": {"type": "string"},
                        "enforce_json_only": {"type": "boolean"},
                        # "max_reasoning_sentences": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                },
            },
            "required": ["prompt_program"],
        },
    }]

    optimizer_instructions = OPTIMIZER_INSTRUCTIONS

    prompt = seed_prompt
    train_history_reports: List[Dict[str, Any]] = []
    best_train_report: Optional[Dict[str, Any]] = None
    best_prompt = prompt
    best_dev_report: Optional[Dict[str, Any]] = None
    best_dev_score = float("-inf")

    for t in range(iters):
        step = t + 1
        if (step % test_every) == 0 or step == 1:
            test_report_json = evaluate_prompt_tool(
                client,
                prompt,
                test_items,
                logger=logger,
                step=step,
                mode="test",
                max_workers=eval_workers,
                target_model=target_model,
            )
            logger.log("test_stats", step=step, payload=test_report_json)

        logger.log("iter_prompt", step=step, payload=prompt.model_dump())
        print(f"\n=== RPT Iteration {step}/{iters} (mode={mode}) ===")

        input_list: List[Dict[str, Any]] = [{
            "role": "user",
            "content": (
                f"Iteration {step}/{iters}\n\n"
                "Call `evaluate_prompt` on the CURRENT PromptProgram below.\n"
                "Use this JSON as evaluate_prompt.prompt_program:\n\n"
                f"{prompt.model_dump_json(indent=2)}\n"
            ),
        }]

        response = client.responses.parse(
            model=optimizer_name,
            input=input_list,
            instructions=optimizer_instructions,
            tools=tools,
        )
        input_list += response.output

        got_tool_call = False
        for item in response.output:
            if getattr(item, "type", None) != "function_call" or item.name != "evaluate_prompt":
                continue
            got_tool_call = True
            train_report = evaluate_prompt_tool(
                client,
                prompt,
                train_items,
                logger=logger,
                step=step,
                k_topics=k_topics,
                clustering_sample_size=clustering_sample_size,
                optimizer_name=optimizer_name,
                seed=seed,
                mode="train",
                max_workers=eval_workers,
                target_model=target_model,
            )
            logger.log("train_stats", step=step, payload=train_report)
            train_history_reports.append(train_report)

            tool_output = {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(train_report, ensure_ascii=False),
            }
            input_list.append(tool_output)

            train_score = score_report(
                train_report,
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
                best_train_report = train_report

            dev_report = evaluate_prompt_tool(
                client,
                prompt,
                dev_items,
                logger=logger,
                step=step,
                mode="dev",
                max_workers=eval_workers,
                target_model=target_model,
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
                    "train_metrics": train_report["metrics"],
                    "prompt_program": train_report.get("prompt_program", prompt.model_dump()),
                })

        if not got_tool_call:
            raise RuntimeError("Optimizer model did not call evaluate_prompt.")

        previous_train_report = train_history_reports[-2] if len(train_history_reports) >= 2 else None
        parse_collapse = detect_parse_collapse(train_history_reports[-1], previous_train_report)
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

        retry_feedback = []
        if parse_collapse:
            logger.log("parse_collapse_detected", step=step, payload=parse_collapse)
            retry_feedback.append(
                "Hard failure signal: the current prompt caused a parse collapse on the training split "
                f"({parse_collapse['reason']}). Return a corrective PATCH that restores clean parseable "
                "target outputs, or STOP if no safe correction exists."
            )

        for decision_attempt in range(1, PATCH_DECISION_MAX_RETRIES + 1):
            decision_input = list(input_list)
            decision_content = (
                f"{mode_hint}\n"
                "Using the most recent `function_call_output` evaluation report above, now decide whether to STOP or output a PATCH.\n"
                "Return a LoopDecision JSON with fields:\n"
                "  - action: 'patch' or 'stop'\n"
                "  - patch (only if action='patch')\n"
                "  - stop_reason (only if action='stop')\n\n"
            )
            if retry_feedback:
                decision_content += (
                    "Guardrail feedback from previous decision attempts:\n"
                    + "\n".join(f"  - {feedback}" for feedback in retry_feedback)
                    + "\n\n"
                )
            decision_content += json.dumps(decision_payload, ensure_ascii=False, indent=2)

            decision_input.append({
                "role": "user",
                "content": decision_content,
            })

            decision_resp = client.responses.parse(
                model=optimizer_name,
                input=decision_input,
                instructions=optimizer_instructions,
                text_format=LoopDecision,
            )
            decision: LoopDecision = decision_resp.output_parsed
            if decision_attempt > 1 or retry_feedback:
                logger.log(
                    "decision_attempt",
                    step=step,
                    payload={
                        "attempt": decision_attempt,
                        "decision": decision.model_dump(),
                        "guardrail_feedback": retry_feedback,
                    },
                )

            if decision.action == "stop":
                break
            if decision.action != "patch" or decision.patch is None:
                retry_feedback.append("Invalid LoopDecision: action must be 'patch' with patch, or 'stop'.")
                continue

            patch_ok, patch_error = validate_prompt_patch(decision.patch)
            if patch_ok:
                break
            logger.log(
                "patch_rejected_meta_text",
                step=step,
                payload={
                    "attempt": decision_attempt,
                    "error": patch_error,
                    "patch": decision.patch.model_dump(),
                },
            )
            retry_feedback.append(patch_error or "Patch contained invalid optimizer-facing text.")
        else:
            decision = LoopDecision(
                action="stop",
                stop_reason=(
                    "Optimizer could not produce a valid clean patch after "
                    f"{PATCH_DECISION_MAX_RETRIES} attempts."
                ),
            )
            logger.log("decision_forced_stop", step=step, payload=decision.model_dump())

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

    final_test_report_json = evaluate_prompt_tool(
        client,
        best_prompt,
        test_items,
        logger=logger,
        step=iters,
        mode="test",
        max_workers=eval_workers,
        target_model=target_model,
    )
    logger.log("final_test_stats", step=iters, payload=final_test_report_json)
    return best_prompt


# ----------------------------
# Seed prompt
# ----------------------------

def make_seed_prompt() -> PromptProgram:
    return PromptProgram(
        system=(
            "You are an analysis expert tasked with answering questions using your knowledge."
            # "Your task is to analyze the XBRL context and provide an accurate and very concise answer to the question."
            # "You are a careful XBRL formula-construction assistant. "
            # "Map the requested financial ratio to the correct US-GAAP tags from the provided XBRL context."
        ),
        instruction=(
            "**Instructions:**\n"
            "- Show your reasoning step-by-step\n"
            "- Be concise but thorough in your analysis\n"
            "- Double-check your calculations and logic before providing the final answer\n\n"
            "Your output should be a JSON with fields:\n"
            "- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations\n"
            "- answer: your concise final answer.\n"
            "- confidence: a number in [0,1] representing your confidence in the final answer.\n"

            # "Solve the finance formula problem carefully. Return JSON with fields: answer, confidence, justification.\n"
            # "- answer: the final plain floating point number only; no units, currency symbols, commas, or extra prose.\n"
            # "- confidence: number in [0,1].\n"
            # "- justification: brief explanation of the formula, substitutions, and any unit conversions.\n"
        ),
        enforce_json_only=True,
        # max_reasoning_sentences=5,
    )


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=500, help="Number of sampled train examples for optimizer feedback.")
    ap.add_argument("--n_dev", type=int, default=300, help="Number of sampled dev examples for model selection.")
    ap.add_argument("--n_test", type=int, default=200, help="Number of sampled test examples.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--mode", type=str, default="all_reports", choices=["all_reports", "history_summary", "last_report", "last_k_reports"])
    ap.add_argument("--k_topics", type=int, default=10)
    ap.add_argument("--k_reports", type=int, default=5)
    ap.add_argument("--clustering_sample_size", type=int, default=100)
    ap.add_argument("--test_every", type=int, default=5)
    ap.add_argument("--eval_workers", type=int, default=20, help="Number of worker threads for dev/test evaluation.")
    ap.add_argument(
        "--optimizer_name",
        type=str,
        default="gpt-5",
        help="Model name used for optimizer/controller and critique calls.",
    )
    ap.add_argument("--target_model", type=str, default="gpt-4.1", help="Target model used for prompt evaluation.")
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
    args = ap.parse_args()

    random.seed(args.seed)
    client = OpenAI()


    raw_train = load_jsonl(str(XBRL_FORMULA_DATA_DIR / "train.jsonl"), args.n_train, args.seed)
    raw_dev = load_jsonl(str(XBRL_FORMULA_DATA_DIR / "val.jsonl"), args.n_dev, args.seed)
    raw_test = load_jsonl(str(XBRL_FORMULA_DATA_DIR / "test.jsonl"), args.n_test, args.seed)
    train_items = process_formula_data(raw_train)
    dev_items = process_formula_data(raw_dev)
    test_items = process_formula_data(raw_test)

    print(
        f"Loaded {len(train_items)} XBRL formula train items, "
        f"{len(dev_items)} dev items, and {len(test_items)} test items."
    )

    logger = JsonlLogger(
        os.path.join(
            "logs",
            "xbrl_formula",
            "openai",
            args.optimizer_name,
            (
                f"log_{args.mode}_iters_{args.iters}_train_{args.n_train}_dev_{args.n_dev}_"
                f"test_{args.n_test}_seed_{args.seed}_k_topics_{args.k_topics}_"
                f"cluster_desc_comprehensive_optimizer_non_minimal_"
                f"pcw_{args.prompt_complexity_weight:g}_pcu_{args.prompt_complexity_unit:g}.jsonl"
            ),
        )
    )
    seed_prompt = make_seed_prompt()
    best_prompt = run_rpt(
        client,
        train_items,
        dev_items,
        test_items,
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
        target_model=args.target_model,
        prompt_complexity_weight=args.prompt_complexity_weight,
        prompt_complexity_unit=args.prompt_complexity_unit,
    )

    print("\n=== FINAL PROMPTPROGRAM ===")
    print(best_prompt.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
