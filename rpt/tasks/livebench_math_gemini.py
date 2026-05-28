"""
Reflective Prompt Tuning (RPT) for LiveBench Math with Gemini on Vertex AI.

This script keeps the target/eval model path identical to `rpt.tasks.livebench_math`
and only swaps the optimizer/controller + critic path to Gemini. Unlike the
OpenAI version, the optimizer tool loop is implemented with the native
`google-genai` SDK and an explicit `evaluate_prompt` function declaration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from tqdm import tqdm

from rpt.tasks import livebench_math as base
from rpt.gemini_utils import (
    build_cleaned_log_path,
    high_thinking_config,
    make_gemini_client,
    parse_gemini_structured_response,
    write_cleaned_log as _write_cleaned_log,
)

def build_run_log_path(
    args: argparse.Namespace,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
) -> str:
    return os.path.join(
        "logs",
        "livebench_math",
        "gemini",
        args.optimizer_name,
        (
            f"log_{args.mode}_iters_{args.iters}_train_{n_train}_val_{n_val}"
            f"_test_{n_test}_split_seed_{args.split_seed}_seed_{args.seed}"
            f"_k_topics_{args.k_topics}_cluster_desc_comprehensive"
            f"_optimizer_non_minimal"
            f"_pcw_{args.prompt_complexity_weight:g}_pcu_{args.prompt_complexity_unit:g}.jsonl"
        ),
    )


def write_cleaned_log(log_path: str, cleaned_path: Optional[str] = None) -> str:
    return _write_cleaned_log(
        log_path,
        cleaned_path,
        required_events={"iter_prompt", "train_stats", "val_stats", "decision"},
    )


def load_resume_state(
    log_path: str,
    seed_prompt: base.PromptProgram,
    *,
    prompt_complexity_weight: float,
    prompt_complexity_unit: float,
) -> Dict[str, Any]:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Resume log not found: {log_path}")

    iter_prompts: Dict[int, base.PromptProgram] = {}
    train_reports: Dict[int, Dict[str, Any]] = {}
    val_reports: Dict[int, Dict[str, Any]] = {}
    decisions: Dict[int, base.LoopDecision] = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse {log_path}:{line_no}: {exc}") from exc

            step = rec.get("step")
            if not isinstance(step, int) or step <= 0:
                continue
            payload = rec.get("payload") or {}
            event = rec.get("event")

            if event == "iter_prompt":
                iter_prompts[step] = base.PromptProgram.model_validate(payload)
            elif event == "train_stats":
                train_reports[step] = payload
            elif event == "val_stats":
                val_reports[step] = payload
            elif event == "decision":
                decisions[step] = base.LoopDecision.model_validate(payload)

    completed_steps = 0
    for step in range(1, max(iter_prompts.keys(), default=0) + 1):
        if (
            step not in iter_prompts
            or step not in train_reports
            or step not in val_reports
            or step not in decisions
        ):
            break
        completed_steps = step
        if decisions[step].action == "stop":
            break

    prompt = seed_prompt
    train_history_reports: List[Dict[str, Any]] = []
    best_train_report: Optional[Dict[str, Any]] = None
    best_val_report: Optional[Dict[str, Any]] = None
    best_val_score = float("-inf")
    best_prompt = seed_prompt
    stopped = False

    for step in range(1, completed_steps + 1):
        prompt = iter_prompts[step]
        train_report = train_reports[step]
        train_history_reports.append(train_report)

        train_score = base.score_report(
            train_report,
            prompt_complexity_weight=prompt_complexity_weight,
            prompt_complexity_unit=prompt_complexity_unit,
        )
        if best_train_report is None or base.better_score(
            train_score,
            base.score_report(
                best_train_report,
                prompt_complexity_weight=prompt_complexity_weight,
                prompt_complexity_unit=prompt_complexity_unit,
            ),
        ):
            best_train_report = train_report

        val_report = val_reports[step]
        val_score = base.score_report(
            val_report,
            prompt_complexity_weight=prompt_complexity_weight,
            prompt_complexity_unit=prompt_complexity_unit,
        )
        if base.better_score(val_score, best_val_score):
            best_val_score = val_score
            best_val_report = val_report
            best_prompt = prompt

        decision = decisions[step]
        if decision.action == "patch":
            if decision.patch is None:
                raise ValueError(f"Step {step} in {log_path} has action='patch' with no patch payload.")
            prompt = base.apply_patch(prompt, decision.patch)
        elif decision.action == "stop":
            stopped = True
            break
        else:
            raise ValueError(f"Step {step} in {log_path} has unexpected decision action: {decision.action!r}")

    return {
        "completed_steps": completed_steps,
        "next_step": completed_steps + 1,
        "prompt": prompt,
        "train_history_reports": train_history_reports,
        "best_train_report": best_train_report,
        "best_val_report": best_val_report,
        "best_val_score": best_val_score,
        "best_prompt": best_prompt,
        "stopped": stopped,
    }


def critique_one_trace_with_gemini(
    optimizer_client: genai.Client,
    trace: Dict[str, Any],
    model: str
) -> base.FailureCritique:
    critic_system = f"""You are a strict evaluation critic for math failures.
You are given one failed model attempt with:
- task metadata
- question
- gold answer
- predicted answer
- model confidence
- model reasoning

Your goal is to diagnose why the model failed.

Instructions:
1) Produce 1-3 failure_modes with:
   - label: 2-6 words, consistent across similar errors
   - definition: comprehensive explanation of the failure mode
   - why: brief, self-contained explanation for THIS example
   - basis: cite what in the trace shows this
2) Make labels concrete and clusterable.
3) If you cannot identify a clear failure mode, return an empty list.
4) Output only valid JSON matching the schema.
"""

    response = optimizer_client.models.generate_content(
        model=model,
        contents=json.dumps(trace, ensure_ascii=False),
        config=genai_types.GenerateContentConfig(
            system_instruction=critic_system,
            response_mime_type="application/json",
            response_schema=base.FailureCritique,
            # thinking_config=high_thinking_config(),
            temperature=0,
        ),
    )
    return parse_gemini_structured_response(response, base.FailureCritique)


def critique_single_trace_gemini(
    optimizer_client: genai.Client,
    row: Dict[str, Any],
    trace: base.EvalItemTrace,
    optimizer_name: str,
) -> Dict[str, Any]:
    crit = critique_one_trace_with_gemini(
        optimizer_client,
        trace.model_dump(),
        model=optimizer_name,
    )
    return {
        "idx": trace.idx,
        "row_id": row.get("id"),
        "question": row.get("question", ""),
        "critique": crit,
        "impact": 1.0 - float(trace.score),
    }


def evaluate_prompt_tool_gemini(
    target_client: OpenAI,
    optimizer_client: genai.Client,
    prompt: base.PromptProgram,
    items: List[Dict[str, Any]],
    logger: base.JsonlLogger,
    *,
    target_model: str,
    optimizer_name: str,
    step: int = 0,
    mode: str = "train",
    k_topics: int = 10,
    clustering_sample_size: int = 100,
    seed: int = 0,
    max_workers: int = 20,
    critique_workers: int = 1,
    enable_critiques: bool = True,
) -> Dict[str, Any]:
    traces: List[base.EvalItemTrace] = []
    failure_labels: List[Dict[str, Any]] = []
    format_errors = 0
    confs: List[float] = []
    brier_terms: List[float] = []
    eval_results: List[Dict[str, Any]] = []
    incorrect_eval_results: List[Dict[str, Any]] = []
    max_workers = max(1, min(max_workers, len(items) if items else 1))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                base.evaluate_single_item,
                target_client,
                prompt,
                row,
                idx,
                target_model,
            )
            for idx, row in enumerate(items)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {mode} items"):
            eval_results.append(future.result())

    eval_results.sort(key=lambda item: item["idx"])

    for result in eval_results:
        row = result["row"]
        parsed = result["parsed"]
        trace = result["trace"]
        traces.append(trace)
        logger.log(
            "target_trace",
            step,
            {
                "idx": trace.idx,
                "row_id": trace.row_id,
                "task": trace.task,
                "subtask": trace.subtask,
                "question": trace.question,
                "gold": trace.gold,
                "pred": parsed.answer if parsed else None,
                "confidence": float(parsed.confidence) if parsed else None,
                "reasoning": parsed.reasoning if parsed else None,
                "error": result["err"],
                "score": float(trace.score),
                "full_credit": trace.full_credit,
            },
        )

        if parsed is None:
            format_errors += 1
            continue

        conf = float(trace.confidence or 0.0)
        confs.append(conf)
        brier_terms.append((conf - float(trace.score)) ** 2)

        if mode == "train" and enable_critiques and trace.score < 1.0 - 1e-12:
            incorrect_eval_results.append(result)

    if mode == "train" and enable_critiques and incorrect_eval_results:
        critique_results: List[Dict[str, Any]] = []
        actual_critique_workers = max(1, min(critique_workers, len(incorrect_eval_results)))
        with ThreadPoolExecutor(max_workers=actual_critique_workers) as executor:
            futures = [
                executor.submit(
                    critique_single_trace_gemini,
                    optimizer_client,
                    result["row"],
                    result["trace"],
                    optimizer_name,
                )
                for result in incorrect_eval_results
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Critiquing train errors"):
                try:
                    critique_results.append(future.result())
                except Exception as exc:
                    critique_results.append(
                        {
                            "idx": -1,
                            "row_id": None,
                            "question": "",
                            "critique": None,
                            "impact": 0.0,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

        critique_results.sort(key=lambda item: item["idx"])
        for critique_result in critique_results:
            crit = critique_result["critique"]
            if critique_result.get("error"):
                logger.log(
                    "item_critique_error",
                    step,
                    {
                        "idx": critique_result["idx"],
                        "row_id": critique_result["row_id"],
                        "question": critique_result["question"],
                        "error": critique_result["error"],
                    },
                )
                continue
            logger.log(
                "item_critique",
                step,
                {
                    "idx": critique_result["idx"],
                    "row_id": critique_result["row_id"],
                    "question": critique_result["question"],
                    "critique": crit.model_dump(),
                },
            )
            start_id = len(failure_labels)
            for fm in crit.failure_modes:
                failure_labels.append(
                    {
                        "id": start_id,
                        "text": f"{fm.label}: {fm.definition}",
                        "example": fm.why,
                        "impact": critique_result["impact"],
                    }
                )
                start_id += 1

    n = len(traces)
    em = sum(float(trace.score) for trace in traces) / max(1, n)
    per_task_scores = base.collect_livebench_math_task_scores(traces)
    avg_conf = sum(confs) / max(1, len(confs))
    brier = sum(brier_terms) / max(1, len(brier_terms))
    fmt_rate = format_errors / max(1, n)

    report = base.EvalReport(
        iteration=step,
        prompt_program=prompt.model_dump(),
        metrics=base.EvalMetrics(
            n=n,
            task_score=em,
            avg_confidence=avg_conf,
            brier=brier,
            format_error_rate=fmt_rate,
            per_task_scores=per_task_scores,
        ),
        insights=None,
    )

    if mode in {"val", "test"}:
        return report.model_dump()

    if base.HAS_CLUSTER_FUSION and failure_labels:
        clustering_cfg = base.ClusterFusionConfig(
            k_topics=k_topics,
            partition=base.PartitionConfig(
                num_groups=max(2, 2 * k_topics),
                sample_size=clustering_sample_size,
                seed=seed,
                cosine_order=True,
            ),
            domain_guidance=(
                "You will receive short failure-mode labels produced by an iterative prompt optimization method. "
                "Each record describes a recurring failure pattern in model behavior."
            ),
            feature_context="failure modes",
            text_field="text",
            topic_desc_mode="comprehensive",
        )
        topics = base.run_clusterfusion(failure_labels, clustering_cfg, get_topics=True)
        logger.log("failure_mode_clusters", step, {"topics": topics})
        selected_topics = [t for t in topics if t.get("prevalence", 0.0) >= 0.10] or topics[: min(len(topics), 3)]
        for topic in selected_topics:
            topic.pop("prevalence", None)
            topic.pop("topic_id", None)
        report.insights = base.TraceInsights(
            failure_modes=[base.FailureModeTopic(**topic) for topic in selected_topics]
        )
    elif failure_labels:
        examples = [item["text"] for item in failure_labels[: min(5, len(failure_labels))]]
        report.insights = base.TraceInsights(
            failure_modes=[
                base.FailureModeTopic(
                    name="uncategorized failures",
                    definition="Representative failure labels collected without clustering.",
                    examples=examples,
                )
            ]
        )

    return report.model_dump()


def make_evaluate_prompt_tool() -> genai_types.Tool:
    return genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name="evaluate_prompt",
                description=(
                    "Evaluate the current PromptProgram on the LiveBench-math training split "
                    "using the target model. Returns an evaluation report with metrics and failure insights."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "prompt_program": {
                            "type": "OBJECT",
                            "description": (
                                "The full PromptProgram to evaluate on the target model. "
                                "Fields: system, instruction, enforce_json_only."
                            ),
                            "properties": {
                                "system": {"type": "STRING"},
                                "instruction": {"type": "STRING"},
                                "enforce_json_only": {"type": "BOOLEAN"},
                            },
                        }
                    },
                    "required": ["prompt_program"],
                },
            )
        ]
    )


def request_evaluate_prompt_call(
    optimizer_client: genai.Client,
    *,
    optimizer_name: str,
    optimizer_instructions: str,
    prompt: base.PromptProgram,
    step: int,
    iters: int,
) -> Any:
    user_text = (
        f"Iteration {step}/{iters}\n\n"
        "Call `evaluate_prompt` on the CURRENT PromptProgram below.\n"
        "Use this JSON as evaluate_prompt.prompt_program:\n\n"
        f"{prompt.model_dump_json(indent=2)}"
    )
    return optimizer_client.models.generate_content(
        model=optimizer_name,
        contents=user_text,
        config=genai_types.GenerateContentConfig(
            system_instruction=optimizer_instructions,
            tools=[make_evaluate_prompt_tool()],
            tool_config=genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["evaluate_prompt"],
                )
            ),
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            thinking_config=high_thinking_config(),
            temperature=0,
        ),
    )


def request_loop_decision(
    optimizer_client: genai.Client,
    *,
    optimizer_name: str,
    optimizer_instructions: str,
    first_response: Any,
    train_report: Dict[str, Any],
    decision_payload: Dict[str, Any],
    mode_hint: str,
) -> base.LoopDecision:
    function_response_part = genai_types.Part.from_function_response(
        name="evaluate_prompt",
        response={"report": train_report},
    )
    decision_user = (
        f"{mode_hint}\n"
        "Using the most recent function_call_output evaluation report above, now decide whether to STOP or output a PATCH.\n"
        "Return a LoopDecision JSON with fields:\n"
        "  - action: 'patch' or 'stop'\n"
        "  - patch (only if action='patch')\n"
        "  - stop_reason (only if action='stop')\n\n"
        f"{json.dumps(decision_payload, ensure_ascii=False, indent=2)}"
    )
    contents: List[Any] = [first_response.candidates[0].content, function_response_part, decision_user]
    response = optimizer_client.models.generate_content(
        model=optimizer_name,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=optimizer_instructions,
            response_mime_type="application/json",
            response_schema=base.LoopDecision,
            thinking_config=high_thinking_config(),
            temperature=0,
        ),
    )
    return parse_gemini_structured_response(response, base.LoopDecision)


def run_rpt_gemini(
    target_client: OpenAI,
    optimizer_client: genai.Client,
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    seed_prompt: base.PromptProgram,
    logger: base.JsonlLogger,
    *,
    target_model: str,
    optimizer_name: str,
    iters: int = 5,
    mode: str = "all_reports",
    test_every: int = 5,
    clustering_sample_size: int = 100,
    k_topics: int = 10,
    k_reports: int = 5,
    seed: int = 0,
    eval_workers: int = 20,
    critique_workers: int = 1,
    prompt_complexity_weight: float = 0.0,
    prompt_complexity_unit: float = 1000.0,
    resume_state: Optional[Dict[str, Any]] = None,
) -> base.PromptProgram:
    optimizer_instructions = base.OPTIMIZER_INSTRUCTIONS
    prompt = seed_prompt
    train_history_reports: List[Dict[str, Any]] = []
    best_train_report: Optional[Dict[str, Any]] = None
    best_prompt = prompt
    best_val_report: Optional[Dict[str, Any]] = None
    best_val_score = float("-inf")
    completed_steps = 0
    skip_optimization_loop = False

    if resume_state is not None:
        prompt = resume_state["prompt"]
        train_history_reports = list(resume_state["train_history_reports"])
        best_train_report = resume_state["best_train_report"]
        best_prompt = resume_state["best_prompt"]
        best_val_report = resume_state["best_val_report"]
        best_val_score = float(resume_state["best_val_score"])
        completed_steps = int(resume_state["completed_steps"])
        print(
            f"Resuming from iteration {completed_steps + 1}/{iters} "
            f"after {completed_steps} completed iteration(s)."
        )
        if resume_state.get("stopped"):
            print(f"Resume log already contains a STOP decision at iteration {completed_steps}.")
            skip_optimization_loop = True
        elif completed_steps >= iters:
            print("Resume log already covers all requested iterations.")
            skip_optimization_loop = True

    for t in range(completed_steps, iters) if not skip_optimization_loop else range(0):
        step = t + 1

        if step % test_every == 0: # or step == 1:
            test_report = base.evaluate_prompt_tool(
                target_client,
                prompt,
                test_items,
                logger=logger,
                target_model=target_model,
                step=step,
                mode="test",
                max_workers=eval_workers,
                enable_critiques=False,
            )
            logger.log("test_stats", step, test_report)

        logger.log("iter_prompt", step, prompt.model_dump())
        print(f"\n=== RPT Iteration {step}/{iters} (mode={mode}) ===")

        tool_response = request_evaluate_prompt_call(
            optimizer_client,
            optimizer_name=optimizer_name,
            optimizer_instructions=optimizer_instructions,
            prompt=prompt,
            step=step,
            iters=iters,
        )
        function_calls = getattr(tool_response, "function_calls", None) or []
        evaluate_calls = [fc for fc in function_calls if fc.name == "evaluate_prompt"]
        if not evaluate_calls:
            raise RuntimeError("Optimizer model did not call evaluate_prompt. Fix the controller prompt or tool schema.")

        train_report = evaluate_prompt_tool_gemini(
            target_client,
            optimizer_client,
            prompt,
            train_items,
            logger=logger,
            target_model=target_model,
            optimizer_name=optimizer_name,
            step=step,
            mode="train",
            k_topics=k_topics,
            clustering_sample_size=clustering_sample_size,
            seed=seed,
            max_workers=eval_workers,
            critique_workers=critique_workers,
        )
        logger.log("train_stats", step, train_report)
        train_history_reports.append(train_report)

        train_score = base.score_report(
            train_report,
            prompt_complexity_weight=prompt_complexity_weight,
            prompt_complexity_unit=prompt_complexity_unit,
        )
        if best_train_report is None or base.better_score(
            train_score,
            base.score_report(
                best_train_report,
                prompt_complexity_weight=prompt_complexity_weight,
                prompt_complexity_unit=prompt_complexity_unit,
            ),
        ):
            best_train_report = train_report

        val_report = base.evaluate_prompt_tool(
            target_client,
            prompt,
            val_items,
            logger=logger,
            target_model=target_model,
            step=step,
            mode="val",
            max_workers=eval_workers,
            enable_critiques=False,
        )
        logger.log("val_stats", step, val_report)

        val_score = base.score_report(
            val_report,
            prompt_complexity_weight=prompt_complexity_weight,
            prompt_complexity_unit=prompt_complexity_unit,
        )
        if base.better_score(val_score, best_val_score):
            best_val_score = val_score
            best_val_report = val_report
            best_prompt = prompt
            logger.log(
                "best_update",
                step,
                {
                    "selection_split": "val",
                    "score": best_val_score,
                    "val_metrics": val_report["metrics"],
                    "train_metrics": train_report["metrics"],
                    "prompt_program": prompt.model_dump(),
                },
            )

        current_summary = base.build_current_summary(train_history_reports, best_train_report)
        if mode == "last_report":
            decision_payload = {
                "mode": "last_report",
                "current_prompt_program": prompt.model_dump(),
                "current_summary": current_summary,
            }
            mode_hint = "You are only given the current iteration summary."
        elif mode == "all_reports":
            decision_payload = {
                "mode": "all_reports",
                "current_prompt_program": prompt.model_dump(),
                "history": train_history_reports,
            }
            mode_hint = (
                "You are given PAST report history, plus a separate current_summary for the current iteration.\n"
                "Use history for trajectory and current_summary for the decision now.\n"
            )
        elif mode == "history_summary":
            decision_payload = {
                "mode": "history_summary",
                "current_prompt_program": prompt.model_dump(),
                "history": train_history_reports[:-1],
                "current_summary": current_summary,
            }
            mode_hint = "You are given past report history plus a current summary."
        elif mode == "last_k_reports":
            decision_payload = {
                "mode": "last_k_reports",
                "current_prompt_program": prompt.model_dump(),
                "history": train_history_reports[-k_reports:],
            }
            mode_hint = f"You are given the last {k_reports} evaluation reports."
        else:
            raise ValueError(f"Unknown mode: {mode}")

        decision = request_loop_decision(
            optimizer_client,
            optimizer_name=optimizer_name,
            optimizer_instructions=optimizer_instructions,
            first_response=tool_response,
            train_report=train_report,
            decision_payload=decision_payload,
            mode_hint=mode_hint,
        )
        logger.log("decision", step, decision.model_dump())

        if decision.action == "stop":
            print(f"[STOP] {decision.stop_reason}")
            break
        if decision.action == "patch" and decision.patch is not None:
            prompt = base.apply_patch(prompt, decision.patch)
            continue
        raise RuntimeError("Invalid LoopDecision from optimizer model.")

    print("\n=== FINAL PROMPTPROGRAM ===")
    print(best_prompt.model_dump_json(indent=2))
    print("Best val score:", best_val_score, "Best val metrics:", best_val_report["metrics"] if best_val_report else None)

    final_test_report = base.evaluate_prompt_tool(
        target_client,
        best_prompt,
        test_items,
        logger=logger,
        target_model=target_model,
        step=iters,
        mode="test",
        max_workers=eval_workers,
        enable_critiques=False,
    )
    logger.log("final_test_stats", iters, final_test_report)
    return best_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="Seed for optimizer-side randomness.")
    ap.add_argument("--split_seed", type=int, default=base.DEFAULT_SPLIT_SEED, help="Seed used to shuffle and partition the LiveBench math pool.")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--mode", type=str, default="all_reports", choices=["all_reports", "history_summary", "last_report", "last_k_reports"])
    ap.add_argument("--k_reports", type=int, default=5)
    ap.add_argument("--k_topics", type=int, default=10)
    ap.add_argument("--clustering_sample_size", type=int, default=100)
    ap.add_argument("--test_every", type=int, default=5)
    ap.add_argument("--eval_workers", type=int, default=20)
    ap.add_argument("--critique_workers", type=int, default=2, help="Number of concurrent Gemini critique calls during train evaluation.")
    ap.add_argument(
        "--optimizer_name",
        type=str,
        default="gemini-3.1-pro",
        help="Model name used for optimizer/controller and critique calls.",
    )
    ap.add_argument("--optimizer_vertex_project", type=str, default=os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    ap.add_argument("--optimizer_vertex_location", type=str, default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    ap.add_argument("--target_model", type=str, default="gpt-4.1", help="Model being prompt-tuned/evaluated.")
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
    ap.add_argument("--data_dir", type=str, default=base.DEFAULT_DATA_DIR)
    ap.add_argument("--prepare_only", action="store_true", help="Create/load cached train/val/test splits and exit.")
    ap.add_argument("--evaluate_only", action="store_true", help="Evaluate the seed prompt on the cached splits and exit.")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the existing log for this exact run configuration.",
    )
    ap.add_argument(
        "--resume_from_log",
        type=str,
        default="",
        help="Explicit JSONL log path to resume from. Overrides the default computed log path.",
    )
    ap.add_argument(
        "--write_cleaned_log",
        action="store_true",
        help="Write a canonical cleaned companion log for downstream analysis.",
    )
    ap.add_argument(
        "--cleaned_log_path",
        type=str,
        default="",
        help="Explicit path for the cleaned log copy. Defaults to the raw log path with a _cleaned suffix.",
    )
    ap.add_argument(
        "--clean_log_only",
        action="store_true",
        help="Rewrite the existing raw log into a cleaned companion log and exit without running optimization.",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    train_items, val_items, test_items = base.load_or_create_livebench_math_splits(
        base_dir=args.data_dir,
        split_seed=args.split_seed,
    )
    base.ensure_disjoint_split_ids({"train": train_items, "val": val_items, "test": test_items})

    print(
        "Loaded LiveBench math splits: "
        f"{len(train_items)} train, {len(val_items)} val, {len(test_items)} test "
        f"(split_seed={args.split_seed})."
    )

    if args.prepare_only:
        return
    log_path = str(args.resume_from_log or "").strip() or build_run_log_path(
        args,
        n_train=len(train_items),
        n_val=len(val_items),
        n_test=len(test_items),
    )
    cleaned_log_path = str(args.cleaned_log_path or "").strip() or build_cleaned_log_path(log_path)
    if args.clean_log_only:
        write_cleaned_log(log_path, cleaned_log_path)
        raise SystemExit(0)

    logger = base.JsonlLogger(log_path)

    seed_prompt = base.make_seed_prompt()

    if args.evaluate_only:
        reports = base.run_seed_prompt_evaluation(
            base.client,
            train_items,
            val_items,
            test_items,
            seed_prompt,
            logger,
            target_model=args.target_model,
            eval_workers=args.eval_workers,
        )
        print(json.dumps(reports, indent=2))
        return

    optimizer_client = make_gemini_client(
        project=args.optimizer_vertex_project,
        location=args.optimizer_vertex_location,
    )
    resume_state = None
    if args.resume or args.resume_from_log:
        resume_state = load_resume_state(
            log_path,
            seed_prompt,
            prompt_complexity_weight=args.prompt_complexity_weight,
            prompt_complexity_unit=args.prompt_complexity_unit,
        )
        write_cleaned_log(log_path, cleaned_log_path)
        logger.log(
            "resume_state",
            resume_state["next_step"],
            {
                "log_path": log_path,
                "completed_steps": resume_state["completed_steps"],
                "next_step": resume_state["next_step"],
                "stopped": resume_state["stopped"],
            },
        )

    best_prompt = run_rpt_gemini(
        base.client,
        optimizer_client,
        train_items,
        val_items,
        test_items,
        seed_prompt,
        logger,
        target_model=args.target_model,
        optimizer_name=args.optimizer_name,
        iters=args.iters,
        mode=args.mode,
        test_every=args.test_every,
        k_topics=args.k_topics,
        clustering_sample_size=args.clustering_sample_size,
        k_reports=args.k_reports,
        seed=args.seed,
        eval_workers=args.eval_workers,
        critique_workers=args.critique_workers,
        prompt_complexity_weight=args.prompt_complexity_weight,
        prompt_complexity_unit=args.prompt_complexity_unit,
        resume_state=resume_state,
    )
    if args.write_cleaned_log or args.resume or args.resume_from_log:
        write_cleaned_log(log_path, cleaned_log_path)

    print("\n=== FINAL PROMPTPROGRAM ===")
    print(best_prompt.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
