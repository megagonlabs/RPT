"""
Reflective Prompt Tuning (RPT) for HotpotQA with Gemini on Vertex AI.

This script keeps the target/eval model path identical to `rpt.tasks.hotpotqa`
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

from rpt.tasks import hotpotqa as base
from rpt.gemini_utils import (
    build_cleaned_log_path,
    high_thinking_config,
    make_gemini_client,
    parse_gemini_structured_response,
    write_cleaned_log as _write_cleaned_log,
)
from rpt.paths import HOTPOTQA_DATA_DIR


def build_run_log_path(args: argparse.Namespace) -> str:
    return (
        f"logs/hotpotqa/gemini/{args.optimizer_name}/"
        f"log_{args.mode}_iters_{args.iters}_k_topics_{args.k_topics}_"
        f"clustering_sample_size_{args.clustering_sample_size}_"
        f"cluster_desc_comprehensive_"
        f"optimizer_non_minimal_"
        f"pcw_{args.prompt_complexity_weight:g}_pcu_{args.prompt_complexity_unit:g}_final.jsonl"
    )


def write_cleaned_log(log_path: str, cleaned_path: Optional[str] = None) -> str:
    return _write_cleaned_log(
        log_path,
        cleaned_path,
        required_events={"iter_prompt", "train_stats", "dev_stats", "decision"},
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
    dev_reports: Dict[int, Dict[str, Any]] = {}
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
            elif event == "dev_stats":
                dev_reports[step] = payload
            elif event == "decision":
                decisions[step] = base.LoopDecision.model_validate(payload)

    completed_steps = 0
    for step in range(1, max(iter_prompts.keys(), default=0) + 1):
        if (
            step not in iter_prompts
            or step not in train_reports
            or step not in dev_reports
            or step not in decisions
        ):
            break
        completed_steps = step
        if decisions[step].action == "stop":
            break

    prompt = seed_prompt
    train_history_reports: List[Dict[str, Any]] = []
    best_train_report: Optional[Dict[str, Any]] = None
    best_dev_report: Optional[Dict[str, Any]] = None
    best_dev_score = float("-inf")
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

        dev_report = dev_reports[step]
        dev_score = base.score_report(
            dev_report,
            prompt_complexity_weight=prompt_complexity_weight,
            prompt_complexity_unit=prompt_complexity_unit,
        )
        if base.better_score(dev_score, best_dev_score):
            best_dev_score = dev_score
            best_dev_report = dev_report
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
        "best_dev_report": best_dev_report,
        "best_dev_score": best_dev_score,
        "best_prompt": best_prompt,
        "stopped": stopped,
    }


def critique_one_trace_with_gemini(
    optimizer_client: genai.Client,
    trace: Dict[str, Any],
    model: str,
) -> base.FailureCritique:
    critic_system = f"""
        You are a strict evaluation critic for HotpotQA failures.
        You are given ONE QA trace with:
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

    user_payload = {
        "question": trace.get("question"),
        "context": trace.get("context", None),
        "gold_answer": trace.get("gold"),
        "predicted_answer": trace.get("pred"),
        "confidence": trace.get("confidence"),
        "justification": trace.get("justification"),
    }

    response = optimizer_client.models.generate_content(
        model=model,
        contents=json.dumps(user_payload, ensure_ascii=False),
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
        "impact": 1.0 - float(trace.f1),
    }


def evaluate_prompt_tool_gemini(
    target_client: OpenAI,
    optimizer_client: genai.Client,
    prompt: base.PromptProgram,
    hotpot_items: List[Dict[str, Any]],
    logger: base.JsonlLogger,
    *,
    max_context_chars: int = 90000,
    step: int = 0,
    mode: str = "train",
    k_topics: int = 10,
    clustering_sample_size: int = 100,
    optimizer_name: str = "gemini-3.1-pro",
    seed: int = 0,
    max_workers: int = 20,
    critique_workers: int = 1,
    enable_critiques: bool = True,
) -> Dict[str, Any]:
    traces: List[base.EvalItemTrace] = []
    failure_labels: List[Dict[str, Any]] = []
    fmt_errors = 0
    confs: List[float] = []
    brier_terms: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    incorrect_results: List[Dict[str, Any]] = []
    max_workers = max(1, min(max_workers, len(hotpot_items) if hotpot_items else 1))
    eval_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                base.evaluate_single_item,
                target_client,
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

        logger.log(
            "target_trace",
            step=step,
            payload={
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
            },
        )

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

        if mode == "train" and enable_critiques and not trace.correct_em:
            incorrect_results.append(result)

    n = len(traces)
    em_rate = sum(1 for t in traces if t.correct_em) / max(1, n)
    f1_avg = sum(t.f1 for t in traces) / max(1, n)
    precision_avg = sum(precisions) / max(1, len(precisions))
    recall_avg = sum(recalls) / max(1, len(recalls))
    avg_conf = sum(confs) / max(1, len(confs))
    brier = sum(brier_terms) / max(1, len(brier_terms))
    fmt_rate = fmt_errors / max(1, n)

    report = base.EvalReport(
        iteration=step,
        prompt_program=prompt.model_dump(),
        metrics=base.EvalMetrics(
            n=n,
            exact_match=em_rate,
            f1=f1_avg,
            precision=precision_avg,
            recall=recall_avg,
            avg_confidence=avg_conf,
            brier=brier,
            format_error_rate=fmt_rate,
        ),
        insights=None,
    )

    if mode in {"dev", "test"} or not enable_critiques:
        return json.loads(report.model_dump_json())

    if incorrect_results:
        critique_results: List[Dict[str, Any]] = []
        actual_critique_workers = max(1, min(critique_workers, len(incorrect_results)))
        with ThreadPoolExecutor(max_workers=actual_critique_workers) as executor:
            futures = [
                executor.submit(
                    critique_single_trace_gemini,
                    optimizer_client,
                    result["row"],
                    result["trace"],
                    optimizer_name,
                )
                for result in incorrect_results
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Critiquing errors"):
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
                    step=step,
                    payload={
                        "idx": critique_result["idx"],
                        "row_id": critique_result["row_id"],
                        "question": critique_result["question"],
                        "error": critique_result["error"],
                    },
                )
                continue
            failure_id_start = len(failure_labels)
            for fm in crit.failure_modes:
                failure_labels.append(
                    {
                        "id": failure_id_start,
                        "text": f"{str(fm.label)}:{str(fm.definition)}",
                        "impact": critique_result["impact"],
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

    if not failure_labels:
        return json.loads(report.model_dump_json())

    domain_guidance = (
        "You will receive short failure-mode labels produced by an iterative prompt optimization method. "
        "Each record is describing a failure mode in the model’s behavior. "
    )
    feature_context = "failure modes"
    clustering_cfg = base.ClusterFusionConfig(
        k_topics=k_topics,
        partition=base.PartitionConfig(
            num_groups=2 * k_topics,
            sample_size=clustering_sample_size,
            seed=seed,
            cosine_order=True,
        ),
        domain_guidance=domain_guidance,
        feature_context=feature_context,
        text_field="text",
        topic_desc_mode="comprehensive",
    )
    topics = base.run_clusterfusion(failure_labels, clustering_cfg, get_topics=True)
    logger.log("failure_mode_clusters", step=step, payload={"topics": topics})

    selected_topics = [t for t in topics if t.get("prevalence", 0) >= 0.10]
    if len(selected_topics) < 1:
        selected_topics = topics[: min(len(topics), 3)]

    for topic in selected_topics:
        topic.pop("prevalence", None)
        topic.pop("topic_id", None)

    report.insights = base.TraceInsights(
        failure_modes=[base.FailureModeTopic(**topic) for topic in selected_topics]
    )
    return json.loads(report.model_dump_json())


def make_evaluate_prompt_tool() -> genai_types.Tool:
    return genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name="evaluate_prompt",
                description=(
                    "Evaluate a PromptProgram on the HotpotQA train set using the target model (gpt-4.1). "
                    "Returns an evaluation report (metrics + analysis insights)."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "prompt_program": {
                            "type": "OBJECT",
                            "description": (
                                "The FULL PromptProgram to evaluate on the target model. "
                                "Copy the current PromptProgram from the user message.\n"
                                "Fields: system, instruction, enforce_json_only, max_justification_sentences."
                            ),
                            "properties": {
                                "system": {"type": "STRING", "description": "System message for the target model."},
                                "instruction": {"type": "STRING", "description": "Task instruction shown to the target model."},
                                "enforce_json_only": {"type": "BOOLEAN", "description": "If true, target must output only JSON."},
                                "max_justification_sentences": {
                                    "type": "INTEGER",
                                    "description": "Max number of sentences in the target model justification.",
                                    "minimum": 1,
                                    "maximum": 10,
                                },
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
        f"{prompt.model_dump_json(indent=2)}\n"
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
        "Using the most recent `function_call_output` evaluation report above, now decide whether to STOP or output a PATCH.\n"
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
    hotpot_train_items: List[Dict[str, Any]],
    hotpot_dev_items: List[Dict[str, Any]],
    hotpot_test_items: List[Dict[str, Any]],
    seed_prompt: base.PromptProgram,
    logger: base.JsonlLogger,
    *,
    iters: int = 5,
    mode: str = "last_report",
    test_every: int = 5,
    clustering_sample_size: int = 100,
    k_topics: int = 10,
    k_reports: int = 5,
    seed: int = 0,
    eval_workers: int = 20,
    critique_workers: int = 1,
    optimizer_name: str = "gemini-3.1-pro",
    prompt_complexity_weight: float = 0.0,
    prompt_complexity_unit: float = 1000.0,
    resume_state: Optional[Dict[str, Any]] = None,
) -> base.PromptProgram:
    optimizer_instructions = base.OPTIMIZER_INSTRUCTIONS

    prompt = seed_prompt
    train_history_reports: List[Dict[str, Any]] = []
    best_train_report = None

    best_prompt = prompt
    best_dev_report = None
    best_dev_score = float("-inf")
    completed_steps = 0
    skip_optimization_loop = False

    if resume_state is not None:
        prompt = resume_state["prompt"]
        train_history_reports = list(resume_state["train_history_reports"])
        best_train_report = resume_state["best_train_report"]
        best_prompt = resume_state["best_prompt"]
        best_dev_report = resume_state["best_dev_report"]
        best_dev_score = float(resume_state["best_dev_score"])
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

        if (step % test_every) == 0: # or step == 1:
            test_report_json = base.evaluate_prompt_tool(
                target_client,
                prompt,
                hotpot_test_items,
                logger=logger,
                step=step,
                mode="test",
                max_workers=eval_workers,
            )
            logger.log("test_stats", step=step, payload=test_report_json)

        logger.log("iter_prompt", step=step, payload=prompt.model_dump())
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

        report = evaluate_prompt_tool_gemini(
            target_client,
            optimizer_client,
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
            critique_workers=critique_workers,
        )
        logger.log("train_stats", step=step, payload=report)
        train_history_reports.append(report)

        train_score = base.score_report(
            report,
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
            best_train_report = report

        dev_report = base.evaluate_prompt_tool(
            target_client,
            prompt,
            hotpot_dev_items,
            logger=logger,
            step=step,
            mode="dev",
            max_workers=eval_workers,
            enable_critiques=False,
        )
        logger.log("dev_stats", step=step, payload=dev_report)

        curr_dev_score = base.score_report(
            dev_report,
            prompt_complexity_weight=prompt_complexity_weight,
            prompt_complexity_unit=prompt_complexity_unit,
        )
        if base.better_score(curr_dev_score, best_dev_score):
            best_dev_score = curr_dev_score
            best_dev_report = dev_report
            best_prompt = prompt
            logger.log(
                "best_update",
                step=step,
                payload={
                    "selection_split": "dev",
                    "score": best_dev_score,
                    "dev_metrics": dev_report["metrics"],
                    "train_metrics": report["metrics"],
                    "prompt_program": report.get("prompt_program", prompt.model_dump()),
                },
            )

        current_summary = base.build_current_summary(train_history_reports, best_train_report)
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

        decision = request_loop_decision(
            optimizer_client,
            optimizer_name=optimizer_name,
            optimizer_instructions=optimizer_instructions,
            first_response=tool_response,
            train_report=report,
            decision_payload=decision_payload,
            mode_hint=mode_hint,
        )
        logger.log("decision", step=step, payload=decision.model_dump())

        if decision.action == "stop":
            print(f"[STOP] {decision.stop_reason}")
            break

        if decision.action == "patch" and decision.patch is not None:
            prompt = base.apply_patch(prompt, decision.patch)
            continue

        raise RuntimeError("Invalid LoopDecision from optimizer model.")

    print("\n=== FINAL PROMPTPROGRAM ===")
    print(best_prompt.model_dump_json(indent=2))
    print("Best dev score:", best_dev_score, "Best dev metrics:", best_dev_report["metrics"] if best_dev_report else None)

    final_test_report_json = base.evaluate_prompt_tool(
        target_client,
        best_prompt,
        hotpot_test_items,
        logger=logger,
        step=iters,
        mode="test",
        max_workers=eval_workers,
    )
    logger.log("final_test_stats", step=iters, payload=final_test_report_json)
    return best_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=300, help="HotpotQA train sample size used for optimizer feedback.")
    ap.add_argument("--n_dev", type=int, default=300, help="HotpotQA val sample size (100-200 recommended).")
    ap.add_argument("--n_test", type=int, default=500, help="HotpotQA test sample size (100-200 recommended).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling and clustering.")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--mode", type=str, default="all_reports", choices=["all_reports", "history_summary", "last_report", "last_k_reports"], help="`history_summary` provides past history plus a separate current summary for the decision step.")
    ap.add_argument("--k_topics", type=int, default=10, help="Number of failure mode clusters for the optimizer-side critic.")
    ap.add_argument("--k_reports", type=int, default=5, help="Number of recent reports to provide in 'last_k_reports' mode.")
    ap.add_argument("--clustering_sample_size", type=int, default=100, help="Number of failure examples to sample for clustering.")
    ap.add_argument("--test_every", type=int, default=5, help="test every N iterations (including iteration 1 for the initial prompt)")
    ap.add_argument("--eval_workers", type=int, default=20, help="Number of worker threads for target-model evaluation.")
    ap.add_argument("--critique_workers", type=int, default=3, help="Number of concurrent Gemini critique calls during train evaluation.")
    ap.add_argument(
        "--optimizer_name",
        type=str,
        default="gemini-3.1-pro",
        help="Model name used for optimizer/controller and critique calls.",
    )
    ap.add_argument("--optimizer_vertex_project", type=str, default=os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    ap.add_argument("--optimizer_vertex_location", type=str, default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
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

    log_path = str(args.resume_from_log or "").strip() or build_run_log_path(args)
    cleaned_log_path = str(args.cleaned_log_path or "").strip() or build_cleaned_log_path(log_path)
    if args.clean_log_only:
        write_cleaned_log(log_path, cleaned_log_path)
        raise SystemExit(0)

    target_client = OpenAI()

    train_path = str(HOTPOTQA_DATA_DIR / "train.jsonl")
    dev_path = str(HOTPOTQA_DATA_DIR / "dev.jsonl")
    test_path = str(HOTPOTQA_DATA_DIR / "test.jsonl")

    hotpot_train_items = base.load_or_create_hotpot_split(
        path=train_path,
        sample_n=args.n_train,
        seed=args.seed,
    )
    train_ids = [item.get("id") for item in hotpot_train_items]

    hotpot_test_items = base.load_or_create_hotpot_split(
        path=test_path,
        sample_n=args.n_test,
        seed=args.seed + 1,
        excluding_ids=train_ids,
    )
    test_ids = [item.get("id") for item in hotpot_test_items]

    hotpot_dev_items = base.load_or_create_hotpot_split(
        path=dev_path,
        sample_n=args.n_dev,
        seed=args.seed + 2,
        excluding_ids=train_ids + test_ids,
    )

    base.ensure_disjoint_hotpot_splits(
        {
            "train": hotpot_train_items,
            "dev": hotpot_dev_items,
            "test": hotpot_test_items,
        }
    )

    print(
        f"Loaded {len(hotpot_train_items)} HotpotQA train items, "
        f"{len(hotpot_dev_items)} dev items, and {len(hotpot_test_items)} test items."
    )
    base.prime_context_cache(hotpot_train_items)
    base.prime_context_cache(hotpot_dev_items)
    base.prime_context_cache(hotpot_test_items)

    if args.evaluate_only:
        logger = base.JsonlLogger(
            f"logs/hotpotqa/gemini/log_evaluate_only_mode_{args.mode}_iters_{args.iters}_k_topics_{args.k_topics}_clustering_sample_size_{args.clustering_sample_size}_cluster_desc_comprehensive_seed_prompt.jsonl"
        )
        reports = base.run_seed_prompt_evaluation(
            target_client,
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

    logger = base.JsonlLogger(log_path)
    optimizer_client = make_gemini_client(
        project=args.optimizer_vertex_project,
        location=args.optimizer_vertex_location,
    )

    seed_prompt = base.make_seed_prompt()
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
            step=resume_state["next_step"],
            payload={
                "log_path": log_path,
                "completed_steps": resume_state["completed_steps"],
                "next_step": resume_state["next_step"],
                "stopped": resume_state["stopped"],
            },
        )

    best_prompt = run_rpt_gemini(
        target_client,
        optimizer_client,
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
        critique_workers=args.critique_workers,
        optimizer_name=args.optimizer_name,
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
