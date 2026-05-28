"""
Reflective Prompt Tuning (RPT) for XBRL Formula with Gemini on Vertex AI.

This script keeps the target/eval model path identical to `rpt.tasks.xbrl_formula`
and only swaps the optimizer/controller + critic path to Gemini. Unlike the
OpenAI version, the optimizer tool loop is implemented with the native
`google-genai` SDK and an explicit `evaluate_prompt` function declaration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from tqdm import tqdm

from rpt.tasks import xbrl_formula as base
from rpt.gemini_utils import (
    make_gemini_client,
    medium_thinking_config,
    parse_gemini_structured_response,
)
from rpt.paths import XBRL_FORMULA_DATA_DIR


def critique_one_trace_with_gemini(
    optimizer_client: genai.Client,
    trace: Dict[str, Any],
    model: str = "gemini-3.1-pro",
) -> base.FailureCritique:
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

    response = optimizer_client.models.generate_content(
        model=model,
        contents=json.dumps(user_payload, ensure_ascii=False),
        config=genai_types.GenerateContentConfig(
            system_instruction=critic_system,
            response_mime_type="application/json",
            response_schema=base.FailureCritique,
            thinking_config=medium_thinking_config(),
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
    }


def evaluate_prompt_tool_gemini(
    target_client: OpenAI,
    optimizer_client: genai.Client,
    prompt: base.PromptProgram,
    items: List[Dict[str, Any]],
    logger: base.JsonlLogger,
    *,
    hard_k: int = 8,
    step: int = 0,
    mode: str = "dev",
    k_topics: int = 10,
    clustering_sample_size: int = 100,
    optimizer_name: str = "gemini-3.1-pro",
    seed: int = 0,
    target_model: str = "gpt-4.1",
    max_workers: int = 20,
    critique_workers: int = 1,
) -> Dict[str, Any]:
    del hard_k

    traces: List[base.EvalItemTrace] = []
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
                base.evaluate_single_item,
                target_client,
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
        actual_critique_workers = max(1, min(critique_workers, len(incorrect_eval_results)))
        with ThreadPoolExecutor(max_workers=actual_critique_workers) as executor:
            futures = [
                executor.submit(
                    critique_single_trace_gemini,
                    optimizer_client,
                    items[result["idx"]],
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

    metrics = base.EvalMetrics(
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
        clustering_cfg = base.ClusterFusionConfig(
            k_topics=k_topics,
            partition=base.PartitionConfig(
                num_groups=max(2, 2 * k_topics),
                sample_size=clustering_sample_size,
                seed=seed,
                cosine_order=True,
            ),
            domain_guidance=domain_guidance,
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

        insights = base.TraceInsights(
            failure_modes=[base.FailureModeTopic(**topic) for topic in selected_topics]
        )

    report = base.EvalReport(
        iteration=step,
        prompt_program=prompt.model_dump(),
        metrics=metrics,
        insights=insights,
    )
    return report.model_dump()


def make_evaluate_prompt_tool() -> genai_types.Tool:
    return genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name="evaluate_prompt",
                description=(
                    "Evaluate a PromptProgram on the XBRL formula-construction training split using the target model. "
                    "Returns an evaluation report with performance metrics and analysis insights."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "prompt_program": {
                            "type": "OBJECT",
                            "properties": {
                                "system": {"type": "STRING"},
                                "instruction": {"type": "STRING"},
                                "enforce_json_only": {"type": "BOOLEAN"},
                                # "max_reasoning_sentences": {"type": "INTEGER", "minimum": 1, "maximum": 10},
                            },
                        },
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
            thinking_config=medium_thinking_config(),
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
    plateau_warmup_active: bool = False,
    plateau_warmup_iters: int = 0,
) -> base.LoopDecision:
    function_response_part = genai_types.Part.from_function_response(
        name="evaluate_prompt",
        response={"report": train_report},
    )
    warmup_guidance = ""
    if plateau_warmup_active:
        warmup_guidance = (
            f"Warm-up phase: you are still within the first {plateau_warmup_iters} iterations. "
            "Do not choose STOP based on plateau, regression, or non-improving streak alone. "
            "Prefer a targeted PATCH unless the process is invalid or cannot continue.\n"
        )
    decision_user = (
        f"{mode_hint}\n"
        "Using the most recent `function_call_output` evaluation report above, now decide whether to STOP or output a PATCH.\n"
        f"{warmup_guidance}"
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
            thinking_config=medium_thinking_config(),
            temperature=0,
        ),
    )
    return parse_gemini_structured_response(response, base.LoopDecision)


def build_decision_request(
    *,
    mode: str,
    prompt: base.PromptProgram,
    train_history_reports: List[Dict[str, Any]],
    best_train_report: Optional[Dict[str, Any]],
    k_reports: int,
    step: int,
    plateau_warmup_iters: int,
) -> tuple[Dict[str, Any], str, bool]:
    current_summary = base.build_current_summary(train_history_reports, best_train_report)
    plateau_warmup_active = step <= plateau_warmup_iters
    decision_current_summary = current_summary
    if plateau_warmup_active:
        decision_current_summary = dict(current_summary)
        decision_current_summary["non_improving_streak"] = 0

    if mode == "last_report":
        decision_payload = {
            "mode": "last_report",
            "current_prompt_program": prompt.model_dump(),
            "current_summary": decision_current_summary,
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
            "current_summary": decision_current_summary,
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

    return decision_payload, mode_hint, plateau_warmup_active


def load_resume_state(
    log_path: str,
    *,
    seed_prompt: base.PromptProgram,
    iters: int,
    prompt_complexity_weight: float,
    prompt_complexity_unit: float,
) -> Dict[str, Any]:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Resume log path does not exist: {log_path}")

    step_prompts: Dict[int, base.PromptProgram] = {}
    train_reports: Dict[int, Dict[str, Any]] = {}
    dev_reports: Dict[int, Dict[str, Any]] = {}
    decisions: Dict[int, base.LoopDecision] = {}
    saw_final_test = False

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            event = rec.get("event")
            step = rec.get("step")
            payload = rec.get("payload")
            if not isinstance(step, int):
                continue

            if event == "iter_prompt" and isinstance(payload, dict):
                step_prompts[step] = base.PromptProgram.model_validate(payload)
            elif event == "train_stats" and isinstance(payload, dict):
                train_reports[step] = payload
            elif event == "dev_stats" and isinstance(payload, dict):
                dev_reports[step] = payload
            elif event == "decision" and isinstance(payload, dict):
                decisions[step] = base.LoopDecision.model_validate(payload)
            elif event == "final_test_stats":
                saw_final_test = True

    train_history_reports: List[Dict[str, Any]] = []
    best_train_report: Optional[Dict[str, Any]] = None
    best_train_score = float("-inf")
    for step in sorted(train_reports):
        report = train_reports[step]
        train_history_reports.append(report)
        train_score = base.score_report(
            report,
            prompt_complexity_weight=prompt_complexity_weight,
            prompt_complexity_unit=prompt_complexity_unit,
        )
        if best_train_report is None or base.better_score(train_score, best_train_score):
            best_train_report = report
            best_train_score = train_score

    best_dev_report: Optional[Dict[str, Any]] = None
    best_dev_score = float("-inf")
    best_prompt = seed_prompt
    for step in sorted(dev_reports):
        report = dev_reports[step]
        curr_dev_score = base.score_report(
            report,
            prompt_complexity_weight=prompt_complexity_weight,
            prompt_complexity_unit=prompt_complexity_unit,
        )
        if base.better_score(curr_dev_score, best_dev_score):
            best_dev_score = curr_dev_score
            best_dev_report = report
            best_prompt = base.PromptProgram.model_validate(
                report.get("prompt_program", seed_prompt.model_dump())
            )

    evaluated_steps = sorted(set(train_reports) & set(dev_reports))
    pending_decision_step = next(
        (step for step in reversed(evaluated_steps) if step not in decisions),
        None,
    )
    if pending_decision_step is not None:
        pending_prompt = step_prompts.get(pending_decision_step)
        if pending_prompt is None:
            pending_prompt = base.PromptProgram.model_validate(
                train_reports[pending_decision_step]["prompt_program"]
            )
        return {
            "prompt": pending_prompt,
            "train_history_reports": train_history_reports,
            "best_train_report": best_train_report,
            "best_prompt": best_prompt,
            "best_dev_report": best_dev_report,
            "best_dev_score": best_dev_score,
            "next_step": pending_decision_step,
            "pending_decision_step": pending_decision_step,
            "completed": False,
        }

    latest_prompt_step = max(step_prompts, default=0)
    if latest_prompt_step and (
        latest_prompt_step not in train_reports or latest_prompt_step not in dev_reports
    ):
        return {
            "prompt": step_prompts[latest_prompt_step],
            "train_history_reports": train_history_reports,
            "best_train_report": best_train_report,
            "best_prompt": best_prompt,
            "best_dev_report": best_dev_report,
            "best_dev_score": best_dev_score,
            "next_step": latest_prompt_step,
            "pending_decision_step": None,
            "completed": False,
        }

    if saw_final_test:
        return {
            "prompt": best_prompt,
            "train_history_reports": train_history_reports,
            "best_train_report": best_train_report,
            "best_prompt": best_prompt,
            "best_dev_report": best_dev_report,
            "best_dev_score": best_dev_score,
            "next_step": min(iters + 1, max(latest_prompt_step, max(evaluated_steps, default=0)) + 1),
            "pending_decision_step": None,
            "completed": True,
        }

    if decisions:
        last_decision_step = max(decisions)
        last_decision = decisions[last_decision_step]
        step_prompt = step_prompts.get(last_decision_step)
        if step_prompt is None and last_decision_step in train_reports:
            step_prompt = base.PromptProgram.model_validate(
                train_reports[last_decision_step]["prompt_program"]
            )
        if step_prompt is None:
            step_prompt = seed_prompt

        if last_decision.action == "patch" and last_decision.patch is not None:
            return {
                "prompt": base.apply_patch(step_prompt, last_decision.patch),
                "train_history_reports": train_history_reports,
                "best_train_report": best_train_report,
                "best_prompt": best_prompt,
                "best_dev_report": best_dev_report,
                "best_dev_score": best_dev_score,
                "next_step": last_decision_step + 1,
                "pending_decision_step": None,
                "completed": False,
            }

        if last_decision.action == "stop":
            return {
                "prompt": best_prompt,
                "train_history_reports": train_history_reports,
                "best_train_report": best_train_report,
                "best_prompt": best_prompt,
                "best_dev_report": best_dev_report,
                "best_dev_score": best_dev_score,
                "next_step": min(iters + 1, last_decision_step + 1),
                "pending_decision_step": None,
                "completed": False,
            }

    return {
        "prompt": seed_prompt,
        "train_history_reports": train_history_reports,
        "best_train_report": best_train_report,
        "best_prompt": best_prompt,
        "best_dev_report": best_dev_report,
        "best_dev_score": best_dev_score,
        "next_step": 1,
        "pending_decision_step": None,
        "completed": False,
    }


def run_rpt_gemini(
    target_client: OpenAI,
    optimizer_client: genai.Client,
    train_items: List[Dict[str, Any]],
    dev_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
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
    target_model: str = "gpt-4.1",
    prompt_complexity_weight: float = 0.0,
    prompt_complexity_unit: float = 1000.0,
    plateau_warmup_iters: int = 15,
    resume_state: Optional[Dict[str, Any]] = None,
) -> base.PromptProgram:
    optimizer_instructions = base.OPTIMIZER_INSTRUCTIONS

    prompt = seed_prompt
    train_history_reports: List[Dict[str, Any]] = []
    best_train_report: Optional[Dict[str, Any]] = None
    best_prompt = prompt
    best_dev_report: Optional[Dict[str, Any]] = None
    best_dev_score = float("-inf")
    start_step = 1
    pending_decision_step: Optional[int] = None

    if resume_state is not None:
        prompt = resume_state["prompt"]
        train_history_reports = list(resume_state["train_history_reports"])
        best_train_report = resume_state["best_train_report"]
        best_prompt = resume_state["best_prompt"]
        best_dev_report = resume_state["best_dev_report"]
        best_dev_score = resume_state["best_dev_score"]
        start_step = resume_state["next_step"]
        pending_decision_step = resume_state.get("pending_decision_step")

        completed_iters = len(train_history_reports)
        print(
            f"Resuming from {completed_iters} completed train iterations. "
            f"Next step: {start_step}."
        )
        if resume_state.get("completed"):
            print("Resume log already contains final_test_stats; skipping additional work.")
            return best_prompt

    if pending_decision_step is not None:
        step = pending_decision_step
        print(f"\n=== Reconstructing missing decision for RPT Iteration {step}/{iters} (mode={mode}) ===")
        tool_response = request_evaluate_prompt_call(
            optimizer_client,
            optimizer_name=optimizer_name,
            optimizer_instructions=optimizer_instructions,
            prompt=prompt,
            step=step,
            iters=iters,
        )
        decision_payload, mode_hint, plateau_warmup_active = build_decision_request(
            mode=mode,
            prompt=prompt,
            train_history_reports=train_history_reports,
            best_train_report=best_train_report,
            k_reports=k_reports,
            step=step,
            plateau_warmup_iters=plateau_warmup_iters,
        )
        decision = request_loop_decision(
            optimizer_client,
            optimizer_name=optimizer_name,
            optimizer_instructions=optimizer_instructions,
            first_response=tool_response,
            train_report=train_history_reports[-1],
            decision_payload=decision_payload,
            mode_hint=mode_hint,
            plateau_warmup_active=plateau_warmup_active,
            plateau_warmup_iters=plateau_warmup_iters,
        )
        logger.log("decision", step=step, payload=decision.model_dump())

        if decision.action == "stop":
            print(f"[STOP] {decision.stop_reason}")
            start_step = iters + 1
        elif decision.action == "patch" and decision.patch is not None:
            prompt = base.apply_patch(prompt, decision.patch)
            start_step = step + 1
        else:
            raise RuntimeError("Invalid LoopDecision from optimizer model.")

    for step in range(start_step, iters + 1):
        if (step % test_every) == 0 or step == 1:
            test_report_json = base.evaluate_prompt_tool(
                target_client,
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
            raise RuntimeError("Optimizer model did not call evaluate_prompt.")

        train_report = evaluate_prompt_tool_gemini(
            target_client,
            optimizer_client,
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
            critique_workers=critique_workers,
            target_model=target_model,
        )
        logger.log("train_stats", step=step, payload=train_report)
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

        dev_report = base.evaluate_prompt_tool(
            target_client,
            prompt,
            dev_items,
            logger=logger,
            step=step,
            mode="dev",
            max_workers=eval_workers,
            target_model=target_model,
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
                    "train_metrics": train_report["metrics"],
                    "prompt_program": train_report.get("prompt_program", prompt.model_dump()),
                },
            )

        decision_payload, mode_hint, plateau_warmup_active = build_decision_request(
            mode=mode,
            prompt=prompt,
            train_history_reports=train_history_reports,
            best_train_report=best_train_report,
            k_reports=k_reports,
            step=step,
            plateau_warmup_iters=plateau_warmup_iters,
        )

        decision = request_loop_decision(
            optimizer_client,
            optimizer_name=optimizer_name,
            optimizer_instructions=optimizer_instructions,
            first_response=tool_response,
            train_report=train_report,
            decision_payload=decision_payload,
            mode_hint=mode_hint,
            plateau_warmup_active=plateau_warmup_active,
            plateau_warmup_iters=plateau_warmup_iters,
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
        test_items,
        logger=logger,
        step=iters,
        mode="test",
        max_workers=eval_workers,
        target_model=target_model,
    )
    logger.log("final_test_stats", step=iters, payload=final_test_report_json)
    return best_prompt


def main() -> None:
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
    ap.add_argument("--critique_workers", type=int, default=1, help="Number of concurrent Gemini critique calls during train evaluation.")
    ap.add_argument(
        "--optimizer_name",
        type=str,
        default="gemini-3.1-pro",
        help="Model name used for optimizer/controller and critique calls.",
    )
    ap.add_argument("--optimizer_vertex_project", type=str, default=os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    ap.add_argument("--optimizer_vertex_location", type=str, default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
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
    ap.add_argument(
        "--plateau_warmup_iters",
        type=int,
        default=15,
        help="Number of early iterations where Gemini should keep patching before using plateau as a stop signal.",
    )
    ap.add_argument(
        "--resume_log_path",
        type=str,
        default="",
        help="Optional existing JSONL log to resume in place after an interrupted run.",
    )
    ap.add_argument(
        "--output_log_path",
        type=str,
        default="",
        help="Optional output JSONL path. When resuming, this can be a new branched log file.",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    target_client = OpenAI()

    raw_train = base.load_jsonl(str(XBRL_FORMULA_DATA_DIR / "train.jsonl"), args.n_train, args.seed)
    raw_dev = base.load_jsonl(str(XBRL_FORMULA_DATA_DIR / "val.jsonl"), args.n_dev, args.seed)
    raw_test = base.load_jsonl(str(XBRL_FORMULA_DATA_DIR / "test.jsonl"), args.n_test, args.seed)
    train_items = base.process_formula_data(raw_train)
    dev_items = base.process_formula_data(raw_dev)
    test_items = base.process_formula_data(raw_test)

    print(
        f"Loaded {len(train_items)} XBRL formula train items, "
        f"{len(dev_items)} dev items, and {len(test_items)} test items."
    )

    default_log_path = os.path.join(
            "logs",
            "xbrl_formula",
            "gemini",
            args.optimizer_name,
            (
                f"log_{args.mode}_iters_{args.iters}_train_{args.n_train}_dev_{args.n_dev}_"
                f"test_{args.n_test}_seed_{args.seed}_k_topics_{args.k_topics}_"
                f"cluster_desc_comprehensive_optimizer_non_minimal_"
                f"pcw_{args.prompt_complexity_weight:g}_pcu_{args.prompt_complexity_unit:g}_wrm_iters_{args.plateau_warmup_iters}.jsonl"
            ),
        )
    resume_log_path = os.path.abspath(os.path.expanduser(args.resume_log_path)) if args.resume_log_path else ""
    output_log_path = os.path.abspath(os.path.expanduser(args.output_log_path)) if args.output_log_path else ""

    if resume_log_path and output_log_path and output_log_path != resume_log_path:
        if os.path.exists(output_log_path):
            raise FileExistsError(
                "Refusing to resume into an existing different output log path, to avoid duplicating history. "
                f"Choose a new path or delete the existing file first: {output_log_path}"
            )
        output_dir = os.path.dirname(output_log_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        shutil.copyfile(resume_log_path, output_log_path)
        print(f"Copied resume log to new output log: {output_log_path}")

    log_path = output_log_path or resume_log_path or default_log_path
    logger = base.JsonlLogger(log_path)
    optimizer_client = make_gemini_client(
        project=args.optimizer_vertex_project,
        location=args.optimizer_vertex_location,
    )

    seed_prompt = base.make_seed_prompt()
    resume_state = None
    if resume_log_path:
        resume_state = load_resume_state(
            resume_log_path,
            seed_prompt=seed_prompt,
            iters=args.iters,
            prompt_complexity_weight=args.prompt_complexity_weight,
            prompt_complexity_unit=args.prompt_complexity_unit,
        )
    best_prompt = run_rpt_gemini(
        target_client,
        optimizer_client,
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
        critique_workers=args.critique_workers,
        optimizer_name=args.optimizer_name,
        target_model=args.target_model,
        prompt_complexity_weight=args.prompt_complexity_weight,
        prompt_complexity_unit=args.prompt_complexity_unit,
        plateau_warmup_iters=args.plateau_warmup_iters,
        resume_state=resume_state,
    )

    print("\n=== FINAL PROMPTPROGRAM ===")
    print(best_prompt.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
