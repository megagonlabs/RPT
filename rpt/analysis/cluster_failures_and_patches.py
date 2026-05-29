from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal, Iterable
from collections import defaultdict
from openai import OpenAI

from rpt.analysis.paths import analysis_dir_for_log, resolve_log_path


# ============================================================
# 0) IO utilities
# ============================================================

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_json(path: str | Path, obj: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ============================================================
# 1) Parse log -> per-step records
# ============================================================

@dataclass
class StepRecord:
    step: int
    iter_prompt: Optional[Dict[str, Any]] = None
    analysis_output: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None

def parse_steps_from_log(events: List[Dict[str, Any]]) -> Dict[int, StepRecord]:
    steps: Dict[int, StepRecord] = {}
    for e in events:
        if "step" not in e:
            continue
        step = int(e["step"])
        rec = steps.get(step)
        if rec is None:
            rec = StepRecord(step=step)
            steps[step] = rec

        ev = e.get("event")
        payload = e.get("payload", {})

        if ev == "iter_prompt":
            rec.iter_prompt = payload
        elif ev == "failure_mode_clusters": #ev == "analysis_output" or
            rec.analysis_output = payload
        elif ev == "decision":
            rec.decision = payload

    return steps

def iter_transitions(steps: Dict[int, StepRecord]) -> List[Tuple[int, int]]:
    ks = sorted(steps.keys())
    pairs = []
    for i in range(len(ks) - 1):
        t, t1 = ks[i], ks[i + 1]
        if steps[t].iter_prompt and steps[t1].iter_prompt and steps[t].analysis_output:
            pairs.append((t, t1))
    return pairs

# ============================================================
# 2) Failure mode corpus (names-only, canonicalized)
# ============================================================

def canonicalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\d+", "<num>", s)
    s = re.sub(r"[“”\"'`]", "", s)
    s = re.sub(r"[^a-z0-9<>\s\-\_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # normalize a few common variants
    s = s.replace("over-confident", "overconfidence").replace("over confident", "overconfidence")
    s = s.replace("multi hop", "multihop").replace("two hop", "twohop")
    s = s.replace("x-or-y", "xorY").replace("x or y", "xorY")
    return s

# change below function to extract failure_modenames and desctriptions here. 
def extract_failure_mode_names(analysis_output: Dict[str, Any]) -> List[str]: 
    """
    Tries to extract failure mode names robustly.
    Adjust if your schema differs.

    Common shapes:
      analysis_output["failure_modes"] = [{"name": "...", ...}, ...]
      analysis_output["failures"] = [{"title": "..."}]
      analysis_output["failure_modes"] = ["...", "..."]
    """
    if not analysis_output:
        return []

    fm = None
    if "failure_modes" in analysis_output:
        fm = analysis_output["failure_modes"]
    
    elif "topics" in analysis_output: #added
        fm = analysis_output["topics"]
    
    if fm is None:
        return []

    names: List[str] = []
    if isinstance(fm, list):
        for item in fm:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                # try common name keys
                # if "name" in item and item["name"] and "description" in item and item["description"]:
                if "name" in item and item["name"] and "definition" in item and item["definition"]:
                    names.append(str(item["name"]) + ": " + str(item["definition"]))
                    # for nk in ("name", "description"): # we can later add description. 
                    #     if nk in item and item[nk]:
                    #         names.append(str(item[nk]))
                    #         break
    elif isinstance(fm, dict):
        # sometimes dict of name->details
        for k in fm.keys():
            names.append(str(k))

    # drop empties
    return [n for n in names if n and n.strip()]

def build_failure_docs(
    steps: Dict[int, StepRecord],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns:
      docs: list[str] for clustering
      meta: list[dict] same length with step, raw_name
    Each failure-mode entry is one datapoint.
    """
    docs: List[str] = []
    meta: List[Dict[str, Any]] = []

    for step, rec in sorted(steps.items()):
        if not rec.analysis_output:
            continue
        names = extract_failure_mode_names(rec.analysis_output)
        for raw in names:
            doc = canonicalize_text(raw)
            if not doc:
                continue
            docs.append(doc)
            meta.append({"step": step, "raw_name": raw})

    return docs, meta

# ============================================================
# 3) Patch corpus: LLM diff bulletizer -> each bullet is datapoint
# ============================================================

def synthesize_prompt_text(iter_prompt_payload: Dict[str, Any]) -> str:
    """
    Make a stable text view for the LLM.
    """
    system = iter_prompt_payload.get("system", "") or ""
    instruction = iter_prompt_payload.get("instruction", "") or ""
    extras = {k: v for k, v in iter_prompt_payload.items() if k not in ("system", "instruction")}
    extras_txt = json.dumps(extras, ensure_ascii=False, indent=2) if extras else ""
    return (
        "SYSTEM:\n" + system.strip() +
        "\n\nINSTRUCTION:\n" + instruction.strip() +
        ("\n\nEXTRAS:\n" + extras_txt if extras_txt else "")
    ).strip()

# ---- OpenAI Structured Outputs wrapper ----

@dataclass
class OpenAIConfig:
    model: str = "gpt-4.1"
    temperature: float = 0.0

class OpenAIWrapper:
    def __init__(self, cfg: OpenAIConfig):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cfg = cfg

    def structured(self, *, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.client.responses.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema.get("name", "schema"),
                    "schema": schema["schema"],
                    "strict": True,
                }
            },
        )
        raw = getattr(resp, "output_text", "") or ""
        if not raw:
            # fallback: collect output_text blocks
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        raw += getattr(c, "text", "")
        return json.loads(raw)

ChangeType = Literal["add", "remove", "tighten", "relax", "refactor", "other"]

PATCH_BULLETS_SCHEMA: Dict[str, Any] = {
  "name": "PatchBulletsPlusOnly",
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "bullets": {
        "type": "array",
        "minItems": 3,
        "maxItems": 12,
        "items": {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "change_type": { "type": "string", "enum": ["add", "tighten", "relax", "refactor"] },
            "bullet": { "type": "string" },
            "evidence_pointer": { "type": "string" }
          },
          "required": ["change_type", "bullet", "evidence_pointer"]
        }
      }
    },
    "required": ["bullets"]
  }
}

PATCH_BULLETS_SYSTEM = """You are given two prompts generated by a prompt optimization method: prompt_t and prompt_t+1.
Your task is to summarize ONLY the changes that are introduced in prompt_t+1 relative to prompt_t (i.e., Pt+1 − Pt).
You are also provided with OPTIONAL_PATCH_RATIONALE which may help explain intent, but you must ground every bullet in the prompt text.

Output requirements:
- 3 to 12 bullets total.
- Each bullet describes ONE change only.
- Each bullet MUST start with exactly one of: "Add", "Tighten", "Relax", "Refactor".
- Every bullet MUST be grounded in text that appears in prompt_t+1.
- Provide an evidence_pointer (<= ~15 words) copied from prompt_t+1 that supports the bullet.
- If OPTIONAL_PATCH_RATIONALE conflicts with the prompts, ignore the rationale and follow the prompts.

Return strict JSON.
"""

def llm_diff_to_bullets(
    llm: OpenAIWrapper,
    prompt_t_text: str,
    prompt_t1_text: str,
    *,
    decision_rationale: Optional[str] = None,
) -> List[Dict[str, Any]]:
    user = f"""Generate atomic bullets describing the differences.

PROMPT_T:
{prompt_t_text}

PROMPT_T_PLUS_1:
{prompt_t1_text}

OPTIONAL_PATCH_RATIONALE:
{decision_rationale or ""}
"""
    out = llm.structured(system=PATCH_BULLETS_SYSTEM, user=user, schema=PATCH_BULLETS_SCHEMA)
    bullets = out.get("bullets", [])
    # safety: keep only non-empty bullets
    cleaned = []
    for b in bullets:
        if not b.get("bullet"):
            continue
        cleaned.append({
            "change_type": b["change_type"],
            "bullet": b["bullet"].strip(),
            "evidence_pointer": (b.get("evidence_pointer") or "").strip(),
        })
    return cleaned

def extract_decision_rationale(decision_payload: Optional[Dict[str, Any]]) -> str:
    if not decision_payload:
        return ""
    patch = decision_payload.get("patch") or {}
    return patch.get("rationale", "") or ""

def build_patch_docs(
    steps: Dict[int, StepRecord],
    transitions: List[Tuple[int, int]],
    llm: OpenAIWrapper,
    cache_path: Optional[str | Path] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Each LLM bullet becomes one document.
    Returns:
      docs: list[str] (canonicalized bullet text for clustering)
      meta: list[dict] same length with (t, t1, change_type, raw_bullet, evidence_pointer)
    Uses optional JSON cache to avoid re-calling the LLM repeatedly.
    """
    cache: Dict[str, Any] = {}
    if cache_path and Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))

    docs: List[str] = []
    meta: List[Dict[str, Any]] = []

    for (t, t1) in transitions:
        key = f"{t}->{t1}"
        if key in cache:
            bullets = cache[key]["bullets"]
        else:
            pt = synthesize_prompt_text(steps[t].iter_prompt or {})
            pt1 = synthesize_prompt_text(steps[t1].iter_prompt or {})
            rationale = extract_decision_rationale(steps[t].decision)
            bullets = llm_diff_to_bullets(llm, pt, pt1, decision_rationale=rationale)
            cache[key] = {"bullets": bullets}

            if cache_path:
                write_json(cache_path, cache)

        for b in bullets:
            raw_bullet = b["bullet"]
            doc = canonicalize_text(raw_bullet)
            if not doc:
                continue
            docs.append(doc)
            meta.append({
                "transition": key,
                "t": t,
                "t1": t1,
                "change_type": b["change_type"],
                "raw_bullet": raw_bullet,
                "evidence_pointer": b.get("evidence_pointer", ""),
            })

    return docs, meta

# ============================================================
# 4) Clustering (BERTopic preferred; fallback to embeddings+HDBSCAN)
# ============================================================

def cluster_with_bertopic(
    docs: List[str],
    min_topic_size: int = 10,
    seed_topic_list: Optional[List[List[str]]] = None,
    n_neighbors: int = 10,
    # min_dist: float = 0.05,
    # min_samples: int = 2,
    # top_n_words: int = 12,
) -> Dict[str, Any]:
    """
    Returns dict with:
      topics: list[int] topic id per doc
      topic_info: list[dict] summary rows
      representations: dict topic_id -> list[str] representative docs
    """
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    # Use a small, strong baseline embedding model
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    # umap_model = umap.UMAP(
    #     n_neighbors=n_neighbors,
    #     # n_components=5,
    #     # min_dist=min_dist,
    #     metric="cosine",
    #     random_state=42,
    # )

    # hdbscan_model = hdbscan.HDBSCAN(
    #     min_cluster_size=min_topic_size,
    #     # min_samples=min_samples,
    #     metric="euclidean",
    #     cluster_selection_method="leaf",  # finer splits
    #     prediction_data=True,
    # )

    topic_model = BERTopic(
        embedding_model=emb_model,
        min_topic_size=min_topic_size,
        # umap_model=umap_model,
        # hdbscan_model=hdbscan_model,
        # top_n_words=top_n_words,
        seed_topic_list=seed_topic_list,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs)

    info_df = topic_model.get_topic_info()

    # Representative docs: take top n docs per topic by c-TF-IDF similarity (BERTopic has method)
    # We'll do a simple: for each topic, store 10 example docs from that topic.
    reps: Dict[int, List[str]] = defaultdict(list)
    for doc, tid in zip(docs, topics):
        if tid == -1:
            continue
        if len(reps[tid]) < 10:
            reps[tid].append(doc)

    return {
        "topics": topics,
        "topic_info": info_df.to_dict(orient="records"),
        "representations": {str(k): v for k, v in reps.items()},
        "model": topic_model,  # optional, can be large; remove if you plan to save JSON only
    }

def cluster_with_embeddings_hdbscan(
    docs: List[str],
    *,
    min_cluster_size: int = 10,
) -> Dict[str, Any]:
    """
    Fallback if BERTopic isn't available.
    Uses sentence-transformers + HDBSCAN and extracts top words by TF-IDF per cluster.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    import hdbscan

    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    X = emb_model.encode(docs, show_progress_bar=True, normalize_embeddings=True)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(X)  # -1 = noise

    # top words per cluster via TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    tfidf = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    topic_info = []
    representations: Dict[int, List[str]] = defaultdict(list)

    for label in sorted(set(labels)):
        if label == -1:
            continue
        idxs = [i for i, l in enumerate(labels) if l == label]
        # representative docs (first 10)
        for i in idxs[:10]:
            representations[label].append(docs[i])

        # top tf-idf terms averaged across docs in cluster
        avg = tfidf[idxs].mean(axis=0)
        avg = np.asarray(avg).ravel()
        top_idx = avg.argsort()[::-1][:12]
        words = [vocab[i] for i in top_idx if avg[i] > 0][:12]

        topic_info.append({
            "Topic": int(label),
            "Count": len(idxs),
            "TopWords": ", ".join(words),
        })

    return {
        "topics": labels.tolist(),
        "topic_info": topic_info,
        "representations": {str(k): v for k, v in representations.items()},
        "model": None,
    }


# ----------------------------
# 0) OpenAI structured output
# ----------------------------
def structured_call_openai(
    model: str,
    system: str,
    user_obj: Dict[str, Any],
    schema: Dict[str, Any],
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Requires:
      pip install openai
      export OPENAI_API_KEY=...
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    user = json.dumps(user_obj, ensure_ascii=False, indent=2)

    resp = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": True,
            }
        },
    )

    # In the OpenAI Responses API, output_text is the easiest reliable accessor.
    return json.loads(resp.output_text)


# ----------------------------
# 1) Schemas
# ----------------------------
FAILURE_LABEL_SCHEMA = {
    "name": "FailureTopicLabel",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic_name": {"type": "string", "description": "3–7 words"},
            "summary": {"type": "string", "description": "2–4 sentences"},
            "key_signals": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 6,
            },
            "typical_symptom": {"type": "string", "description": "1 sentence"},
        },
        "required": ["topic_name", "summary", "key_signals", "typical_symptom"],
    },
}

PATCH_LABEL_SCHEMA = {
    "name": "PatchTopicLabel",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic_name": {"type": "string", "description": "3–7 words"},
            "summary": {"type": "string", "description": "2–4 sentences"},
            "edit_signature": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 6,
            },
            "risk_tradeoff": {"type": "string", "description": "1–2 sentences"},
        },
        "required": ["topic_name", "summary", "edit_signature", "risk_tradeoff"],
    },
}

FAILURE_SYSTEM = """You are labeling a cluster of FAILURE MODE NAMES derived from the analysis component of a prompt optimization method.
Input includes:
- top_words: cluster keywords
- examples: representative failure names (may be canonicalized)

Task:
- Create a short topic_name (3–7 words)
- Write a concise summary (2–4 sentences) of the failure pattern
- List 2–6 key_signals (surface cues / phrases / situations)
- Provide a 1-sentence typical_symptom describing how the model fails

Rules:
- Stay grounded in examples/top_words.
Return strict JSON.
"""

PATCH_SYSTEM = """You are labeling a cluster of PATCH BULLETS describing edits in consecutive prompt versions from a prompt optimization method.

Input includes:
- top_words: cluster keywords
- examples: representative edit bullets (may be canonicalized)

Task:
- Create a short topic_name (3–7 words)
- Write a concise summary (2–4 sentences) of the edit family
- List 2–6 edit_signature items (what these edits look like)
- Provide a short risk_tradeoff (how this can hurt, e.g., brittleness)

Rules:
- Stay grounded in examples/top_words.
Return strict JSON.
"""

# ----------------------------
# 2) Helpers for your file format
# ----------------------------
def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def parse_top_words(top_words_field: str, k: int = 12) -> List[str]:
    """
    Your TopWords field looks like:
      "token, token token, ..."

    Convert to list[str] with trimming.
    """
    if not top_words_field:
        return []
    parts = [p.strip() for p in top_words_field.split(",")]
    parts = [p for p in parts if p]
    return parts[:k]

def get_topic_rows(topic_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    topic_info contains Topic=-1 sometimes in BERTopic; you likely want to skip -1.
    """
    rows = []
    for r in topic_info:
        t = int(r.get("Topic"))
        if t == -1:
            continue
        rows.append(r)
    # sort by topic id for stable outputs
    rows.sort(key=lambda x: int(x["Topic"]))
    return rows

def get_examples(representations: Dict[str, Any], topic_id: int, max_examples: int = 10) -> List[str]:
    """
    representations uses string keys like "0", "1", ...
    """
    ex = representations.get(str(topic_id), []) or []
    if not isinstance(ex, list):
        return []
    # keep best N
    return [str(x) for x in ex[:max_examples]]


# ----------------------------
# 3) Main labeling functions
# ----------------------------
def label_topic_file(
    in_path: str | Path,
    out_path: str | Path,
    kind: str,  # "failures" or "patches"
    model: str = "gpt-4.1",
    max_examples: int = 10,
    ) -> Dict[str, Any]:
    data = load_json(in_path)

    topic_info = data.get("topic_info", [])
    representations = data.get("representations", {})

    rows = get_topic_rows(topic_info)

    labeled: Dict[str, Any] = {
        "source_file": str(in_path),
        "kind": kind,
        "model": model,
        "topics": {},
    }

    for r in rows:
        topic_id = int(r["Topic"])
        count = int(r.get("Count", 0))
        top_words = parse_top_words(str(r.get("TopWords", "")), k=12)
        examples = get_examples(representations, topic_id, max_examples=max_examples)

        user_obj = {
            "topic_id": topic_id,
            "count": count,
            "top_words": top_words,
            "examples": examples,
        }

        if kind == "failures":
            label = structured_call_openai(
                model=model,
                system=FAILURE_SYSTEM,
                user_obj=user_obj,
                schema=FAILURE_LABEL_SCHEMA,
                temperature=0.0,
            )
        elif kind == "patches":
            label = structured_call_openai(
                model=model,
                system=PATCH_SYSTEM,
                user_obj=user_obj,
                schema=PATCH_LABEL_SCHEMA,
                temperature=0.0,
            )
        else:
            raise ValueError(f"Unknown kind: {kind}")

        labeled["topics"][str(topic_id)] = {
            "topic_id": topic_id,
            "count": count,
            "top_words": top_words,
            "examples": examples,
            "label": label,
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8")
    return labeled
# ============================================================
# Main: run 1-4 end-to-end
# ============================================================

def main(
    log_path: str,
    out_dir: str,
    task_name: Optional[str] = None,
    model_name: Optional[str] = None,
    logs_root: str = "logs",
    openai_model: str = "gpt-4.1",
    cache_path: Optional[str] = None,
    min_topic_size_errors: int = 10,
    min_topic_size_patches: int = 4,
):
    # events = load_jsonl(log_path)
    # log_path_1 = "logs/hotpotqa/log_last_report_iters_20_dev_450_test_150_seed_0.jsonl"
    # events_1 = load_jsonl(log_path_1)
    # log_path_2 = "logs/hotpotqa/log_last_report_iters_20_dev_450_test_150_seed_7.jsonl"
    # events_2 = load_jsonl(log_path_2)
    # log_path_3 = "logs/hotpotqa/log_last_report_iters_20_dev_450_test_150_seed_42.jsonl"
    # events_3 = load_jsonl(log_path_3)
    # events = events_1 + events_2 + events_3

    # log_path = "logs/hotpotqa/log_last_report_iters_20_dev_450_test_500_seed_0.jsonl"
    log_path = resolve_log_path(log_path, task_name=task_name, model_name=model_name, logs_root=logs_root)
    events = load_jsonl(log_path)

    steps = parse_steps_from_log(events)
    transitions = iter_transitions(steps)

    # Store outputs under clustering_results/<task_name>/<model_name>/<log_stem>/ when model_name is available.
    out_dir = analysis_dir_for_log(log_path, root=out_dir, task_name=task_name, model_name=model_name, logs_root=logs_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsed {len(steps)} steps, {len(transitions)} transitions suitable for patch clustering.")

    # ---- (1) Failure docs ----
    failure_docs, failure_meta = build_failure_docs(steps)
    print(f"Failure docs: {len(failure_docs)}")

    # ---- (2) Patch docs (LLM bullets) ----
    llm = OpenAIWrapper(OpenAIConfig(model=openai_model, temperature=0.0))

    patch_docs, patch_meta = build_patch_docs(
        steps,
        transitions,
        llm,
        cache_path=cache_path or (Path(out_dir) / "patch_bullets_cache.json"),
    )
    print(f"Patch bullet docs: {len(patch_docs)}")

    # Save raw corpora (for inspection)
    write_json(Path(out_dir) / "failure_docs.json", {"docs": failure_docs, "meta": failure_meta})
    write_json(Path(out_dir) / "patch_docs.json", {"docs": patch_docs, "meta": patch_meta})

    # ---- (3) Cluster failures ----
    failures_cluster = None
    patches_cluster = None

    try:
        failures_cluster = cluster_with_bertopic(failure_docs, min_topic_size_errors=min_topic_size_errors)
        # Remove model before saving JSON (model not JSON serializable)
        model_obj = failures_cluster.pop("model", None)
        write_json(Path(out_dir) / "failure_topics.json", failures_cluster)
        print("Clustered failures with BERTopic.")
    except Exception as e:
        print(f"BERTopic failed for failures ({e}). Falling back to embeddings+HDBSCAN.")
        failures_cluster = cluster_with_embeddings_hdbscan(failure_docs, min_cluster_size=min_topic_size_errors)
        write_json(Path(out_dir) / "failure_topics.json", failures_cluster)

    # ---- (4) Cluster patches ----
    try:
        patches_cluster = cluster_with_bertopic(patch_docs, min_topic_size_errors=min_topic_size_patches)
        patches_cluster.pop("model", None)
        write_json(Path(out_dir) / "patch_topics.json", patches_cluster)
        print("Clustered patches with BERTopic.")
    except Exception as e:
        print(f"BERTopic failed for patches ({e}). Falling back to embeddings+HDBSCAN.")
        patches_cluster = cluster_with_embeddings_hdbscan(patch_docs, min_cluster_size=min_topic_size_patches)
        write_json(Path(out_dir) / "patch_topics.json", patches_cluster)

    # ---- (5) Join assignments back onto meta (so you can inspect examples per topic) ----
    failure_assign = failures_cluster["topics"]
    patch_assign = patches_cluster["topics"]

    failure_labeled = []
    for doc, meta, tid in zip(failure_docs, failure_meta, failure_assign):
        failure_labeled.append({**meta, "doc": doc, "topic": int(tid)})

    patch_labeled = []
    for doc, meta, tid in zip(patch_docs, patch_meta, patch_assign):
        patch_labeled.append({**meta, "doc": doc, "topic": int(tid)})

    write_json(Path(out_dir) / "failure_labeled.json", failure_labeled)
    write_json(Path(out_dir) / "patch_labeled.json", patch_labeled)

    print(f"Wrote outputs to: {out_dir}")
    print("Key files:")
    print(" - failure_topics.json / patch_topics.json  (topic summaries)")
    print(" - failure_labeled.json / patch_labeled.json (each datapoint with topic id)")
    print(" - patch_bullets_cache.json (LLM bullet cache)")

    print("Now you can run label_topic_file to get human-readable labels.")
    failure_out = Path(out_dir) / "failure_topic_labels.json"
    patch_out = Path(out_dir) / "patch_topic_labels.json"
    label_topic_file(in_path=Path(out_dir) / "failure_topics.json", out_path=failure_out, kind="failures", model=openai_model, max_examples=10)
    label_topic_file(in_path=Path(out_dir) / "patch_topics.json", out_path=patch_out, kind="patches", model=openai_model, max_examples=10)

    print("Wrote:")
    print(" -", failure_out)
    print(" -", patch_out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log_path",
        default="logs/xbrl_formula/gpt-5/example.jsonl",
        help="Path to the JSONL log. You can also pass logs/<task>/<log_stem> without .jsonl.",
    )
    ap.add_argument("--log_name", default=None, help="Deprecated alias for a log path or bare log name.")
    ap.add_argument("--task_name", default=None, help="Task name used in clustering_results/<task_name>/<model_name>/<log_stem> when it cannot be inferred.")
    ap.add_argument("--model_name", default=None, help="Model name subdir used in clustering_results/<task_name>/<model_name>/<log_stem>.")
    ap.add_argument("--logs_root", default="logs", help="Root directory containing task log folders.")
    ap.add_argument("--out", default="clustering_results", help="Output directory")
    ap.add_argument("--model", default="gpt-4.1", help="OpenAI model for diff bulletization and topic labeling")
    ap.add_argument("--min_topic_size_errors", type=int, default=4)
    ap.add_argument("--min_topic_size_patches", type=int, default=4)
    ap.add_argument("--cache", default=None, help="Cache path for LLM bullets (JSON)")
    args = ap.parse_args()
    log_ref = args.log_name or args.log_path

    main(
        log_path=log_ref,
        out_dir=args.out,
        task_name=args.task_name,
        model_name=args.model_name,
        logs_root=args.logs_root,
        openai_model=args.model,
        cache_path=args.cache,
        min_topic_size_errors=args.min_topic_size_errors,
        min_topic_size_patches=args.min_topic_size_patches,
    )
