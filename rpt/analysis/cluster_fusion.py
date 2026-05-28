"""
clusterfusion_cosine_gpt5.py

ClusterFusion-style 3-step pipeline with cosine ordering:
1) Embedding-guided subset partition (KMeans + balanced sampling + cosine ordering)
2) GPT-5 topic extraction (YOUR updated prompt + JSON format)
3) GPT-5 topic assignment (YOUR updated prompt + {"topic": "<topic_name>"} output)

Dependencies:
  pip install openai numpy scikit-learn

Env:
  export OPENAI_API_KEY=...

Typical inputs:
  - failure_docs.json  (list[str] OR list[{"doc":...}])
  - patch_docs.json    (list[str] OR list[{"doc":...}])

Outputs:
  - /mnt/data/failure_clusterfusion_cosine.json
  - /mnt/data/patch_clusterfusion_cosine.json
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from openai import OpenAI
import os
import argparse
from tqdm import tqdm
from collections import Counter, defaultdict

from rpt.analysis.paths import resolve_analysis_dir


# -------------------------
# Config


# -------------------------
EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1"

# -------------------------
# IO
# -------------------------
def _extract_json_text(raw_text: str) -> Optional[str]:
    text = str(raw_text or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        fence_match = re.match(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()

    decoder = json.JSONDecoder()

    try:
        decoder.decode(text)
        return text
    except Exception:
        pass

    for start_idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            _, end_idx = decoder.raw_decode(text[start_idx:])
            return text[start_idx : start_idx + end_idx]
        except Exception:
            continue
    return None


def _load_model_json(raw_text: str) -> Any:
    json_text = _extract_json_text(raw_text)
    if not json_text:
        preview = str(raw_text or "").strip()[:500]
        raise ValueError(f"Could not extract JSON from model response: {preview}")
    return json.loads(json_text)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# Embeddings
# -------------------------
def embed_texts(client: OpenAI, texts: List[str], batch_size: int = 128) -> np.ndarray:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    return np.array(vecs, dtype=np.float32)


def cosine_sim_to_ref(vecs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row in vecs and ref."""
    v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    r = ref / (np.linalg.norm(ref) + 1e-12)
    return (v @ r).astype(np.float32)


# -------------------------
# Step 1: subset partition + cosine ordering
# -------------------------
@dataclass
class PartitionConfig:
    num_groups: int = 50
    sample_size: int = 300
    seed: int = 0
    cosine_order: bool = True


def balanced_sample_indices(labels: np.ndarray, sample_size: int, seed: int = 0) -> List[int]:
    """
    Balanced sampling across KMeans groups.
    If group is too small, sample with replacement to meet quota.
    """
    rng = random.Random(seed)

    groups: Dict[int, List[int]] = {}
    for i, g in enumerate(labels.tolist()):
        groups.setdefault(int(g), []).append(i)

    G = len(groups)
    if G == 0:
        return []

    base = sample_size // G
    rem = sample_size % G

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    chosen: List[int] = []
    for idx, gid in enumerate(group_ids):
        quota = base + (1 if idx < rem else 0)
        members = groups[gid]
        if len(members) >= quota:
            chosen.extend(rng.sample(members, quota))
        else:
            # all members, then sample with replacement
            chosen.extend(members)
            if len(members) > 0:
                while len([x for x in chosen if x in members]) < quota:
                    chosen.append(rng.choice(members))

    if len(chosen) > sample_size:
        chosen = chosen[:sample_size]
    return chosen


def cosine_order_sample(sample_idx: List[int], emb: np.ndarray) -> List[int]:
    """
    Similarity-based ordering: sort by cosine similarity to the first sampled record.
    """
    if not sample_idx:
        return sample_idx
    first = sample_idx[0]
    ref = emb[first]
    sims = cosine_sim_to_ref(emb[sample_idx], ref)
    order = np.argsort(-sims)  # descending
    return [sample_idx[i] for i in order.tolist()]


def embedding_guided_subset_partition(
    emb: np.ndarray,
    cfg: PartitionConfig) -> Tuple[np.ndarray, List[int]]:
    n_samples = int(len(emb))
    if n_samples == 0:
        return np.array([], dtype=int), []

    # KMeans requires n_samples >= n_clusters; clamp for small failure sets.
    n_groups = max(1, min(cfg.num_groups, n_samples))
    if n_groups == 1:
        labels = np.zeros(n_samples, dtype=int)
    else:
        km = KMeans(n_clusters=n_groups, random_state=cfg.seed, n_init="auto")
        labels = km.fit_predict(emb)

    sample_idx = balanced_sample_indices(labels, cfg.sample_size, seed=cfg.seed)

    if cfg.cosine_order:
        sample_idx = cosine_order_sample(sample_idx, emb)
    else:
        sample_idx = sorted(sample_idx, key=lambda i: (int(labels[i]), i))

    return labels, sample_idx


# -------------------------
# Step 2: GPT-5 topic extraction (UPDATED PROMPT + UPDATED SCHEMA)
# -------------------------
TOPIC_EXTRACTION_SCHEMA_V2 = {
    "type": "json_schema",
    "json_schema": {
        "name": "TopicExtractionV2",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "topics": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "topic": {"type": "string"},
                            "description": {"type": "string"},
                            "examples": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1
                            },
                        },
                        "required": ["topic", "description", "examples"],
                    },
                }
            },
            "required": ["topics"],
        },
    },
}


def llm_extract_topics(
    client: OpenAI,
    exemplars: List[str],
    k_topics: int,
    feature_context: str,
    domain_guidance: Optional[str] = None,
    temperature: float = 0.0,
    topic_desc_mode: str = "comprehensive"  # "comprehensive" or "concise"
    ) -> List[Dict[str, Any]]:
    """
    Uses your updated extraction prompt and returns a normalized topic list:
      [{"topic_id": i, "name": ..., "definition": ..., "examples": [...]}, ...]
    """
    guidance = (domain_guidance or "").strip()

    system = "You are an intelligent assistant skilled in summarizing and extracting insights."
    if guidance:
        system += f"\nDomain guidance:\n{guidance}\n"

    user = (
        f"You are now tasked with reviewing records related to {feature_context}. The list of records is provided below:\n"
        f"Records: {exemplars}\n"
        f"Your goal is to extract key topics from these records.\n"
        f"For each identified topic, please provide a {topic_desc_mode} explanation along with examples.\n"
        f"The total number of topics you extract should be exactly {k_topics}, not more than {k_topics} and not fewer than {k_topics}.\n"
        f"The result should be returned in JSON format, where each key represents an index, and the corresponding value is a dictionary with:\n"
        f"•  A topic name as the key, and \n"
        f"•  A description and some examples as the value."
    )

    resp = client.responses.create(
        model=LLM_MODEL,
        temperature=temperature,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text={
            "format": {
                "type": "json_schema",
                "name": TOPIC_EXTRACTION_SCHEMA_V2["json_schema"]["name"],
                "schema": TOPIC_EXTRACTION_SCHEMA_V2["json_schema"]["schema"],
                "strict": True,
            }
        },
    )

    raw = json.loads(resp.output_text)

    topics_list = raw.get("topics", [])
    if not isinstance(topics_list, list):
        topics_list = []

    parsed: List[Dict[str, Any]] = []
    for i, t in enumerate(topics_list):
        if not isinstance(t, dict):
            continue

        name = str(t.get("topic", "")).strip()
        desc = str(t.get("description", "")).strip()
        exs = t.get("examples", [])
        if not isinstance(exs, list):
            exs = []

        parsed.append(
            {
                "topic_id": i,
                "name": name,
                "definition": desc,
                "examples": [str(x) for x in exs][:8],
            }
        )

    # Enforce exactly k_topics (defensive; schema should already ensure this if you set min/max)
    if len(parsed) < k_topics:
        while len(parsed) < k_topics:
            parsed.append(
                {
                    # "topic_id": len(parsed),
                    "name": f"Topic {len(parsed)}",
                    "definition": "Placeholder (model returned fewer topics than requested).",
                    "examples": ["(missing)"],
                }
            )
    parsed = parsed[:k_topics]

    # Reindex cleanly -- topic_id does not have further use, therefore ommitted from the output. If needed, it can be easily re-added as the index in the list.
    for i, t in enumerate(parsed):
        t["topic_id"] = i

    return parsed



# -------------------------
# Step 3: GPT-5 assignment (UPDATED PROMPT + UPDATED SCHEMA)
# -------------------------
ASSIGNMENT_SCHEMA = {
    "name": "TopicAssignmentV2",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    },
}


def llm_assign_topic(
    client: OpenAI,
    record_text: str,
    topic_def_dict: Dict[str, str],
    feature_context: str,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Assign record_text to one of the topic_def_dict keys.
    Returns:
      {"topic": <topic_name>, "topic_id": <int>, "topic_definition": <str>}
    """
    if not topic_def_dict:
        raise ValueError("topic_def_dict is empty")

    valid_topics = list(topic_def_dict.keys())
    valid_set = set(valid_topics)

    system = "You are a helpful assistant, that can help me label each record into topics."
    user = (
        f"Following records is about {feature_context}. Please classify the record into one of the following topics, "
        f"which are represented as a dictionary.  Its keys are the names of the topics and values are the descriptions "
        f"of the topic: {topic_def_dict}\n"
        f"Record: {record_text}\n"
        f"Return the result in JSON format with the following format: key 'topic', with value as the picked topic."
    )

    for _ in range(max_retries + 1):
        resp = client.responses.create(
            model=LLM_MODEL,
            temperature=temperature,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": ASSIGNMENT_SCHEMA["name"],
                    "schema": ASSIGNMENT_SCHEMA["schema"],
                    "strict": True,
                }
            },
        )
        out = json.loads(resp.output_text)
        picked = str(out.get("topic", "")).strip()
        if picked in valid_set:
            topic_id = valid_topics.index(picked)
            return {
                "topic": picked,
                "topic_id": topic_id,
                "topic_definition": topic_def_dict.get(picked, ""),
            }

    # Fallback to first topic
    first = valid_topics[0]
    return {
        "topic": first,
        "topic_id": 0,
        "topic_definition": topic_def_dict.get(first, ""),
        "fallback": True,
    }


# -------------------------
# End-to-end runner
# -------------------------
@dataclass
class ClusterFusionConfig:
    k_topics: int = 10
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    domain_guidance: Optional[str] = None
    feature_context: str = "records"
    text_field: str = "text"
    topic_desc_mode: str = "comprehensive"  # "comprehensive" or "concise"


def run_clusterfusion(records: List[Dict[str, Any]], cfg: ClusterFusionConfig, get_topics=False) -> Dict[str, Any]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    texts = [str(r.get(cfg.text_field, "")) for r in records]
    impacts = [float(r.get("impact", 1.0)) for r in records]  # optional, only if you have impact scores and want to include them in the guidance
    examples = [str(r.get("example", "")) for r in records]  # optional, only if you have example texts and want to include them in the guidance
    emb = embed_texts(client, texts)

    # Step 1
    group_labels, sample_idx = embedding_guided_subset_partition(emb, cfg.partition)
    exemplars = [texts[i] for i in sample_idx]
    # how to also add examples for each text in the exemplar set so the clustering method can also include examples?
    exemplars = [f"{e}\nExample: {ex}" if ex else e for e, ex in zip(exemplars, [examples[i] for i in sample_idx])]

    # Step 2
    topics = llm_extract_topics(
        client=client,
        exemplars=exemplars,
        k_topics=cfg.k_topics,
        feature_context=cfg.feature_context,
        domain_guidance=cfg.domain_guidance,
        temperature=0.0,
        topic_desc_mode=cfg.topic_desc_mode,
    )

    # Build topic definition dict for step 3 prompt
    topic_def_dict = {t["name"]: t["definition"] for t in topics}

    # Step 3 (assign all records)
    assignments: List[int] = []
    assigned_topics: List[str] = []
    for t in tqdm(texts, desc="Assigning topics"):
        a = llm_assign_topic(
            client=client,
            record_text=t,
            topic_def_dict=topic_def_dict,
            feature_context=cfg.feature_context,
            temperature=0.0,
            max_retries=2,
        )
        assignments.append(int(a["topic_id"]))
        assigned_topics.append(a["topic"])

    if get_topics:
        counts_by_name = Counter(assigned_topics)          # topic name -> count 
        total_count = sum(counts_by_name.values())
        # how to include impact scores in prevalence? e.g. sum of impact scores for records assigned to the topic, divided by total impact scores across all records (to get a weighted prevalence that accounts for both frequency and impact)
        impact_by_id = defaultdict(float)   # topic_id -> sum impact
        for tid, imp in zip(assignments, impacts):
            impact_by_id[tid] += float(imp)
        total_impact = sum(float(x) for x in impacts)
        for i, t in enumerate(topics):
            name = t["name"]
            topic_frequency = int(counts_by_name.get(name, 0)) / max(1, total_count)
            average_impact = impact_by_id.get(t["topic_id"], 0) / max(1e-6, total_impact)  # avoid division by zero
            t["prevalence"] = (topic_frequency + average_impact) * 0.5
        
        # Sort topics by prevalence
        topics.sort(key=lambda t: t.get("prevalence", 0), reverse=True)
        return topics

    return {
        "k_topics": cfg.k_topics,
        "partition": {
            "num_groups": cfg.partition.num_groups,
            "sample_size": cfg.partition.sample_size,
            "seed": cfg.partition.seed,
            "ordering": "cosine_to_first" if cfg.partition.cosine_order else "cluster_index",
        },
        "feature_context": cfg.feature_context,
        "domain_guidance": cfg.domain_guidance,
        "topics": topics,  # includes examples
        "topic_def_dict": topic_def_dict,
        "assignments": assignments,            # topic_id per record
        "assigned_topic_names": assigned_topics,  # topic name per record
        "sample_indices": sample_idx,
        "group_labels": group_labels.tolist(),
        "records": records,
    }


# -------------------------
# Convenience: load records from common file formats
# -------------------------
def load_records_from_docs_json(path: str) -> List[Dict[str, Any]]:
    """
    Accepts:
      - list[str]
      - list[dict] with one of prefer_fields
    Returns list[{"text":..., "id":...}]
    """
    data = load_json(path)
    out = []
    if isinstance(data, dict):
        docs = data.get("docs", [])
        if isinstance(docs, list):
            for i, doc in enumerate(docs):
                out.append({"id": i, "text": str(doc)})
        return out

    raise ValueError(f"Unsupported input format for {path}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run ClusterFusion with cosine ordering on failures and patches.")
    argparser.add_argument(
        "--log_path",
        type=str,
        default="logs/xbrl_formula/gpt-5/example.jsonl",
        help="Path to the source log. You can omit .jsonl.",
    )
    argparser.add_argument("--docs_dir", type=str, default=None, help="Existing clustering-results dir, task/log stem, or bare log stem.")
    argparser.add_argument("--task_name", type=str, default=None, help="Task name for clustering_results/<task_name>/<model_name>/<log_stem>.")
    argparser.add_argument("--model_name", type=str, default=None, help="Model name for clustering_results/<task_name>/<model_name>/<log_stem>.")
    argparser.add_argument("--clustering_root", type=str, default="clustering_results", help="Root directory for clustering outputs.")
    argparser.add_argument("--k_topics", type=int, default=10, help="Number of topics to extract.")
    argparser.add_argument("--sample_size", type=int, default=100, help="Sample size for clustering.")
    argparser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and clustering.")
    argparser.add_argument("--cosine_order", action="store_true", help="Whether to order the sample by cosine similarity to the first sampled record.")
    argparser.add_argument("--domain", type=str, choices=["failures", "patches"], required=True, help="Which domain to run on.")
    args = argparser.parse_args()
    # ---- Failures ----

    analysis_ref = args.docs_dir or args.log_path
    analysis_dir = resolve_analysis_dir(
        analysis_ref,
        task_name=args.task_name,
        model_name=args.model_name,
        root=args.clustering_root,
    )
    input_path = analysis_dir / ("failure_docs.json" if args.domain == "failures" else "patch_docs.json")
    records = load_records_from_docs_json(input_path)

    # all three seeds for more robust results
    # input_path_1 = "clustering_results/" + "log_last_report_iters_20_dev_450_test_150_seed_0" + "/failure_docs.json" if args.domain == "failures" else "clustering_results/" + "log_last_report_iters_20_dev_450_test_150_seed_0" + "/patch_docs.json"
    # input_path_2 = "clustering_results/" + "log_last_report_iters_20_dev_450_test_150_seed_7" + "/failure_docs.json" if args.domain == "failures" else "clustering_results/" + "log_last_report_iters_20_dev_450_test_150_seed_7" + "/patch_docs.json"
    # input_path_3 = "clustering_results/" + "log_last_report_iters_20_dev_450_test_150_seed_42" + "/failure_docs.json" if args.domain == "failures" else "clustering_results/" + "log_last_report_iters_20_dev_450_test_150_seed_42" + "/patch_docs.json"
    # records_1 = load_records_from_docs_json(input_path_1)
    # records_2 = load_records_from_docs_json(input_path_2)
    # records_3 = load_records_from_docs_json(input_path_3)
    # records = records_1 + records_2 + records_3

    if args.domain == "failures":
        domain_guidance = (
            "You will receive short failure-mode labels produced by an iterative prompt optimization method. "
            "Each record is a concise label describing a recurring error pattern in the model’s behavior. "
        )
        feature_context = "failure modes"
    elif args.domain == "patches":
        domain_guidance = (
            "You will receive short prompt-edit bullets ('patches') produced by an iterative prompt optimization method. "
            "Each record describes an atomic change made to the prompt between two iterations. "
        )
        feature_context = "prompt patches"

    cfg = ClusterFusionConfig(
        k_topics=args.k_topics,
        partition=PartitionConfig(num_groups=2*args.k_topics, sample_size=args.sample_size, seed=args.seed, cosine_order=True),
        domain_guidance=domain_guidance,
        feature_context=feature_context,
        text_field="text",
    )

    out = run_clusterfusion(records, cfg)
    save_json(analysis_dir / f"{args.domain}_clusterfusion_cosine.json", out)

    # out_path = f"clustering_results/log_last_reports_iters_20_dev_450/{args.domain}_clusterfusion_cosine.json"
    # save_json(out_path, out)
