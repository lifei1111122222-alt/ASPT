"""PartNet structured-3D experiments: resume frozen inputs, generate, evaluate.

The formal pipeline uses the partnet_* entry points below. Previously published
open-world helpers are retained in a labelled compatibility section; its legacy
AABB metrics are NOT used by the new PartNet evaluator.

Existing manifest, descriptions and geometry-adjacency JSON files are read-only.
No PartNet GT geometry or mesh adjacency is sent to the LLM.
API keys are read from DEEPSEEK_API_KEY, never stored in generated files.
"""
from __future__ import annotations
import os
import json
import math
import itertools
import re
import copy
import csv
import hashlib
import time
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from scipy.spatial import cKDTree

PIPELINE_VERSION = "partnet-runner-1.0"
REPRESENTATIONS = ("anchor", "global", "local")
REPRESENTATION_SCHEMA_VERSION = "fixed-parts-obb-v1"
EVALUATION_VERSION = "partnet-obb-eval-v1"

# =============================================================================
# 1. Configuration and resumable JSON I/O
# =============================================================================
@dataclass
class PartNetConfig:
    partnet_root: str = r"E:\datasets\partnet"
    project_dir: str = "."
    output_dir: str = "partnet_results"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    max_tokens: int = 8192
    max_attempts: int = 3
    workers: int = 4
    timeout: float = 180.0
    transport_retries: int = 2
    base_url: str = "https://api.deepseek.com"
    extra_body: dict = field(default_factory=dict)
    near_contact_threshold: float = 0.003
    zero_tolerance: float = 1e-9
    random_anchor_seeds: tuple = (0, 1, 2, 3, 4)
    scale_factors: tuple = (0.8, 1.2, 1.5)
    bootstrap_samples: int = 10000
    statistics_seed: int = 42
    expected_objects: int | None = 880
    object_limit: int | None = None
    show_every: int = 20
    allow_api: bool = True

    def project_path(self, name: str) -> Path:
        return Path(self.project_dir).expanduser().resolve() / name

    def output_path(self, name: str = "") -> Path:
        base = Path(self.output_dir).expanduser()
        if not base.is_absolute():
            base = self.project_path(str(base))
        return base / name

    def request_config(self) -> dict:
        return {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens, "base_url": self.base_url, "extra_body": self.extra_body, "response_format": {"type": "json_object"}}


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(data: Any, path: str | Path) -> None:
    """Atomic replace; originals are not truncated on an interrupted write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_error(error: Exception) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted>", f"{type(error).__name__}: {error}")


def _index_objects(data: dict, label: str) -> dict:
    objects = data.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{label}: expected objects list")
    index = {}
    for row in objects:
        object_id = str(row["object_id"])
        if object_id in index:
            raise ValueError(f"{label}: duplicate object_id {object_id}")
        index[object_id] = row
    return index


def _read_cache(path: Path, fingerprint: str) -> dict | None:
    if not path.exists():
        return None
    record = load_json(path)
    if record.get("fingerprint") != fingerprint:
        raise ValueError(f"缓存配置或输入已改变：{path}。请使用新的 output_dir；不会静默覆盖或重调 API。")
    return record


def create_deepseek_client(api_key: str | None = None, base_url: str = "https://api.deepseek.com", timeout: float = 180.0):
    """Use an explicit key or DEEPSEEK_API_KEY without storing the credential."""
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("Missing API key: pass api_key or set DEEPSEEK_API_KEY.")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)


def _parse_json_reply(text: str) -> dict:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject_nonfinite(value):
        raise ValueError(f"nonfinite JSON: {value}")
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("incomplete fenced JSON")
        text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError("JSON top level must be an object")
    return value


def _generate_valid_json(client, cfg: PartNetConfig, path: Path, messages: list, validator, fingerprint: str) -> dict:
    """At most max_attempts returned completions over all reruns, not per rerun.

    Every raw completion is atomically saved BEFORE it is parsed. Retry feedback
    contains only schema errors; no GT geometry or adjacency is provided.
    """
    record = _read_cache(path, fingerprint) or {"fingerprint": fingerprint, "status": "pending", "attempts": [], "transport_errors": []}
    if record.get("status") == "valid":
        return record
    for attempt in record["attempts"]:
        if "validation" not in attempt:
            try:
                attempt["result"] = _parse_json_reply(attempt["raw_text"])
                attempt["validation"] = validator(attempt["result"])
                if attempt.get("finish_reason") not in ("stop", None):
                    attempt["validation"] = {"valid": False, "errors": ["finish_reason:" + str(attempt["finish_reason"])]}
            except (ValueError, TypeError, KeyError) as error:
                attempt["validation"] = {"valid": False, "errors": [_safe_error(error)]}
            if attempt["validation"]["valid"]:
                record.update(status="valid", result=attempt["result"], attempt_count=len(record["attempts"]))
                save_json(record, path)
                return record
    while len(record["attempts"]) < cfg.max_attempts:
        request_messages = copy.deepcopy(messages)
        if record["attempts"]:
            previous = record["attempts"][-1]
            request_messages.append({"role": "assistant", "content": previous["raw_text"] or "{}"})
            feedback = "Correct only the invalid JSON/schema fields listed below. Keep the original fixed inventory and task. Do not optimize geometry using feedback. Return the complete JSON object. Errors: " + json.dumps(previous["validation"]["errors"], ensure_ascii=False)
            request_messages.append({"role": "user", "content": feedback})
        request = {"model": cfg.model, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens, "response_format": {"type": "json_object"}, "messages": request_messages, "stream": False}
        if cfg.extra_body:
            request["extra_body"] = copy.deepcopy(cfg.extra_body)
        response = None
        for transport_index in range(cfg.transport_retries + 1):
            try:
                response = client.chat.completions.create(**request)
                break
            except Exception as error:
                record["transport_errors"].append({"error": _safe_error(error), "time": time.time()})
                record["status"] = "transport_error"
                save_json(record, path)
                status = getattr(error, "status_code", None)
                if status in (400, 401, 402, 403, 404, 422):
                    raise RuntimeError(f"API 请求被拒绝，未自动更换模型/配置。{_safe_error(error)}") from error
                if transport_index < cfg.transport_retries:
                    time.sleep(min(2 ** transport_index, 8))
        if response is None:
            record["attempt_count"] = len(record["attempts"])
            return record
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif usage is not None and not isinstance(usage, dict):
            usage = str(usage)
        attempt = {"attempt": len(record["attempts"]) + 1, "raw_text": choice.message.content or "", "finish_reason": choice.finish_reason, "returned_model": getattr(response, "model", None), "system_fingerprint": getattr(response, "system_fingerprint", None), "usage": usage, "request_messages": request_messages}
        record["attempts"].append(attempt)
        record["request_config"] = cfg.request_config()
        save_json(record, path)
        try:
            result = _parse_json_reply(attempt["raw_text"])
            validation = validator(result)
            if choice.finish_reason not in ("stop", None):
                validation = {"valid": False, "errors": ["finish_reason:" + str(choice.finish_reason)]}
            attempt["result"] = result
            attempt["validation"] = validation
        except (ValueError, TypeError, KeyError) as error:
            attempt["validation"] = {"valid": False, "errors": [_safe_error(error)]}
        record["attempt_count"] = len(record["attempts"])
        if attempt["validation"]["valid"]:
            record.update(status="valid", result=attempt["result"])
            save_json(record, path)
            return record
        record["status"] = "invalid"
        save_json(record, path)
    record["status"] = "exhausted"
    record["attempt_count"] = len(record["attempts"])
    save_json(record, path)
    return record


def _run_jobs(jobs: list, worker, cfg: PartNetConfig, title: str) -> dict:
    results = {}
    if not jobs:
        return results
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        try:
            for done, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                results[job["object_id"]] = future.result()
                if done % max(1, cfg.show_every) == 0 or done == len(jobs):
                    print(f"{title}: {done}/{len(jobs)}")
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return results


def load_partnet_inputs(cfg: PartNetConfig) -> dict:
    """Only essential integrity gates. No caption regeneration or calibration."""
    if cfg.near_contact_threshold != 0.003 or cfg.zero_tolerance != 1e-9:
        raise ValueError("本版本使用冻结的 adjacency 阈值 0.003 / 1e-9，不在 benchmark 上重新调参。")
    names = {"manifest": "partnet_benchmark_manifest.json", "descriptions": "partnet_descriptions_880.json", "adjacency": "partnet_physical_adjacency_880.json"}
    missing = [name for name in names.values() if not cfg.project_path(name).is_file()]
    if missing:
        raise FileNotFoundError("请把已完成的文件放在 project_dir：" + ", ".join(missing))
    raw = {key: load_json(cfg.project_path(name)) for key, name in names.items()}
    indices = {key: _index_objects(value, names[key]) for key, value in raw.items()}
    ids = set(indices["manifest"])
    if cfg.expected_objects is not None and len(ids) != cfg.expected_objects:
        raise ValueError(f"Manifest 有 {len(ids)} 个对象，预期 {cfg.expected_objects}")
    for key in ("descriptions", "adjacency"):
        if set(indices[key]) != ids:
            raise ValueError(f"{names[key]} 的 object IDs 不等于冻结 manifest")
    objects = []
    for item in raw["manifest"]["objects"]:
        oid = str(item["object_id"])
        description = indices["descriptions"][oid]
        adjacency = indices["adjacency"][oid]
        if item["category"] != description["category"] or item["category"] != adjacency["category"]:
            raise ValueError(f"category mismatch: {oid}")
        parts = build_fixed_shared_parts(description["source"])
        if len(parts) != item["geometry_part_count"] or len(adjacency["nodes"]) != len(parts):
            raise ValueError(f"fixed part count mismatch: {oid}")
        if not isinstance(description["description"], str) or not description["description"].strip():
            raise ValueError(f"empty frozen description: {oid}")
        threshold = adjacency.get("near_contact_threshold", raw["adjacency"].get("near_contact_threshold"))
        if threshold is None or abs(float(threshold) - cfg.near_contact_threshold) > 1e-12:
            raise ValueError(f"adjacency threshold mismatch: {oid}")
        node_ids = {node["partnet_id"] for node in adjacency["nodes"]}
        if len(node_ids) != len(adjacency["nodes"]):
            raise ValueError(f"duplicate adjacency node ID: {oid}")
        seen_edges = set()
        for edge in adjacency["edges"]:
            a, b = edge["partnet_id_a"], edge["partnet_id_b"]
            key = frozenset((a, b))
            distance = float(edge["exact_mesh_distance"])
            if a not in node_ids or b not in node_ids or a == b or key in seen_edges or not np.isfinite(distance) or not 0 <= distance <= cfg.near_contact_threshold:
                raise ValueError(f"invalid/duplicate reference edge: {oid}")
            seen_edges.add(key)
        objects.append({"object_id": oid, "category": item["category"], "description": description["description"], "parts": parts})
    if cfg.object_limit is not None:
        if cfg.object_limit < 1:
            raise ValueError("object_limit must be positive or None")
        objects = objects[:cfg.object_limit]
    manifest_info = {"pipeline_version": PIPELINE_VERSION, "input_hashes": {key: content_hash(value) for key, value in raw.items()}, "selected_ids": [x["object_id"] for x in objects], "generation": cfg.request_config(), "max_attempts": cfg.max_attempts}
    path = cfg.output_path("run_manifest.json")
    if path.exists() and load_json(path) != manifest_info:
        raise ValueError("run_manifest 与当前输入/LLM 配置不一致。新实验请换 output_dir，不覆盖旧实验。")
    save_json(manifest_info, path)
    print(f"冻结输入: {len(objects)} objects；描述/邻接图只读，不调用 LLM。")
    return {"objects": objects, "manifest": raw["manifest"], "adjacency": indices["adjacency"], "description_data": raw["descriptions"], "input_hashes": manifest_info["input_hashes"]}


# =============================================================================
# 2. Fixed inventory and shared connection-only generation
# =============================================================================
def build_fixed_shared_parts(source: dict) -> list:
    parts = []
    counts = Counter()
    for item in source["parts"]:
        name = str(item["name"]).strip()
        count = item["count"]
        if not name or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Invalid fixed inventory name/count")
        for _ in range(count):
            parts.append({"part_id": f"part_{len(parts)}", "name": name, "instance_index": counts[name]})
            counts[name] += 1
    if not parts:
        raise ValueError("Fixed inventory is empty")
    return parts


def validate_shared_connections(parts: list, result: dict) -> dict:
    errors = []
    if not isinstance(result, dict):
        return {"valid": False, "errors": ["result_not_dict"]}
    ids = [p.get("part_id") for p in parts]
    if not ids or any(not isinstance(pid, str) for pid in ids) or len(ids) != len(set(ids)):
        return {"valid": False, "errors": ["invalid_fixed_part_ids"]}
    root = result.get("root_part_id")
    edges = result.get("connections")
    if not isinstance(root, str) or root not in ids:
        errors.append("invalid_root")
    if not isinstance(edges, list):
        return {"valid": False, "errors": errors + ["connections_not_list"]}
    if len(edges) != len(ids) - 1:
        errors.append("wrong_connection_count")
    children = {pid: [] for pid in ids}
    indegree = {pid: 0 for pid in ids}
    seen = set()
    for edge in edges:
        if not isinstance(edge, dict):
            errors.append("connection_not_dict")
            continue
        parent, child = edge.get("parent_part_id"), edge.get("child_part_id")
        if not isinstance(parent, str) or not isinstance(child, str) or parent not in children or child not in children:
            errors.append("unknown_endpoint")
            continue
        if parent == child or (parent, child) in seen:
            errors.append("self_or_duplicate_edge")
            continue
        seen.add((parent, child))
        children[parent].append(child)
        indegree[child] += 1
    for pid in ids:
        expected = 0 if pid == root else 1
        if indegree[pid] != expected:
            errors.append(f"invalid_indegree:{pid}")
    reached, stack = set(), [root] if isinstance(root, str) and root in children else []
    while stack:
        node = stack.pop()
        if node not in reached:
            reached.add(node)
            stack.extend(children[node])
    if len(reached) != len(ids):
        errors.append("not_connected_from_root")
    # With N-1 distinct edges, root indegree 0, every other indegree 1,
    # and complete root reachability, a directed cycle is impossible.
    return {"valid": not errors, "errors": sorted(set(errors)), "connection_count": len(edges), "root_part_id": root}


def validate_shared_part_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or not isinstance(spec.get("parts"), list):
        return {"valid": False, "errors": ["invalid_spec"]}
    parts = spec["parts"]
    if any(not isinstance(p, dict) for p in parts):
        return {"valid": False, "errors": ["part_not_dict"]}
    check = validate_shared_connections(parts, spec)
    errors = list(check["errors"])
    if [p.get("part_id") for p in parts] != [f"part_{i}" for i in range(len(parts))]:
        errors.append("noncanonical_part_id_order")
    groups = defaultdict(list)
    for p in parts:
        if not isinstance(p.get("name"), str) or not p["name"].strip():
            errors.append("invalid_part_name")
        groups[str(p.get("name"))].append(p.get("instance_index"))
    for name, values in groups.items():
        if any(type(v) is not int for v in values) or sorted(values) != list(range(len(values))):
            errors.append("invalid_instance_indices:" + name)
    return {"valid": not errors, "errors": sorted(set(errors)), "part_count": len(parts), "connection_count": check.get("connection_count", 0)}


def assemble_shared_part_spec(category: str, parts: list, connection_result: dict) -> dict:
    return {"object_category": category, "root_part_id": connection_result["root_part_id"], "parts": copy.deepcopy(parts), "connections": copy.deepcopy(connection_result["connections"])}
def build_shared_connection_prompt(
    description,
    parts
):
    parts_json=json.dumps(
        parts,
        ensure_ascii=False,
        indent=2
    )

    return f"""
Infer one plausible directed structural tree over the FIXED physical parts below.

The physical part inventory is authoritative and immutable.
You must NOT add, remove, rename, merge, or split parts.

Object description:
{description}

Fixed physical parts:
{parts_json}

Return exactly one JSON object with this schema:

{{
  "root_part_id": "part_0",
  "connections": [
    {{
      "parent_part_id": "part_0",
      "child_part_id": "part_1"
    }}
  ]
}}

Requirements:
- Use only the supplied part_id values.
- Choose exactly one supplied part as root.
- root_part_id must have no parent.
- Every other part must have exactly one parent.
- Output exactly N-1 connections for N supplied parts.
- Every supplied part must be reachable from the root.
- The graph must be connected.
- The graph must be acyclic.
- Do not create semantic grouping nodes.
- Semantic organization in the description may guide parent-child choices.
- Prefer structurally central or supporting physical parts as parents when reasonable.
- Repeated equivalent parts may share the same parent.
- Do not output coordinates, dimensions, rotations, anchors, bounding boxes, mesh data, or PartNet IDs.
- Do not output explanations.
- Do not output markdown.

Before returning the JSON, internally verify:
1. exactly one root;
2. exactly N-1 edges;
3. every non-root node appears exactly once as child;
4. root never appears as child;
5. all nodes are reachable from root;
6. no cycle exists.

Return JSON only.
""".strip()


def _connection_fingerprint(row: dict, cfg: PartNetConfig) -> str:
    return content_hash({"kind": "shared-connection-v1", "description": row["description"], "parts": row["parts"], "prompt": build_shared_connection_prompt(row["description"], row["parts"]), "request": cfg.request_config(), "max_attempts": cfg.max_attempts})


def _import_existing_shared(inputs: dict, cfg: PartNetConfig) -> None:
    """Import ONLY the finalized connection-only run, never earlier development specs."""
    lookup = {row["object_id"]: row for row in inputs["objects"]}
    for filename in ("partnet_shared_part_specs_880_final.json", "partnet_shared_part_specs_880_final_checkpoint.json"):
        path = cfg.project_path(filename)
        if not path.is_file():
            continue
        data = load_json(path)
        if data.get("model") != cfg.model or float(data.get("temperature", 0.0)) != cfg.temperature:
            raise ValueError(f"{filename} 的 model/temperature 与配置不同；请保持原 shared 配置，或用独立实验目录明确重新运行。")
        for oid, old in _index_objects(data, filename).items():
            if oid not in lookup:
                continue
            row = lookup[oid]
            spec = old.get("spec")
            if old.get("description") != row["description"]:
                raise ValueError(f"Frozen description differs from existing shared spec: {oid}")
            if spec is not None and spec.get("parts") != row["parts"]:
                raise ValueError(f"Existing FINAL shared inventory differs from fixed parts: {oid}")
            destination = cfg.output_path(f"shared/{oid}.json")
            fingerprint = _connection_fingerprint(row, cfg)
            cached = _read_cache(destination, fingerprint)
            valid = spec is not None and validate_shared_part_spec(spec)["valid"]
            if cached is not None:
                if valid and cached.get("status") == "valid" and cached.get("result") != {"root_part_id": spec["root_part_id"], "connections": spec["connections"]}:
                    raise ValueError(f"Conflicting frozen shared trees: {oid}")
                continue
            attempts = copy.deepcopy(old.get("attempts", []))
            for attempt in attempts:
                if "raw_text" not in attempt:
                    attempt["raw_text"] = json.dumps(attempt.get("result"), ensure_ascii=False)
                    attempt["raw_text_reconstructed"] = True
            result = None if spec is None else {"root_part_id": spec["root_part_id"], "connections": spec["connections"]}
            count = int(old.get("attempt_count", len(attempts)))
            record = {"fingerprint": fingerprint, "status": "valid" if valid else ("exhausted" if count >= cfg.max_attempts else "invalid"), "result": result, "attempt_count": count, "attempts": attempts, "transport_errors": [], "imported_from": filename, "legacy_record": old, "raw_response_available": False, "legacy_generation_config": {k: data.get(k) for k in ("model", "temperature", "max_attempts", "protocol")}}
            save_json(record, destination)


def run_partnet_shared_specs(inputs: dict, cfg: PartNetConfig, client=None) -> dict:
    _import_existing_shared(inputs, cfg)
    jobs = []
    for row in inputs["objects"]:
        fingerprint = _connection_fingerprint(row, cfg)
        cached = _read_cache(cfg.output_path(f"shared/{row['object_id']}.json"), fingerprint)
        if cached is None or cached.get("status") not in ("valid", "exhausted"):
            jobs.append(row)
    if jobs and cfg.allow_api:
        client = client or create_deepseek_client(base_url=cfg.base_url, timeout=cfg.timeout)
        def worker(row):
            messages = [{"role": "user", "content": build_shared_connection_prompt(row["description"], row["parts"])}]
            return _generate_valid_json(client, cfg, cfg.output_path(f"shared/{row['object_id']}.json"), messages, lambda result: validate_shared_connections(row["parts"], result), _connection_fingerprint(row, cfg))
        _run_jobs(jobs, worker, cfg, "Shared connections")
    results = {}
    for row in inputs["objects"]:
        record = _read_cache(cfg.output_path(f"shared/{row['object_id']}.json"), _connection_fingerprint(row, cfg))
        if record is None:
            results[row["object_id"]] = {**row, "status": "missing", "spec": None, "attempt_count": 0}
            continue
        spec = assemble_shared_part_spec(row["category"], row["parts"], record["result"]) if record["status"] == "valid" else None
        results[row["object_id"]] = {**row, "spec": spec, "status": record["status"], "attempt_count": record.get("attempt_count", 0), "first_attempt_valid": bool(record.get("attempts") and record["attempts"][0].get("validation", {}).get("valid")), "imported_from": record.get("imported_from")}
    save_json({"protocol": "fixed inventory + shared connection-only tree", "objects": list(results.values())}, cfg.output_path("shared_specs.json"))
    valid = sum(row["status"] == "valid" for row in results.values())
    print(f"Shared specs: valid={valid}/{len(results)}；历史结果复用，失败不补造树。")
    return results


# =============================================================================
# 3. Current PartNet GT preprocessing (geometry only; never sent to the LLM)
# =============================================================================
def load_obj_referenced_geometry(obj_path: str | Path):
    vertices, faces = [], []
    with Path(obj_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            values = line.split()
            if not values:
                continue
            if values[0] == "v" and len(values) >= 4:
                vertices.append([float(v) for v in values[1:4]])
            elif values[0] == "f":
                face = []
                for item in values[1:]:
                    if item.startswith("#"):
                        break
                    index = int(item.split("/")[0])
                    if index == 0:
                        raise ValueError(f"OBJ uses invalid zero index: {obj_path}")
                    face.append(index - 1 if index > 0 else len(vertices) + index)
                for i in range(1, len(face) - 1):
                    faces.append([face[0], face[i], face[i + 1]])
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=int).reshape(-1, 3)
    if not len(v) or not len(f):
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    if not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
        raise ValueError(f"Invalid referenced geometry: {obj_path}")
    used, inverse = np.unique(f, return_inverse=True)
    return v[used], inverse.reshape(-1, 3)


def load_obj_referenced_vertices(obj_path: str | Path) -> np.ndarray:
    return load_obj_referenced_geometry(obj_path)[0]


def get_partnet_exclusive_objs(node):
    node_objs = set(node.get('objs', []) or [])
    child_objs = set()
    for child in node.get('children', []):
        child_objs.update(child.get('objs', []) or [])
    return sorted(node_objs - child_objs)

def build_partnet_gt_tree(object_dir, annotation_tree, include_vertices=False):
    root = annotation_tree[0] if isinstance(annotation_tree, list) else annotation_tree

    def build_node(node):
        exclusive_objs = get_partnet_exclusive_objs(node)
        gt_node = {}
        gt_node['partnet_id'] = node.get('id')
        gt_node['name'] = node.get('name')
        gt_node['text'] = node.get('text')
        gt_node['exclusive_objs'] = exclusive_objs
        gt_node['has_geometry'] = len(exclusive_objs) > 0
        gt_node['children'] = []
        if gt_node['has_geometry']:
            vertex_sets = []
            for obj_name in exclusive_objs:
                obj_path = os.path.join(object_dir, 'objs', obj_name + '.obj')
                if not os.path.exists(obj_path):
                    raise FileNotFoundError(obj_path)
                vertices = load_obj_referenced_vertices(obj_path)
                if len(vertices) > 0:
                    vertex_sets.append(vertices)
            if len(vertex_sets) > 0:
                vertices = np.concatenate(vertex_sets, axis=0)
                vertices = np.unique(vertices, axis=0)
                bbox_min = np.min(vertices, axis=0)
                bbox_max = np.max(vertices, axis=0)
                center = (bbox_min + bbox_max) / 2
                bbox_size = bbox_max - bbox_min
                gt_node['vertex_count'] = int(len(vertices))
                gt_node['bbox_min'] = bbox_min.tolist()
                gt_node['bbox_max'] = bbox_max.tolist()
                gt_node['center'] = center.tolist()
                gt_node['bbox_size'] = bbox_size.tolist()
                if include_vertices:
                    gt_node['vertices'] = vertices
            else:
                raise ValueError('No referenced geometry for part: ' + str(gt_node['partnet_id']))
        for child in node.get('children', []):
            gt_node['children'].append(build_node(child))
        return gt_node
    return build_node(root)

def load_partnet_gt_object(partnet_root, object_id, include_vertices=False):
    object_dir = os.path.join(partnet_root, str(object_id))
    meta_path = os.path.join(object_dir, 'meta.json')
    merged_path = os.path.join(object_dir, 'result_after_merging.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    with open(merged_path, 'r', encoding='utf-8') as f:
        annotation_tree = json.load(f)
    gt_tree = build_partnet_gt_tree(object_dir, annotation_tree, include_vertices=include_vertices)
    result = {}
    result['object_id'] = str(object_id)
    result['category'] = meta.get('model_cat')
    result['model_id'] = meta.get('model_id')
    result['meta'] = meta
    result['tree'] = gt_tree
    return result

def project_partnet_geometry_tree(gt_tree):
    projected_root = {}
    projected_root['partnet_id'] = None
    projected_root['name'] = 'object'
    projected_root['text'] = 'object'
    projected_root['has_geometry'] = False
    projected_root['children'] = []

    def traverse(node, projected_parent, path):
        current_path = path + [node.get('name')]
        next_parent = projected_parent
        if node.get('has_geometry', False):
            projected = {}
            projected['partnet_id'] = node.get('partnet_id')
            projected['name'] = node.get('name')
            projected['text'] = node.get('text')
            projected['has_geometry'] = True
            projected['exclusive_objs'] = list(node.get('exclusive_objs', []))
            projected['center'] = list(node.get('center', []))
            projected['bbox_size'] = list(node.get('bbox_size', []))
            projected['semantic_path'] = list(current_path)
            projected['children'] = []
            projected_parent['children'].append(projected)
            next_parent = projected
        for child in node.get('children', []):
            traverse(child, next_parent, current_path)
    traverse(gt_tree, projected_root, [])
    return projected_root

def collect_partnet_geometry_nodes(tree):
    parts = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if node.get('has_geometry', False):
            parts.append(node)
        for child in reversed(node.get('children', [])):
            stack.append(child)
    return parts

def build_partnet_geometry_index(tree):
    index = {}
    for node in collect_partnet_geometry_nodes(tree):
        partnet_id = node.get('partnet_id')
        if partnet_id is not None:
            index[partnet_id] = node
    return index

def compute_partnet_tree_bounds(gt_tree):
    parts = collect_partnet_geometry_nodes(gt_tree)
    bbox_mins = []
    bbox_maxs = []
    for part in parts:
        if 'bbox_min' not in part or 'bbox_max' not in part:
            continue
        bbox_mins.append(np.array(part['bbox_min'], dtype=float))
        bbox_maxs.append(np.array(part['bbox_max'], dtype=float))
    if len(bbox_mins) == 0:
        return None
    bbox_min = np.min(np.stack(bbox_mins, axis=0), axis=0)
    bbox_max = np.max(np.stack(bbox_maxs, axis=0), axis=0)
    center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min
    scale = np.max(bbox_size)
    result = {}
    result['bbox_min'] = bbox_min.tolist()
    result['bbox_max'] = bbox_max.tolist()
    result['center'] = center.tolist()
    result['bbox_size'] = bbox_size.tolist()
    result['scale'] = float(scale)
    return result

def convert_partnet_vector_to_canonical(vector):
    vector = np.array(vector, dtype=float)
    return np.array([-vector[0], vector[2], vector[1]], dtype=float)

def convert_partnet_gt_tree_to_canonical(gt_tree):

    def traverse(node):
        if node.get('has_geometry', False):
            center = convert_partnet_vector_to_canonical(node['center'])
            size = np.array(node['bbox_size'], dtype=float)
            size = np.array([size[0], size[2], size[1]], dtype=float)
            node['center'] = center.tolist()
            node['bbox_size'] = size.tolist()
            node['bbox_min'] = (center - size / 2).tolist()
            node['bbox_max'] = (center + size / 2).tolist()
            if 'vertices' in node:
                vertices = np.array(node['vertices'], dtype=float)
                converted = np.stack([-vertices[:, 0], vertices[:, 2], vertices[:, 1]], axis=1)
                node['vertices'] = converted
        for child in node.get('children', []):
            traverse(child)
    traverse(gt_tree)
    return gt_tree

def normalize_partnet_gt_tree(gt_tree):
    bounds = compute_partnet_tree_bounds(gt_tree)
    if bounds is None:
        return gt_tree
    object_center = np.array(bounds['center'], dtype=float)
    scale = float(bounds['scale'])
    if scale <= 0:
        return gt_tree

    def traverse(node):
        if node.get('has_geometry', False):
            center = np.array(node['center'], dtype=float)
            size = np.array(node['bbox_size'], dtype=float)
            center = (center - object_center) / scale
            size = size / scale
            node['center'] = center.tolist()
            node['bbox_size'] = size.tolist()
            node['bbox_min'] = (center - size / 2).tolist()
            node['bbox_max'] = (center + size / 2).tolist()
            if 'vertices' in node:
                vertices = np.array(node['vertices'], dtype=float)
                node['vertices'] = (vertices - object_center) / scale
        for child in node.get('children', []):
            traverse(child)
    traverse(gt_tree)
    gt_tree['object_center_before_normalization'] = object_center.tolist()
    gt_tree['object_scale_before_normalization'] = scale
    return gt_tree

def prepare_partnet_gt_object(partnet_root, object_id, include_vertices=False):
    gt = load_partnet_gt_object(partnet_root, object_id, include_vertices=include_vertices)
    convert_partnet_gt_tree_to_canonical(gt['tree'])
    normalize_partnet_gt_tree(gt['tree'])
    gt['geometry_tree'] = project_partnet_geometry_tree(gt['tree'])
    return gt

def canonicalize_part_semantic_name(name):
    if name is None:
        return ''
    name = str(name).strip().lower()
    name = name.replace('-', '_')
    name = name.replace(' ', '_')
    while '__' in name:
        name = name.replace('__', '_')
    return name


def prepare_partnet_gt_cache(inputs: dict, cfg: PartNetConfig) -> dict:
    results, failures = {}, []
    for i, row in enumerate(inputs["objects"], 1):
        oid = row["object_id"]
        reference = inputs["adjacency"][oid]
        fingerprint = content_hash({"gt_parser": "merged-exclusive-canonical-v1", "object_id": oid, "fixed_parts": row["parts"], "adjacency": reference})
        path = cfg.output_path(f"gt/{oid}.json")
        try:
            cached = _read_cache(path, fingerprint)
            if cached is None:
                gt = prepare_partnet_gt_object(cfg.partnet_root, oid)
                parts = collect_partnet_geometry_nodes(gt["tree"])
                expected = Counter(canonicalize_part_semantic_name(p["name"]) for p in row["parts"])
                actual = Counter(canonicalize_part_semantic_name(p["name"]) for p in parts)
                if expected != actual or gt["category"] != row["category"]:
                    raise ValueError(f"GT inventory/category mismatch: {oid}")
                ids = [p["partnet_id"] for p in parts]
                if len(ids) != len(set(ids)) or set(ids) != {p["partnet_id"] for p in reference["nodes"]}:
                    raise ValueError(f"GT and adjacency node identity mismatch: {oid}")
                for part in parts:
                    if not np.isfinite(part["center"]).all() or not np.isfinite(part["bbox_size"]).all() or np.any(np.asarray(part["bbox_size"]) < 0):
                        raise ValueError(f"Invalid GT box: {oid}")
                cached = {"fingerprint": fingerprint, "object_id": oid, "category": row["category"], "parts": [{"partnet_id": p["partnet_id"], "name": p["name"], "center": p["center"], "bbox_size": p["bbox_size"], "world_rotation": np.eye(3).tolist()} for p in parts], "edges": reference["edges"], "normalization_center": gt["tree"]["object_center_before_normalization"], "normalization_scale": gt["tree"]["object_scale_before_normalization"]}
                save_json(cached, path)
            results[oid] = cached
        except (OSError, ValueError, KeyError) as error:
            failures.append({"object_id": oid, "error": _safe_error(error)})
        if i % max(cfg.show_every, 100) == 0 or i == len(inputs["objects"]):
            print(f"GT cache: {i}/{len(inputs['objects'])}")
    save_json({"failures": failures}, cfg.output_path("gt_failures.json"))
    if failures:
        raise RuntimeError(f"GT cache 有 {len(failures)} 个失败，见 gt_failures.json；已完成对象已保存，修复后重跑此格即可。")
    return results


# =============================================================================
# 4. Three representations, one shared input and one decoding configuration
# =============================================================================
_COMMON_REPRESENTATION_PROMPT = """Generate a plausible structured 3D bounding-box layout for the given object.
The user message is identical for all representations. It contains the frozen
natural-language description and the fixed shared part specification.
Use exactly those part IDs once each. Do not add, remove, rename, merge or split
parts. The shared root and directed parent-child edges are immutable.
All supplied parts have geometry, including the root. A category-like part name
may denote residual geometry: do not delete it or replace it by an object envelope.
Coordinate system: +x right, +y front, +z up; right handed.
All quaternions are [w,x,y,z], relative to the stated frame. Identity is [1,0,0,0].
BBox sizes are positive [x_length,y_length,z_length] along the part's local axes.
Choose a common object scale so that the assembled longest extent is about 1.
Respect semantic arrangement, repeated-part regularity and the given structural
connections. Connected parts should be plausibly attached; avoid floating parts
and excessive overlap. Do not emit an extra object/group node.
Return only JSON: {"parts":[...]}.
Each entry contains part_id and bbox_size plus ONLY the fields below.
Do not output part names, semantic data, parent IDs, children, or world_rotation:
these identities/edges are supplied by the shared specification, not regenerated.
"""
_REPRESENTATION_FIELDS = {
    "global": """GLOBAL OBB: Every part contains center (global xyz) and rotation (absolute quaternion).
Example entry: {"part_id":"part_0","bbox_size":[0.4,0.4,0.1],"center":[0,0,0],"rotation":[1,0,0,0]}.
Do not emit relative_position, relative_rotation, anchors, or offset.
""",
    "local": """HIERARCHY LOCAL: The shared root contains center (global xyz) and rotation (absolute quaternion).
Every non-root contains relative_position (child-center displacement in the DIRECT
parent's local frame, in object length units) and relative_rotation (local quaternion).
The transform is Rc=Rp*Rlocal, cc=cp+Rp*relative_position.
Root example: {"part_id":"part_0","bbox_size":[0.4,0.4,0.1],"center":[0,0,0],"rotation":[1,0,0,0]}.
Non-root example: {"part_id":"part_1","bbox_size":[0.1,0.1,0.3],"relative_position":[0.15,0.15,-0.2],"relative_rotation":[1,0,0,0]}.
Do not emit anchors or offset; do not use global centers for non-root parts.
""",
    "anchor": """ANCHOR TREE: The shared root contains center (global xyz) and rotation (absolute quaternion).
Every non-root contains parent_anchor, child_anchor, offset and rotation.
Anchors are fractions of the respective LOCAL bbox size, each coordinate in [-0.5,0.5].
Choose semantically meaningful surface attachment locations; faces are at +/-0.5.
rotation is the child's quaternion relative to its DIRECT parent.
offset is a residual displacement in the DIRECT PARENT'S LOCAL FRAME, in object
length units (not bbox fractions). Prefer [0,0,0]; do not use offset to bypass anchors.
Rc=Rp*Rlocal; cc=cp+Rp*(parent_size*parent_anchor+offset)-Rc*(child_size*child_anchor).
Root example: {"part_id":"part_0","bbox_size":[0.4,0.4,0.1],"center":[0,0,0],"rotation":[1,0,0,0]}.
Non-root example: {"part_id":"part_1","bbox_size":[0.1,0.1,0.3],"parent_anchor":[0.375,0.375,-0.5],"child_anchor":[0,0,0.5],"offset":[0,0,0],"rotation":[1,0,0,0]}.
Do not emit relative_position, relative_rotation, or global centers for non-root parts.
"""
}


def build_partnet_representation_messages(description: str, spec: dict, representation: str) -> list:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    user_message = json.dumps({"description": description, "shared_part_spec": spec}, ensure_ascii=False, indent=2)
    return [{"role": "system", "content": _COMMON_REPRESENTATION_PROMPT + _REPRESENTATION_FIELDS[representation]}, {"role": "user", "content": user_message}]


def quaternion_to_matrix(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid quaternion: expected finite nonzero [w,x,y,z]")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def validate_partnet_representation(raw: dict, spec: dict, representation: str) -> dict:
    errors = []
    if not isinstance(raw, dict) or set(raw) != {"parts"} or not isinstance(raw["parts"], list):
        return {"valid": False, "errors": ["expected_only_parts_list"]}
    ids = [p["part_id"] for p in spec["parts"]]
    parts = raw["parts"]
    actual = [p.get("part_id") for p in parts if isinstance(p, dict)]
    if len(actual) != len(parts) or any(not isinstance(pid, str) for pid in actual) or Counter(actual) != Counter(ids):
        return {"valid": False, "errors": ["part_ids_must_match_fixed_inventory_exactly_once"]}
    for part in parts:
        pid = part["part_id"]
        if representation == "global" or pid == spec["root_part_id"]:
            fields = {"center": 3, "rotation": 4, "bbox_size": 3}
        elif representation == "local":
            fields = {"relative_position": 3, "relative_rotation": 4, "bbox_size": 3}
        else:
            fields = {"parent_anchor": 3, "child_anchor": 3, "offset": 3, "rotation": 4, "bbox_size": 3}
        if set(part) != set(fields) | {"part_id"}:
            errors.append(f"{pid}:required_fields={sorted(set(fields) | {'part_id'})}")
        for field_name, length in fields.items():
            value = part.get(field_name)
            if not isinstance(value, list) or len(value) != length or any(type(x) not in (int, float) for x in value):
                errors.append(f"{pid}:{field_name}:numeric_vector_{length}_required")
                continue
            vector = np.asarray(value, dtype=float)
            if not np.isfinite(vector).all():
                errors.append(f"{pid}:{field_name}:nonfinite")
            elif field_name == "bbox_size" and np.any(vector <= 0):
                errors.append(f"{pid}:bbox_size_must_be_positive")
            elif field_name in ("rotation", "relative_rotation") and np.linalg.norm(vector) < 1e-12:
                errors.append(f"{pid}:zero_quaternion")
            elif field_name in ("parent_anchor", "child_anchor") and np.any(np.abs(vector) > 0.500001):
                errors.append(f"{pid}:{field_name}:outside_bbox")
    return {"valid": not errors, "errors": errors}


def _representation_fingerprint(row: dict, cfg: PartNetConfig, representation: str) -> str:
    return content_hash({"schema": REPRESENTATION_SCHEMA_VERSION, "messages": build_partnet_representation_messages(row["description"], row["spec"], representation), "request": cfg.request_config(), "max_attempts": cfg.max_attempts})


def run_partnet_generation(shared: dict, cfg: PartNetConfig, client=None) -> dict:
    results = {name: {} for name in REPRESENTATIONS}
    for representation in REPRESENTATIONS:
        valid_rows = [row for row in shared.values() if row["status"] == "valid"]
        jobs = []
        for row in valid_rows:
            fingerprint = _representation_fingerprint(row, cfg, representation)
            cached = _read_cache(cfg.output_path(f"generation/{representation}/{row['object_id']}.json"), fingerprint)
            if cached is None or cached.get("status") not in ("valid", "exhausted"):
                jobs.append(row)
        if jobs and cfg.allow_api:
            client = client or create_deepseek_client(base_url=cfg.base_url, timeout=cfg.timeout)
            def worker(row):
                path = cfg.output_path(f"generation/{representation}/{row['object_id']}.json")
                messages = build_partnet_representation_messages(row["description"], row["spec"], representation)
                return _generate_valid_json(client, cfg, path, messages, lambda raw: validate_partnet_representation(raw, row["spec"], representation), _representation_fingerprint(row, cfg, representation))
            _run_jobs(jobs, worker, cfg, representation)
        for row in valid_rows:
            record = _read_cache(cfg.output_path(f"generation/{representation}/{row['object_id']}.json"), _representation_fingerprint(row, cfg, representation))
            results[representation][row["object_id"]] = record or {"status": "missing", "attempt_count": 0, "attempts": []}
        print(f"{representation}: valid={sum(r['status']=='valid' for r in results[representation].values())}/{len(shared)}")
    return results


_BOX_SIGNS = np.asarray(list(itertools.product((-0.5, 0.5), repeat=3)), dtype=float)


def partnet_box_corners(part: dict) -> np.ndarray:
    return np.asarray(part["center"]) + (_BOX_SIGNS * np.asarray(part["bbox_size"])) @ np.asarray(part["world_rotation"]).T


def transform_partnet_representation(raw: dict, spec: dict, representation: str, normalization: dict | None = None) -> dict:
    """Pure transform from RAW parameters. Does not mutate raw/spec.

    Base objects: center the actual oriented-box union AABB, longest extent=1.
    Perturbations/Random Anchor: supply the base normalization, never refit it.
    """
    check = validate_partnet_representation(raw, spec, representation)
    if not check["valid"]:
        raise ValueError("; ".join(check["errors"]))
    parameters = {p["part_id"]: p for p in raw["parts"]}
    metadata = {p["part_id"]: p for p in spec["parts"]}
    parents = {e["child_part_id"]: e["parent_part_id"] for e in spec["connections"]}
    world, visiting = {}, set()
    def solve(pid):
        if pid in world:
            return world[pid]
        if pid in visiting:
            raise ValueError("cycle in shared tree")
        visiting.add(pid)
        p = parameters[pid]
        size = np.asarray(p["bbox_size"], dtype=float)
        if representation == "global" or pid == spec["root_part_id"]:
            center = np.asarray(p["center"], dtype=float)
            rotation = quaternion_to_matrix(p["rotation"])
        else:
            parent = solve(parents[pid])
            pc, pr, ps = np.asarray(parent["center"]), np.asarray(parent["world_rotation"]), np.asarray(parent["bbox_size"])
            if representation == "local":
                rotation = pr @ quaternion_to_matrix(p["relative_rotation"])
                center = pc + pr @ np.asarray(p["relative_position"])
            else:
                rotation = pr @ quaternion_to_matrix(p["rotation"])
                center = pc + pr @ (ps * np.asarray(p["parent_anchor"]) + np.asarray(p["offset"])) - rotation @ (size * np.asarray(p["child_anchor"]))
        result = {**metadata[pid], "center": center.tolist(), "bbox_size": size.tolist(), "world_rotation": rotation.tolist(), "has_geometry": True}
        world[pid] = result
        visiting.remove(pid)
        return result
    parts = [solve(p["part_id"]) for p in spec["parts"]]
    if normalization is None:
        corners = np.concatenate([partnet_box_corners(p) for p in parts])
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        normalization = {"center": ((lo + hi) / 2).tolist(), "scale": float(np.max(hi - lo))}
    shift = np.asarray(normalization["center"], dtype=float)
    scale = float(normalization["scale"])
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("invalid normalization scale")
    for part in parts:
        part["center"] = ((np.asarray(part["center"]) - shift) / scale).tolist()
        part["bbox_size"] = (np.asarray(part["bbox_size"]) / scale).tolist()
        part["world_aabb_size"] = (np.abs(np.asarray(part["world_rotation"])) @ np.asarray(part["bbox_size"])).tolist()
    return {"parts": parts, "normalization": copy.deepcopy(normalization), "representation": representation}


# =============================================================================
# 5. Offline evaluation: semantic matching, pose, and OBB-aware contact proxies
# =============================================================================
METRIC_DEFINITIONS = {
    "center_error": "Mean L2 center error after independent object centering/longest-extent normalization.",
    "relative_translation_error": "Mean L2 error of canonical-frame center differences over all unordered matched part pairs; no GT rotation is invented.",
    "world_aabb_size_mae": "Mean absolute error of |R_pred|*local_size vs GT AABB extents, in object units; no division by thin GT dimensions.",
    "gt_edge_bbox_margin_mae": "Mean absolute difference of predicted OBB SAT margin and GT AABB SAT margin on reference mesh-adjacency edges. Box proxy, not mesh contact-location error.",
    "gt_edge_obb_sat_gap": "Mean max(SAT margin,0) on GT mesh-adjacency edges. SAT gap is a projection-based distance lower bound, not exact Euclidean mesh distance.",
    "gt_edge_bbox_contact_rate": "Fraction of GT mesh-adjacency edges with predicted OBB SAT margin <= 0.003; overlap counts as success. Bounding-box realization proxy only.",
    "shared_edge_obb_sat_deviation": "Mean absolute signed SAT margin on the identical inferred shared tree edges; penalizes gap and penetration, not GT hierarchy accuracy.",
    "shared_edge_obb_sat_gap": "Mean positive SAT margin on identical inferred shared tree edges.",
    "obb_overlap_pair_fraction": "Fraction of all unordered predicted OBB pairs with SAT margin < -1e-9 (strict overlap).",
    "part_volume_sum": "Sum of local OBB volumes after object normalization. Descriptive compactness quantity, not universal quality score.",
}
PRIMARY_METRICS = ("relative_translation_error", "world_aabb_size_mae", "gt_edge_bbox_margin_mae", "gt_edge_obb_sat_gap", "gt_edge_bbox_contact_rate", "shared_edge_obb_sat_deviation", "obb_overlap_pair_fraction")
HIGHER_IS_BETTER = {"gt_edge_bbox_contact_rate"}


def obb_sat_margin(a: dict, b: dict) -> float:
    """15-axis normalized SAT margin: positive=separation; negative=MTV depth.

    For disjoint boxes this is a separating-projection gap, NOT their Euclidean
    surface distance. For overlapping full-dimensional boxes -margin is the
    translational penetration depth along a separating axis.
    """
    ra, rb = np.asarray(a["world_rotation"]), np.asarray(b["world_rotation"])
    axes = np.vstack((ra.T, rb.T, np.cross(ra.T[:, None, :], rb.T[None, :, :]).reshape(-1, 3)))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-10] / norms[norms > 1e-10, None]
    delta = np.asarray(b["center"]) - np.asarray(a["center"])
    radius_a = np.abs(axes @ ra) @ (np.asarray(a["bbox_size"]) / 2)
    radius_b = np.abs(axes @ rb) @ (np.asarray(b["bbox_size"]) / 2)
    return float(np.max(np.abs(axes @ delta) - radius_a - radius_b))


def match_predicted_parts_to_partnet_gt(predicted_parts: list, gt_tree: dict | list) -> dict:
    """Strict semantic groups; the existing center + 1e-6*size assignment objective.

    Base representations get the same algorithm, not the same permutation. The
    resulting base assignment is held FIXED for perturbations and Random Anchor.
    This is exchangeable-instance alignment, not GT identity recovery.
    """
    gt_parts = gt_tree if isinstance(gt_tree, list) else collect_partnet_geometry_nodes(gt_tree)
    pred_groups, gt_groups = defaultdict(list), defaultdict(list)
    for p in predicted_parts:
        pred_groups[canonicalize_part_semantic_name(p["name"])].append(p)
    for p in gt_parts:
        gt_groups[canonicalize_part_semantic_name(p["name"])].append(p)
    matches, unmatched_predicted, unmatched_gt = [], [], []
    for name in sorted(set(pred_groups) | set(gt_groups)):
        ps, gs = pred_groups[name], gt_groups[name]
        if not ps or not gs:
            unmatched_predicted.extend(p["part_id"] for p in ps)
            unmatched_gt.extend(g["partnet_id"] for g in gs)
            continue
        pc, gc = np.asarray([p["center"] for p in ps]), np.asarray([g["center"] for g in gs])
        psize = np.asarray([p.get("world_aabb_size", p["bbox_size"]) for p in ps])
        gsize = np.asarray([g["bbox_size"] for g in gs])
        center_cost = np.linalg.norm(pc[:, None] - gc[None, :], axis=-1)
        cost = center_cost + 1e-6 * np.linalg.norm(psize[:, None] - gsize[None, :], axis=-1)
        ii, jj = linear_sum_assignment(cost)
        for i, j in zip(ii, jj):
            matches.append({"part_id": ps[i]["part_id"], "partnet_id": gs[j]["partnet_id"], "name": name, "match_type": "unique_semantic" if len(ps)==len(gs)==1 else "permutation_invariant", "center_matching_cost": float(center_cost[i,j]), "assignment_cost": float(cost[i,j])})
        unmatched_predicted.extend(ps[i]["part_id"] for i in range(len(ps)) if i not in set(ii))
        unmatched_gt.extend(gs[j]["partnet_id"] for j in range(len(gs)) if j not in set(jj))
    return {"matches": matches, "unmatched_predicted": unmatched_predicted, "unmatched_gt": unmatched_gt, "matched_count": len(matches), "predicted_count": len(predicted_parts), "gt_count": len(gt_parts)}


def _mean_or_none(values) -> float | None:
    return float(np.mean(values)) if len(values) else None


def evaluate_partnet_layout(layout: dict, spec: dict, gt: dict, cfg: PartNetConfig, frozen_matches: list | None = None) -> dict:
    parts = layout["parts"]
    matched = frozen_matches if frozen_matches is not None else match_predicted_parts_to_partnet_gt(parts, gt["parts"])["matches"]
    if len(matched) != len(parts) or len(matched) != len(gt["parts"]):
        raise ValueError("Incomplete strict semantic correspondence; do not silently evaluate a subset.")
    pindex = {p["part_id"]: p for p in parts}
    gindex = {g["partnet_id"]: g for g in gt["parts"]}
    gt_to_pred = {m["partnet_id"]: m["part_id"] for m in matched}
    if len(gt_to_pred) != len(gt["parts"]):
        raise ValueError("Non one-to-one correspondence")
    ps = [pindex[m["part_id"]] for m in matched]
    gs = [gindex[m["partnet_id"]] for m in matched]
    pc, gc = np.asarray([p["center"] for p in ps]), np.asarray([g["center"] for g in gs])
    psize = np.asarray([p["world_aabb_size"] for p in ps])
    gsize = np.asarray([g["bbox_size"] for g in gs])
    ii, jj = np.triu_indices(len(ps), 1)
    relative_errors = np.linalg.norm((pc[jj] - pc[ii]) - (gc[jj] - gc[ii]), axis=1)
    margins = {frozenset((a["part_id"], b["part_id"])): obb_sat_margin(a, b) for a,b in itertools.combinations(parts, 2)}
    gt_margins, pred_gt_margins = [], []
    for edge in gt["edges"]:
        a, b = edge["partnet_id_a"], edge["partnet_id_b"]
        pred_gt_margins.append(margins[frozenset((gt_to_pred[a], gt_to_pred[b]))])
        gt_margins.append(obb_sat_margin(gindex[a], gindex[b]))
    shared_margins = [margins[frozenset((e["parent_part_id"], e["child_part_id"]))] for e in spec["connections"]]
    metrics = {"center_error": float(np.mean(np.linalg.norm(pc-gc, axis=1))), "relative_translation_error": _mean_or_none(relative_errors), "world_aabb_size_mae": float(np.mean(np.abs(psize-gsize))), "gt_edge_bbox_margin_mae": _mean_or_none(np.abs(np.asarray(pred_gt_margins)-np.asarray(gt_margins))), "gt_edge_obb_sat_gap": _mean_or_none(np.maximum(pred_gt_margins, 0)), "gt_edge_bbox_contact_rate": _mean_or_none(np.asarray(pred_gt_margins) <= cfg.near_contact_threshold), "shared_edge_obb_sat_deviation": _mean_or_none(np.abs(shared_margins)), "shared_edge_obb_sat_gap": _mean_or_none(np.maximum(shared_margins, 0)), "obb_overlap_pair_fraction": _mean_or_none(np.asarray(list(margins.values())) < -cfg.zero_tolerance), "part_volume_sum": float(sum(np.prod(p["bbox_size"]) for p in parts))}
    return {"metrics": metrics, "matches": matched, "gt_edge_count": len(gt["edges"]), "shared_edge_count": len(spec["connections"])}


def run_partnet_evaluation(inputs: dict, shared: dict, generation: dict, gt_cache: dict, cfg: PartNetConfig) -> dict:
    records, failures = [], []
    for representation in REPRESENTATIONS:
        for item in inputs["objects"]:
            oid = item["object_id"]
            generated = generation[representation].get(oid, {})
            if generated.get("status") != "valid" or shared[oid]["status"] != "valid":
                continue
            fingerprint = content_hash({"evaluation": EVALUATION_VERSION, "generation": generated["fingerprint"], "gt": gt_cache[oid]["fingerprint"], "threshold": cfg.near_contact_threshold})
            path = cfg.output_path(f"evaluation/{representation}/{oid}.json")
            try:
                cached = _read_cache(path, fingerprint)
                if cached is None:
                    spec = shared[oid]["spec"]
                    layout = transform_partnet_representation(generated["result"], spec, representation)
                    evaluation = evaluate_partnet_layout(layout, spec, gt_cache[oid], cfg)
                    cached = {"fingerprint": fingerprint, "object_id": oid, "category": item["category"], "method": representation, "normalization": layout["normalization"], "layout_parts": layout["parts"], **evaluation}
                    save_json(cached, path)
                records.append(cached)
            except (ValueError, KeyError, TypeError, FloatingPointError) as error:
                failures.append({"object_id": oid, "method": representation, "error": _safe_error(error)})
    complete = set(gt_cache)
    for method in REPRESENTATIONS:
        complete &= {r["object_id"] for r in records if r["method"] == method}
    ordered_complete = [r["object_id"] for r in inputs["objects"] if r["object_id"] in complete]
    result = {"version": EVALUATION_VERSION, "objects": records, "complete_ids": ordered_complete, "failures": failures, "metric_definitions": METRIC_DEFINITIONS}
    save_json(result, cfg.output_path("evaluation_all.json"))
    print(f"静态评估: {len(records)} method-object records；三方法共同有效样本 {len(complete)}/{len(inputs['objects'])}；evaluation failures={len(failures)}")
    return result


# =============================================================================
# 6. Mechanism ablation and all-edge part-scale perturbations (NO API)
# =============================================================================
def randomize_partnet_anchors(raw: dict, spec: dict, seed: int) -> dict:
    result = copy.deepcopy(raw)
    rng = np.random.default_rng(seed)
    for p in result["parts"]:
        if p["part_id"] == spec["root_part_id"]:
            continue
        for field_name in ("parent_anchor", "child_anchor"):
            anchor = rng.uniform(-0.5, 0.5, 3)
            anchor[int(rng.integers(0, 3))] = 0.5 if rng.random() >= 0.5 else -0.5
            p[field_name] = anchor.tolist()
    return result


def run_partnet_random_anchor(inputs: dict, shared: dict, generation: dict, evaluation: dict, gt_cache: dict, cfg: PartNetConfig) -> dict:
    records = []
    bases = {r["object_id"]: r for r in evaluation["objects"] if r["method"] == "anchor"}
    for item in inputs["objects"]:
        oid = item["object_id"]
        if oid not in bases:
            continue
        base = bases[oid]
        fingerprint = content_hash({"base": base["fingerprint"], "random_anchor": "uniform-face-v1", "seeds": cfg.random_anchor_seeds, "normalization": "fixed_to_base", "matching": "fixed_to_base"})
        path = cfg.output_path(f"random_anchor/{oid}.json")
        cached = _read_cache(path, fingerprint)
        if cached is None:
            spec = shared[oid]["spec"]
            raw = generation["anchor"][oid]["result"]
            seed_results = []
            for seed in cfg.random_anchor_seeds:
                modified = randomize_partnet_anchors(raw, spec, seed)
                layout = transform_partnet_representation(modified, spec, "anchor", base["normalization"])
                result = evaluate_partnet_layout(layout, spec, gt_cache[oid], cfg, frozen_matches=base["matches"])
                seed_results.append({"seed": seed, "metrics": result["metrics"]})
            if not seed_results:
                raise ValueError("random_anchor_seeds must not be empty")
            metrics = {key: _mean_or_none([r["metrics"][key] for r in seed_results if r["metrics"][key] is not None]) for key in METRIC_DEFINITIONS}
            cached = {"fingerprint": fingerprint, "object_id": oid, "category": item["category"], "method": "random_anchor", "metrics": metrics, "seed_results": seed_results, "normalization_policy": "base_anchor_fixed", "matching_policy": "base_anchor_fixed"}
            save_json(cached, path)
        records.append(cached)
    result = {"objects": records, "seeds": list(cfg.random_anchor_seeds), "aggregation": "seed means within object; object-level paired statistics"}
    save_json(result, cfg.output_path("random_anchor_all.json"))
    print(f"Random Anchor: {len(records)} objects × {len(cfg.random_anchor_seeds)} seeds；未调用 LLM。")
    return result


def run_partnet_scale_perturbation(inputs: dict, shared: dict, generation: dict, evaluation: dict, cfg: PartNetConfig) -> dict:
    records = []
    bases = {(r["object_id"], r["method"]): r for r in evaluation["objects"]}
    for method in REPRESENTATIONS:
        for item in inputs["objects"]:
            oid = item["object_id"]
            if (oid, method) not in bases:
                continue
            base = bases[(oid, method)]
            fingerprint = content_hash({"base": base["fingerprint"], "perturbation": "all-shared-edges-scale-v1", "factors": cfg.scale_factors, "normalization": "fixed_to_base"})
            path = cfg.output_path(f"perturbation/{method}/{oid}.json")
            cached = _read_cache(path, fingerprint)
            if cached is None:
                spec = shared[oid]["spec"]
                raw = generation[method][oid]["result"]
                base_index = {p["part_id"]: p for p in base["layout_parts"]}
                conditions = []
                for factor in cfg.scale_factors:
                    if factor <= 0:
                        raise ValueError("scale_factors must be positive")
                    edge_records = []
                    for edge in spec["connections"]:
                        parent, child = edge["parent_part_id"], edge["child_part_id"]
                        modified = copy.deepcopy(raw)
                        target = next(p for p in modified["parts"] if p["part_id"] == child)
                        target["bbox_size"] = (np.asarray(target["bbox_size"]) * factor).tolist()
                        layout = transform_partnet_representation(modified, spec, method, base["normalization"])
                        index = {p["part_id"]: p for p in layout["parts"]}
                        before = obb_sat_margin(base_index[parent], base_index[child])
                        after = obb_sat_margin(index[parent], index[child])
                        edge_records.append({"parent_part_id": parent, "child_part_id": child, "before_margin": before, "after_margin": after, "before_deviation": abs(before), "after_deviation": abs(after), "margin_drift": abs(after-before), "deviation_change": abs(after)-abs(before)})
                    metrics = {"contact_deviation": _mean_or_none([e["after_deviation"] for e in edge_records]), "baseline_contact_deviation": _mean_or_none([e["before_deviation"] for e in edge_records]), "margin_drift": _mean_or_none([e["margin_drift"] for e in edge_records]), "deviation_change": _mean_or_none([e["deviation_change"] for e in edge_records])}
                    conditions.append({"scale": float(factor), "metrics": metrics, "edges": edge_records})
                cached = {"fingerprint": fingerprint, "object_id": oid, "category": item["category"], "method": method, "conditions": conditions, "normalization_policy": "base_fixed", "selection_policy": "every_shared_edge_once; only that child bbox scaled per intervention"}
                save_json(cached, path)
            for condition in cached["conditions"]:
                records.append({"object_id": oid, "category": item["category"], "method": method, **condition})
    result = {"objects": records, "aggregation": "edge interventions averaged within object; objects are statistical units", "metric_note": "OBB SAT-based gap/penetration deviation in the PRE-edit normalized frame. Not exact mesh distance."}
    save_json(result, cfg.output_path("perturbation_all.json"))
    print(f"All-edge scale perturbation: {len(records)} method-object-scale records；未调用 LLM。")
    return result


# =============================================================================
# 7. Paired object-level statistics, failure accounting, CSV outputs
# =============================================================================
def _write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summaries(records: list, group_fields: tuple, metric_names: tuple) -> list:
    grouped = defaultdict(list)
    for row in records:
        grouped[tuple(row.get(k) for k in group_fields)].append(row)
    output = []
    for group_key, rows in grouped.items():
        identity = dict(zip(group_fields, group_key))
        for metric in metric_names:
            valid = [r for r in rows if r["metrics"].get(metric) is not None and np.isfinite(r["metrics"][metric])]
            values = np.asarray([r["metrics"][metric] for r in valid], dtype=float)
            by_category = defaultdict(list)
            for r in valid:
                by_category[r["category"]].append(r["metrics"][metric])
            output.append({**identity, "metric": metric, "n": len(values), "mean": float(values.mean()) if len(values) else None, "std": float(values.std(ddof=1)) if len(values)>1 else None, "category_macro_mean": _mean_or_none([np.mean(v) for v in by_category.values()]), "categories_with_data": len(by_category)})
    return output


def paired_object_statistics(anchor: dict, baseline: dict, metric: str, cfg: PartNetConfig, higher_is_better: bool = False) -> dict:
    """Paired object statistics; the SAME numerical-zero rule is used throughout."""
    tolerance = 1e-9
    ids = sorted(set(anchor) & set(baseline))
    ids = [oid for oid in ids if anchor[oid]["metrics"].get(metric) is not None and baseline[oid]["metrics"].get(metric) is not None]
    if not ids:
        return {"n": 0, "mean_improvement": None, "ci_low": None, "ci_high": None, "wilcoxon_p": None, "wins": 0, "ties": 0, "losses": 0, "tie_tolerance": tolerance}
    av = np.asarray([anchor[oid]["metrics"][metric] for oid in ids], dtype=float)
    bv = np.asarray([baseline[oid]["metrics"][metric] for oid in ids], dtype=float)
    if not np.isfinite(av).all() or not np.isfinite(bv).all():
        raise ValueError(f"Nonfinite statistics input: {metric}")
    difference = av - bv if higher_is_better else bv - av
    difference[np.abs(difference) <= tolerance] = 0.0
    strata = defaultdict(list)
    for i, oid in enumerate(ids):
        if anchor[oid]["category"] != baseline[oid]["category"]:
            raise ValueError(f"Paired category mismatch: {oid}")
        strata[anchor[oid]["category"]].append(i)
    count = int(cfg.bootstrap_samples)
    if count < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    rng = np.random.default_rng(cfg.statistics_seed)
    bootstrap = np.zeros(count)
    for indices in strata.values():
        values = difference[indices]
        for start in range(0, count, 256):
            stop = min(start + 256, count)
            draw = rng.integers(0, len(values), size=(stop - start, len(values)))
            bootstrap[start:stop] += values[draw].sum(axis=1) / len(ids)
    interval = np.quantile(bootstrap, [0.025, 0.975]) if count else [None, None]
    if len(difference) < 2:
        p_value = None
    elif not np.any(difference):
        p_value = 1.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p_value = float(wilcoxon(difference, alternative="two-sided", zero_method="wilcox").pvalue)
    return {"n": len(ids), "anchor_mean": float(av.mean()), "baseline_mean": float(bv.mean()), "mean_improvement": float(difference.mean()), "median_improvement": float(np.median(difference)), "ci_low": float(interval[0]) if count else None, "ci_high": float(interval[1]) if count else None, "wilcoxon_p": p_value, "wins": int(np.sum(difference > 0)), "ties": int(np.sum(difference == 0)), "losses": int(np.sum(difference < 0)), "tie_tolerance": tolerance}


def _holm_adjust(rows: list) -> None:
    usable = [r for r in rows if r.get("wilcoxon_p") is not None]
    usable.sort(key=lambda r: r["wilcoxon_p"])
    running = 0.0
    for i, row in enumerate(usable):
        running = max(running, min(1.0, (len(usable)-i)*row["wilcoxon_p"]))
        row["holm_p"] = running
    for row in rows:
        row.setdefault("holm_p", None)


def export_partnet_results(inputs: dict, shared: dict, generation: dict, evaluation: dict, random_anchor: dict, perturbation: dict, cfg: PartNetConfig) -> dict:
    complete = set(evaluation["complete_ids"])
    paired_static = [r for r in evaluation["objects"] if r["object_id"] in complete]
    paired_static += [r for r in random_anchor["objects"] if r["object_id"] in complete]
    all_static = evaluation["objects"] + random_anchor["objects"]
    summary = _summaries(paired_static, ("method",), tuple(METRIC_DEFINITIONS))
    category_summary = _summaries(paired_static, ("category", "method"), tuple(METRIC_DEFINITIONS))
    perturb_rows = [r for r in perturbation["objects"] if r["object_id"] in complete]
    perturb_metrics = ("contact_deviation", "baseline_contact_deviation", "margin_drift", "deviation_change")
    perturb_summary = _summaries(perturb_rows, ("method", "scale"), perturb_metrics)
    stats = []
    lookup = {method: {r["object_id"]: r for r in paired_static if r["method"]==method} for method in (*REPRESENTATIONS, "random_anchor")}
    for baseline in ("global", "local", "random_anchor"):
        for metric in PRIMARY_METRICS:
            row = paired_object_statistics(lookup["anchor"], lookup[baseline], metric, cfg, metric in HIGHER_IS_BETTER)
            stats.append({"experiment": "static" if baseline != "random_anchor" else "random_anchor", "baseline": baseline, "metric": metric, "scale": None, **row})
    _holm_adjust(stats)
    perturb_stats = []
    for scale in cfg.scale_factors:
        by_method = {m: {r["object_id"]: r for r in perturb_rows if r["method"]==m and r["scale"]==scale} for m in REPRESENTATIONS}
        for baseline in ("global", "local"):
            for metric in ("contact_deviation", "margin_drift", "deviation_change"):
                result = paired_object_statistics(by_method["anchor"], by_method[baseline], metric, cfg)
                perturb_stats.append({"experiment": "scale_perturbation", "baseline": baseline, "metric": metric, "scale": scale, **result})
    _holm_adjust(perturb_stats)
    statuses = []
    for item in inputs["objects"]:
        oid = item["object_id"]
        status = {"object_id": oid, "category": item["category"], "shared_status": shared[oid]["status"], "shared_attempts": shared[oid]["attempt_count"], "complete_case": oid in complete}
        for method in REPRESENTATIONS:
            row = generation[method].get(oid, {})
            status[method+"_status"] = row.get("status", "blocked_by_shared")
            status[method+"_attempts"] = row.get("attempt_count", 0)
            status[method+"_first_attempt_valid"] = bool(row.get("attempts") and row["attempts"][0].get("validation", {}).get("valid"))
        statuses.append(status)
    flat = [{"object_id": r["object_id"], "category": r["category"], "method": r["method"], "complete_case": r["object_id"] in complete, **r["metrics"]} for r in all_static]
    tables = {"summary": summary, "category_summary": category_summary, "perturbation_summary": perturb_summary, "paired_statistics": stats+perturb_stats, "generation_status": statuses, "object_metrics": flat, "summary_all_available": _summaries(all_static, ("method",), tuple(METRIC_DEFINITIONS))}
    for name, rows in tables.items():
        _write_csv(rows, cfg.output_path(f"tables/{name}.csv"))
    methodology = {"metric_definitions": METRIC_DEFINITIONS, "main_sample_policy": "complete cases shared across anchor/global/local; failures and all-available results reported separately", "partnet_setting": "GT semantic part inventory conditioned; not free part discovery", "gt_pose_caveat": "Descriptions do not uniquely determine a GT instance. Pose errors measure agreement with one reference instance, not uniquely correct text-to-3D output.", "semantic_matching": "strict semantic groups; branch-specific Hungarian base alignment; fixed base correspondence for Random Anchor", "random_anchor": "same raw base parameters except anchors; fixed base normalization; seed averaging per object", "perturbation": "all shared edges, one child resize per intervention; fixed pre-edit normalization; average edges within object", "statistics": "paired object differences, category-stratified bootstrap 95% CI, two-sided Wilcoxon, Holm correction within static/ablation family and within perturbation family", "limitations": ["Existing adjacency reference is reused, not recomputed or re-labelled as directed GT.", "OBB SAT contact quantities are bounding-box proxies, not GT mesh attachment locations.", "Rotation GT, contact-region GT, Mobility articulation and language editing are not implemented in this PartNet stage."], "complete_object_count": len(complete), "planned_object_count": len(inputs["objects"])}
    response_models = {method: dict(Counter(a.get("returned_model", "unknown") for r in generation[method].values() for a in r.get("attempts", []))) for method in REPRESENTATIONS}
    save_json({"requested": cfg.request_config(), "returned_model_counts": response_models}, cfg.output_path("llm_runtime_metadata.json"))
    save_json(methodology, cfg.output_path("evaluation_protocol.json"))
    save_json(tables, cfg.output_path("tables/results.json"))
    print(f"已导出 tables/*.csv；主表共同有效样本 {len(complete)}/{len(inputs['objects'])}。失败样本见 generation_status.csv。")
    return tables


def partnet_summary_view(tables: dict, kind: str = "summary"):
    import pandas as pd
    frame = pd.DataFrame(tables[kind])
    if frame.empty:
        return frame
    if kind == "summary":
        return frame.pivot(index="method", columns="metric", values="mean").round(6)
    if kind == "perturbation_summary":
        return frame.pivot(index=["scale", "method"], columns="metric", values="mean").round(6)
    return frame


# =============================================================================
# 8. Qualitative layout export
# =============================================================================
def visualize_partnet_comparison(object_id: str, evaluation: dict, gt_cache: dict, cfg: PartNetConfig) -> list:
    """One figure per layout; not a mesh reconstruction."""
    oid = str(object_id)
    available = [r for r in evaluation["objects"] if r["object_id"] == oid]
    if oid not in gt_cache:
        raise KeyError(oid)
    layouts = [("gt_aabb", gt_cache[oid]["parts"])] + [(r["method"], r["layout_parts"]) for r in available]
    files = []
    box_edges = [(i,j) for i in range(8) for j in range(i+1,8) if np.sum(_BOX_SIGNS[i] != _BOX_SIGNS[j]) == 1]
    for name, parts in layouts:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        for part in parts:
            corners = partnet_box_corners(part)
            xs, ys, zs = [], [], []
            for i,j in box_edges:
                xs.extend([corners[i,0],corners[j,0],np.nan])
                ys.extend([corners[i,1],corners[j,1],np.nan])
                zs.extend([corners[i,2],corners[j,2],np.nan])
            ax.plot(xs,ys,zs)
        ax.set(xlim=(-0.55,0.55), ylim=(-0.55,0.55), zlim=(-0.55,0.55), xlabel="x (right)", ylabel="y (front)", zlabel="z (up)", title=f"{oid} | {name}")
        ax.set_box_aspect((1,1,1))
        fig.tight_layout()
        path = cfg.output_path(f"figures/{oid}_{name}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.show()
        plt.close(fig)
        files.append(str(path))
    return files


# =============================================================================
# 9. Legacy open-world compatibility (NOT used for PartNet results above)
# The original metric names/numerical definitions are retained so previously
# saved experiments1 results are not silently redefined. These include legacy
# AABB-like metrics and the old offset transform; do not call them for PartNet.
# =============================================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'olive', 'gray']

system_prompt = '\nYou convert a natural-language description of a 3D object into a hierarchical anchor-based part tree.\n\nReturn ONLY valid JSON.\nDo not output Markdown.\nDo not output explanations.\n\nCoordinate system:\n+x = right\n-x = left\n+y = front\n-y = back\n+z = up\n-z = down\n\nAll coordinates are defined in the local coordinate frame of each parent part.\n\nThe root node represents the whole object.\n\nRoot node format:\n\nname\nsemantic\nbbox_size\nhas_geometry\nchildren\n\nRoot rules:\n\nhas_geometry=false\n\nDo not include:\nparent_anchor\nchild_anchor\noffset\nrotation\n\nEvery non-root node must contain:\n\nname\nsemantic\nbbox_size\nhas_geometry\nparent_anchor\nchild_anchor\noffset\nrotation\nchildren\n\nbbox_size format:\n\n[x_length,y_length,z_length]\n\nAll dimensions are normalized.\n\nThe longest root dimension should approximately equal 1.0.\n\nThe hierarchy must represent real physical attachment.\n\nA child should be attached to the part that physically supports it.\n\nAvoid attaching all parts directly to the root.\n\nPrefer simple meaningful hierarchy.\n\nparent_anchor and child_anchor are normalized bbox coordinates.\n\nEach value usually lies between -0.5 and 0.5.\n\nExamples:\n\ncenter=[0,0,0]\n\nleft=[-0.5,0,0]\n\nright=[0.5,0,0]\n\nfront=[0,0.5,0]\n\nback=[0,-0.5,0]\n\ntop=[0,0,0.5]\n\nbottom=[0,0,-0.5]\n\nparent_anchor describes the attachment location on the parent.\n\nchild_anchor describes the attachment location on the child.\n\noffset is a small residual displacement.\n\nPrefer:\n\noffset=[0,0,0]\n\nrotation is quaternion:\n\n[w,x,y,z]\n\nNo rotation:\n\n[1,0,0,0]\n\nrotation describes child orientation relative to direct parent.\n\nThe downstream program computes:\n\nchild_center =\nparent_anchor_world\n+\noffset_world\n-\nchild_rotation_world @ (child_bbox_size * child_anchor)\n\nTherefore choose anchors according to this equation.\n\nDo not use large offsets.\n\nDo not create unnecessary tiny parts.\n\nDo not invent screws, bolts, or decorations.\n\nRepeated structural parts should have similar dimensions.\n\nMaintain consistent:\nleft/right\nfront/back\ntop/bottom\n\nBefore output:\nverify that the generated structure forms one physically plausible object.\n\nOutput only JSON.\n'

global_obb_system_prompt = '\nYou convert a natural-language description of a 3D object into independent global oriented bounding boxes.\n\nReturn ONLY valid JSON.\n\nDo not output Markdown.\nDo not output explanations.\n\nCoordinate system:\n\n+x right\n-x left\n+y front\n-y back\n+z up\n-z down\n\nAll parts are represented in GLOBAL coordinates.\n\nOutput format:\n\n{\n"name":"",\n"semantic":"",\n"bbox_size":[x,y,z],\n"parts":[]\n}\n\nEach part contains:\n\nname\nsemantic\nbbox_size\ncenter\nrotation\n\nbbox_size:\n\n[x_length,y_length,z_length]\n\ncenter:\n\n[x,y,z]\n\nrotation:\n\n[w,x,y,z]\n\nrotation is absolute world rotation.\n\nDo not use:\nparent_anchor\nchild_anchor\noffset\n\nEvery part independently defines its final world position.\n\nConnected parts should physically touch.\n\nAvoid floating parts.\n\nAvoid unreasonable overlap.\n\nMaintain correct spatial relationships.\n\nDo not add unnecessary small parts.\n\nOutput only JSON.\n'

hierarchy_local_system_prompt = '\nYou are generating a hierarchical 3D object representation.\n\nGiven a natural language object description, decompose the object into semantic parts.\n\nOutput a JSON hierarchy.\n\nEach node must contain:\n\n{\n"name":"",\n"bbox_size":[x,y,z],\n"relative_position":[x,y,z],\n"children":[]\n}\n\nRules:\n\n1. The root object must have children representing meaningful parts.\n\n2. Every child part must define relative_position with respect to its parent part.\n\n3. relative_position represents the child center in the parent local coordinate system.\n\n4. Do not use [0,0,0] unless the child is actually located at the parent center.\n\n5. Preserve realistic object structure.\n\n6. Parts that are physically connected should have reasonable relative positions.\n\n7. Do not output global coordinates.\n\n8. Do not use parent_anchor or child_anchor.\n\n9. Output only valid JSON.\nUse exactly the parts given in the shared part specification.\nDo not add parts.\nDo not remove parts.\nDo not rename parts.\nDo not merge or split parts.\nEvery provided part must appear exactly once.\nUse the provided connections to construct the hierarchy.\nKeep all part names exactly unchanged.\n'

part_decomposition_system_prompt = '\nYou decompose a 3D object description into major geometric parts.\n\nReturn ONLY valid JSON.\n\nDo not output Markdown.\n\nDo not output explanations.\n\nDo not predict:\npositions\nrotations\nbounding boxes\n\nOnly identify geometric parts and physical connections.\n\nReturn:\n\n{\n"name":"",\n"semantic":"",\n"parts":[\n{\n"name":"",\n"semantic":""\n}\n],\n"connections":[\n[\n"parent_part",\n"child_part"\n]\n]\n}\n\nRules:\n\nOnly include meaningful visible geometric components.\n\nDo not include screws, bolts, tiny details.\n\nRepeated physical components should be separated.\n\nconnections represent direct physical support relationships.\n\nExample:\n\nchair:\n\nparts:\nseat\nlegs\nbackrest\n\nconnections:\n\n[\n["seat","front_left_leg"],\n["seat","front_right_leg"],\n["seat","back_left_leg"],\n["seat","back_right_leg"],\n["seat","backrest"]\n]\n\nDo not create unnecessary connections.\n'

def generate_tree_json(object_description, system_message):
    client = create_deepseek_client()
    response = client.chat.completions.create(model=os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'), messages=[{'role': 'system', 'content': system_message}, {'role': 'user', 'content': object_description}], stream=False, temperature=0.0)
    text = response.choices[0].message.content
    text = text.replace('```json', '')
    text = text.replace('```', '')
    return text.strip()

def generate_part_decomposition(object_description):
    text = generate_tree_json(object_description, part_decomposition_system_prompt)
    return (text, json.loads(text))

def infer_semantic_connections(part_spec):
    if 'connections' in part_spec:
        return [tuple(x) for x in part_spec['connections']]
    names = [p['name'] for p in part_spec['parts']]
    connections = []
    if len(names) > 1:
        root = names[0]
        for n in names[1:]:
            connections.append((root, n))
    return connections

def build_shared_structure_input(object_description, part_spec):
    part_spec_text = json.dumps(part_spec, ensure_ascii=False)
    text = 'Object description:\n' + object_description + '\n\nShared part specification:\n' + part_spec_text + '\n\nUse exactly the parts and part names provided in the shared part specification. Do not add, remove, rename, merge, or split parts. Use the provided connections as the structural relationships.'
    return text

def generate_anchor_tree(object_description, part_spec):
    generation_input = build_shared_structure_input(object_description, part_spec)
    text = generate_tree_json(generation_input, system_prompt)
    return (text, json.loads(text))

def generate_global_obb(object_description, part_spec):
    generation_input = build_shared_structure_input(object_description, part_spec)
    text = generate_tree_json(generation_input, global_obb_system_prompt)
    return (text, json.loads(text))

def generate_hierarchy_local(object_description, part_spec):
    generation_input = build_shared_structure_input(object_description, part_spec)
    client = create_deepseek_client()
    response = client.chat.completions.create(model=os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'), messages=[{'role': 'system', 'content': hierarchy_local_system_prompt}, {'role': 'user', 'content': generation_input}], temperature=0.0, max_tokens=6000, response_format={'type': 'json_object'})
    text = response.choices[0].message.content
    tree = json.loads(text)
    prepare_hierarchy_local_tree(tree)
    return (text, tree)

def compute_anchor_tree_world_transform(node, parent_center=None, parent_rotation=None, parent_size=None):
    if parent_center is None:
        parent_center = np.array([0.0, 0.0, 0.0])
    if parent_rotation is None:
        parent_rotation = np.eye(3)
    if parent_size is None:
        parent_size = np.array(node['bbox_size'])
    if 'parent_anchor' not in node:
        center = parent_center
        rotation = parent_rotation
    else:
        pa = np.array(node['parent_anchor'])
        ca = np.array(node['child_anchor'])
        offset = np.array(node.get('offset', [0, 0, 0]))
        local_rotation = quaternion_to_matrix(node['rotation'])
        rotation = parent_rotation @ local_rotation
        parent_point = parent_center + parent_rotation @ (parent_size * pa)
        child_offset = rotation @ (np.array(node['bbox_size']) * ca)
        center = parent_point + offset - child_offset
    node['center'] = center.tolist()
    node['world_rotation'] = rotation.tolist()
    for child in node.get('children', []):
        compute_anchor_tree_world_transform(child, center, rotation, np.array(node['bbox_size']))
    return node

def compute_hierarchy_local_world_transform(node, parent_center=None, parent_rotation=None):
    if parent_center is None:
        parent_center = np.array([0.0, 0.0, 0.0])
    if parent_rotation is None:
        parent_rotation = np.eye(3)
    node['center'] = parent_center.tolist()
    node['world_rotation'] = parent_rotation.tolist()
    for child in node.get('children', []):
        relative = np.array(child.get('relative_position', [0, 0, 0]))
        child_center = parent_center + parent_rotation @ relative
        child_rotation = parent_rotation @ quaternion_to_matrix(child.get('relative_rotation', [1, 0, 0, 0]))
        child['center'] = child_center.tolist()
        child['world_rotation'] = child_rotation.tolist()
        compute_hierarchy_local_world_transform(child, child_center, child_rotation)
    return node

def remove_anchor_information(node):
    node.pop('parent_anchor', None)
    node.pop('child_anchor', None)
    node.pop('offset', None)
    node.pop('rotation', None)
    for child in node.get('children', []):
        remove_anchor_information(child)
    return node

def clear_world_transform(node):
    node.pop('center', None)
    node.pop('world_rotation', None)
    for child in node.get('children', []):
        clear_world_transform(child)
    return node

def generate_anchor_tree_with_parts(object_description, part_spec):
    text, tree = generate_anchor_tree(object_description, part_spec)
    compute_anchor_tree_world_transform(tree)
    parts = collect_geometry_parts_for_eval(tree)
    return (text, tree, parts, part_spec)

def add_global_transform(node):
    if 'parts' in node and 'children' not in node:
        node['children'] = node.pop('parts')
    if 'center' not in node:
        node['center'] = [0, 0, 0]
    if 'world_rotation' not in node:
        if 'rotation' in node:
            node['world_rotation'] = quaternion_to_matrix(node.get('rotation', [1, 0, 0, 0])).tolist()
        else:
            node['world_rotation'] = np.eye(3).tolist()
    for child in node.get('children', []):
        child['has_geometry'] = True
        if 'children' not in child:
            child['children'] = []
        add_global_transform(child)
    return node

def generate_global_obb_with_parts(object_description, part_spec):
    text, tree = generate_global_obb(object_description, part_spec)
    tree['children'] = tree.get('parts', [])
    tree.pop('parts', None)
    tree['has_geometry'] = False
    for child in tree['children']:
        child['has_geometry'] = True
        child['children'] = []
    add_global_transform(tree)
    normalize_tree_scale(tree)
    parts = collect_geometry_parts_for_eval(tree)
    return (text, tree, parts, part_spec)

def prepare_hierarchy_local_tree(tree):

    def traverse(node, is_root=False):
        if 'children' not in node:
            node['children'] = node.get('parts', [])
        node.pop('parts', None)
        node['has_geometry'] = not is_root
        for child in node['children']:
            traverse(child, False)
    traverse(tree, True)
    return tree

def collect_geometry_parts_for_eval(node, parts=None, is_root=True):
    if parts is None:
        parts = []
    children = node.get('children', node.get('parts', []))
    has_geometry = node.get('has_geometry', not is_root)
    if has_geometry and 'bbox_size' in node:
        parts.append(node)
    for child in children:
        collect_geometry_parts_for_eval(child, parts, False)
    return parts

def draw_bbox(ax, center, size, rotation, color):
    center = np.array(center)
    size = np.array(size)
    r = np.array(rotation)
    corners = []
    for x in [-0.5, 0.5]:
        for y in [-0.5, 0.5]:
            for z in [-0.5, 0.5]:
                local = np.array([x, y, z]) * size
                corners.append(center + r @ local)
    edges = [(0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7), (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7)]
    for e in edges:
        p1 = corners[e[0]]
        p2 = corners[e[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)

def draw_tree(ax, node, index=0):
    if node.get('has_geometry', False):
        color = colors[index % len(colors)]
        draw_bbox(ax, node['center'], node['bbox_size'], node['world_rotation'], color)
    for i, child in enumerate(node.get('children', [])):
        draw_tree(ax, child, index + i + 1)

def draw_layout(tree):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_tree(ax, tree)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    plt.show()

def draw_layout_comparison(anchor_tree, global_tree):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_tree(ax, anchor_tree, 0)
    draw_tree(ax, global_tree, 5)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    plt.show()

def compute_box_intersection_volume(a, b):
    ca = np.array(a['center'])
    cb = np.array(b['center'])
    sa = np.array(a['bbox_size'])
    sb = np.array(b['bbox_size'])
    min_a = ca - sa / 2
    max_a = ca + sa / 2
    min_b = cb - sb / 2
    max_b = cb + sb / 2
    overlap = np.maximum(np.minimum(max_a, max_b) - np.maximum(min_a, min_b), 0)
    return float(np.prod(overlap))

def metric_part_volume_sum(tree):
    parts = collect_geometry_parts_for_eval(tree)
    total_volume = 0.0
    for p in parts:
        size = np.array(p['bbox_size'])
        total_volume += np.prod(size)
    return float(total_volume)

def metric_bbox_penetration_ratio(tree):
    parts = collect_geometry_parts_for_eval(tree)
    ratios = []
    for a, b in itertools.combinations(parts, 2):
        inter = compute_box_intersection_volume(a, b)
        if inter > 0:
            va = np.prod(np.array(a['bbox_size']))
            vb = np.prod(np.array(b['bbox_size']))
            ratios.append(inter / min(va, vb))
    if len(ratios) == 0:
        return 0.0
    return float(np.mean(ratios))

def bbox_surface_distance(a, b):
    ca = np.array(a['center'])
    cb = np.array(b['center'])
    sa = np.array(a['bbox_size'])
    sb = np.array(b['bbox_size'])
    gap = np.abs(ca - cb) - (sa + sb) / 2
    gap = np.maximum(gap, 0)
    return float(np.linalg.norm(gap))

def metric_parent_child_contact_error(tree, connections=None):
    errors = []
    if connections is not None:
        for parent_name, child_name in connections:
            parent = find_part_by_name(tree, parent_name)
            child = find_part_by_name(tree, child_name)
            if parent is not None and child is not None:
                errors.append(bbox_surface_distance(parent, child))
    else:

        def traverse(parent):
            for child in parent.get('children', []):
                if child.get('has_geometry', False):
                    errors.append(bbox_surface_distance(parent, child))
                traverse(child)
        traverse(tree)
    if len(errors) == 0:
        return None
    return float(np.mean(errors))

def metric_hierarchy_edge_success_rate(tree, connections, threshold=0.05):
    scores = []
    for parent_name, child_name in connections:
        parent = find_part_by_name(tree, parent_name)
        child = find_part_by_name(tree, child_name)
        if parent is None or child is None:
            continue
        distance = bbox_surface_distance(parent, child)
        size = min(np.linalg.norm(np.array(parent['bbox_size'])), np.linalg.norm(np.array(child['bbox_size'])))
        if size <= 0:
            continue
        scores.append(float(distance / size < threshold))
    if len(scores) == 0:
        return None
    return float(np.mean(scores))

def metric_relative_attachment_consistency(tree, connections):
    errors = []
    for parent_name, child_name in connections:
        parent = find_part_by_name(tree, parent_name)
        child = find_part_by_name(tree, child_name)
        if parent is None or child is None:
            continue
        parent_center = np.array(parent['center'])
        child_center = np.array(child['center'])
        parent_rotation = np.array(parent['world_rotation'])
        parent_size = np.array(parent['bbox_size'])
        child_size = np.array(child['bbox_size'])
        relative = parent_rotation.T @ (child_center - parent_center)
        normalized = relative / (parent_size + 1e-08)
        if 'parent_anchor' in child:
            anchor = np.array(child['parent_anchor'])
            error = np.linalg.norm(normalized - anchor)
        else:
            error = np.linalg.norm(normalized)
        errors.append(error)
    if len(errors) == 0:
        return None
    return float(np.mean(errors))

def metric_hierarchy_attachment_error(tree, connections):
    errors = []
    for parent_name, child_name in connections:
        parent = find_part_by_name(tree, parent_name)
        child = find_part_by_name(tree, child_name)
        if parent is None or child is None:
            continue
        pa = np.array(parent['center'])
        pb = np.array(child['center'])
        sa = np.array(parent['bbox_size'])
        sb = np.array(child['bbox_size'])
        distance = bbox_surface_distance(parent, child)
        scale = min(np.linalg.norm(sa), np.linalg.norm(sb))
        if scale > 0:
            errors.append(distance / scale)
    if len(errors) == 0:
        return None
    return float(np.mean(errors))

def find_part_by_name(tree, name):
    parts = collect_geometry_parts_for_eval(tree)
    target = name.lower().replace('_', '')
    for p in parts:
        current = p['name'].lower().replace('_', '')
        if current == target:
            return p
    for p in parts:
        current = p['name'].lower().replace('_', '')
        if target in current or current in target:
            return p
    return None

def metric_anchor_reconstruction_error(tree):
    errors = []

    def traverse(parent):
        for child in parent.get('children', []):
            if 'parent_anchor' in child:
                pc = np.array(parent['center'])
                ps = np.array(parent['bbox_size'])
                pr = np.array(parent['world_rotation'])
                cc = np.array(child['center'])
                cs = np.array(child['bbox_size'])
                cr = np.array(child['world_rotation'])
                pa = np.array(child['parent_anchor'])
                ca = np.array(child['child_anchor'])
                parent_point = pc + pr @ (ps * pa)
                child_point = cc + cr @ (cs * ca)
                errors.append(np.linalg.norm(parent_point - child_point))
            traverse(child)
    traverse(tree)
    if len(errors) == 0:
        return None
    return float(np.mean(errors))

def metric_spatial_relation_accuracy(tree):
    parts = collect_geometry_parts_for_eval(tree)
    scores = []
    for a, b in itertools.combinations(parts, 2):
        name = a['name'].lower()
        ca = np.array(a['center'])
        cb = np.array(b['center'])
        if 'left' in name:
            scores.append(float(ca[0] < cb[0]))
        if 'right' in name:
            scores.append(float(ca[0] > cb[0]))
        if 'front' in name:
            scores.append(float(ca[1] > cb[1]))
        if 'back' in name:
            scores.append(float(ca[1] < cb[1]))
        if 'top' in name or 'upper' in name:
            scores.append(float(ca[2] > cb[2]))
        if 'bottom' in name or 'lower' in name:
            scores.append(float(ca[2] < cb[2]))
    if len(scores) == 0:
        return None
    return float(np.mean(scores))

def evaluate_single_tree(tree, connections=None):
    result = {}
    result['part_volume_sum'] = metric_part_volume_sum(tree)
    result['bbox_penetration_ratio'] = metric_bbox_penetration_ratio(tree)
    result['parent_child_contact_error'] = metric_parent_child_contact_error(tree, connections)
    result['anchor_reconstruction_error'] = metric_anchor_reconstruction_error(tree)
    result['spatial_relation_accuracy'] = metric_spatial_relation_accuracy(tree)
    if connections is not None:
        result['hierarchy_attachment_error'] = metric_hierarchy_attachment_error(tree, connections)
        result['hierarchy_edge_success_rate'] = metric_hierarchy_edge_success_rate(tree, connections)
        result['relative_attachment_consistency'] = metric_relative_attachment_consistency(tree, connections)
    else:
        result['hierarchy_attachment_error'] = None
        result['hierarchy_edge_success_rate'] = None
        result['relative_attachment_consistency'] = None
    return result

def evaluate_dataset(trees, connection_lists=None):
    results = []
    for i, tree in enumerate(trees):
        connections = None
        if connection_lists is not None:
            connections = connection_lists[i]
        results.append(evaluate_single_tree(tree, connections))
    summary = {}
    keys = results[0].keys()
    for k in keys:
        values = []
        for r in results:
            if r[k] is not None:
                values.append(r[k])
        if len(values) > 0:
            summary[k] = {}
            summary[k]['mean'] = float(np.mean(values))
            summary[k]['std'] = float(np.std(values))
    return summary

def paired_statistical_test(anchor_results, baseline_results, metric, higher_is_better=False, n_bootstrap=10000, seed=42):
    anchor_values = []
    baseline_values = []
    for anchor_result, baseline_result in zip(anchor_results, baseline_results):
        anchor_value = anchor_result.get(metric)
        baseline_value = baseline_result.get(metric)
        if anchor_value is not None and baseline_value is not None:
            anchor_values.append(anchor_value)
            baseline_values.append(baseline_value)
    anchor_values = np.array(anchor_values, dtype=float)
    baseline_values = np.array(baseline_values, dtype=float)
    if higher_is_better:
        improvements = anchor_values - baseline_values
    else:
        improvements = baseline_values - anchor_values
    rng = np.random.default_rng(seed)
    bootstrap_means = []
    for i in range(n_bootstrap):
        indices = rng.integers(0, len(improvements), len(improvements))
        bootstrap_means.append(np.mean(improvements[indices]))
    bootstrap_means = np.array(bootstrap_means)
    ci_low = float(np.percentile(bootstrap_means, 2.5))
    ci_high = float(np.percentile(bootstrap_means, 97.5))
    if np.allclose(improvements, 0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(improvements).pvalue)
    result = {}
    result['n'] = len(improvements)
    result['anchor_mean'] = float(np.mean(anchor_values))
    result['baseline_mean'] = float(np.mean(baseline_values))
    result['mean_improvement'] = float(np.mean(improvements))
    result['median_improvement'] = float(np.median(improvements))
    result['ci_95'] = [ci_low, ci_high]
    result['p_value'] = p_value
    return result

def save_tree_result(tree, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(tree, f, indent=4, ensure_ascii=False)

def load_tree_result(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_evaluation_result(result, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

def load_evaluation_result(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_tree_scale(tree):
    parts = collect_geometry_parts_for_eval(tree)
    if len(parts) == 0:
        return tree
    all_points = []
    for part in parts:
        center = np.array(part['center'], dtype=float)
        size = np.array(part['bbox_size'], dtype=float)
        rotation = np.array(part['world_rotation'], dtype=float)
        for x in [-0.5, 0.5]:
            for y in [-0.5, 0.5]:
                for z in [-0.5, 0.5]:
                    local = np.array([x, y, z]) * size
                    point = center + rotation @ local
                    all_points.append(point)
    all_points = np.array(all_points)
    extent = np.max(all_points, axis=0) - np.min(all_points, axis=0)
    scale = np.max(extent)
    if scale <= 0:
        return tree

    def traverse(node):
        node['bbox_size'] = (np.array(node['bbox_size'], dtype=float) / scale).tolist()
        if 'center' in node:
            node['center'] = (np.array(node['center'], dtype=float) / scale).tolist()
        if 'relative_position' in node:
            node['relative_position'] = (np.array(node['relative_position'], dtype=float) / scale).tolist()
        if 'offset' in node:
            node['offset'] = (np.array(node['offset'], dtype=float) / scale).tolist()
        for child in node.get('children', []):
            traverse(child)
    traverse(tree)
    return tree

def compute_hierarchy_depth(tree):

    def depth(node):
        children = node.get('children', [])
        if len(children) == 0:
            return 0
        return 1 + max((depth(child) for child in children))
    return depth(tree)

def compute_hierarchy_edge_count(tree):
    count = 0
    for child in tree.get('children', []):
        count += 1
        count += compute_hierarchy_edge_count(child)
    return count

def paired_win_tie_loss(anchor_results, baseline_results, metric, higher_is_better=False, tolerance=1e-08):
    wins = 0
    ties = 0
    losses = 0
    differences = []
    for anchor_result, baseline_result in zip(anchor_results, baseline_results):
        anchor_value = anchor_result.get(metric)
        baseline_value = baseline_result.get(metric)
        if anchor_value is None or baseline_value is None:
            continue
        if higher_is_better:
            difference = anchor_value - baseline_value
        else:
            difference = baseline_value - anchor_value
        differences.append(difference)
        if difference > tolerance:
            wins += 1
        elif difference < -tolerance:
            losses += 1
        else:
            ties += 1
    result = {}
    result['n'] = len(differences)
    result['wins'] = wins
    result['ties'] = ties
    result['losses'] = losses
    result['win_rate'] = wins / len(differences)
    result['tie_rate'] = ties / len(differences)
    result['loss_rate'] = losses / len(differences)
    return result

def scale_part_bbox(tree, part_name, scale_factor):
    part = find_part_by_name(tree, part_name)
    if part is None:
        return False
    part['bbox_size'] = (np.array(part['bbox_size'], dtype=float) * scale_factor).tolist()
    return True

def metric_mean_connection_error(tree, connections):
    errors = []
    for parent_name, child_name in connections:
        parent = find_part_by_name(tree, parent_name)
        child = find_part_by_name(tree, child_name)
        if parent is None or child is None:
            continue
        errors.append(bbox_surface_distance(parent, child))
    if len(errors) == 0:
        return None
    return float(np.mean(errors))

def summarize_perturbation_method(perturbation_results, scale_factor, method):
    values = [x[method] for x in perturbation_results if x['scale'] == scale_factor and x[method] is not None]
    result = {}
    result['mean'] = float(np.mean(values))
    result['std'] = float(np.std(values))
    result['n'] = len(values)
    return result

def bbox_contact_deviation(a, b):
    ca = np.array(a['center'], dtype=float)
    cb = np.array(b['center'], dtype=float)
    sa = np.array(a['bbox_size'], dtype=float)
    sb = np.array(b['bbox_size'], dtype=float)
    signed_gap = np.abs(ca - cb) - (sa + sb) / 2
    positive_gap = np.maximum(signed_gap, 0)
    if np.any(signed_gap > 0):
        deviation = np.linalg.norm(positive_gap)
    else:
        deviation = -np.max(signed_gap)
    scale = min(np.linalg.norm(sa), np.linalg.norm(sb))
    if scale <= 0:
        return None
    return float(deviation / scale)

def metric_mean_contact_deviation(tree, connections):
    errors = []
    for parent_name, child_name in connections:
        parent = find_part_by_name(tree, parent_name)
        child = find_part_by_name(tree, child_name)
        if parent is None or child is None:
            continue
        error = bbox_contact_deviation(parent, child)
        if error is not None:
            errors.append(error)
    if len(errors) == 0:
        return None
    return float(np.mean(errors))

def sample_random_bbox_anchor(rng):
    anchor = rng.uniform(-0.5, 0.5, 3)
    axis = int(rng.integers(0, 3))
    sign = 1.0 if rng.random() > 0.5 else -1.0
    anchor[axis] = 0.5 * sign
    return anchor.tolist()

def randomize_tree_anchors(tree, seed=42):
    rng = np.random.default_rng(seed)

    def traverse(node):
        for child in node.get('children', []):
            if 'parent_anchor' in child:
                child['parent_anchor'] = sample_random_bbox_anchor(rng)
            if 'child_anchor' in child:
                child['child_anchor'] = sample_random_bbox_anchor(rng)
            traverse(child)
    traverse(tree)
    return tree
def build_partnet_same_geometry_representations(raw_anchor: dict, spec: dict, base_parts: list, normalization: dict, tolerance: float = 1e-9) -> dict:
    """Exactly re-encode an EXISTING Anchor layout; never optimize/reselect anchors."""
    from scipy.spatial.transform import Rotation
    check = validate_shared_part_spec(spec)
    if not check["valid"]:
        raise ValueError(str(check["errors"]))
    ids = [p["part_id"] for p in spec["parts"]]
    index = {p["part_id"]: p for p in base_parts}
    if len(index) != len(base_parts) or set(index) != set(ids):
        raise ValueError("Base layout IDs differ from shared parts")
    scale = float(normalization["scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid base normalization")
    root = spec["root_part_id"]
    parents = {e["child_part_id"]: e["parent_part_id"] for e in spec["connections"]}
    identity = {"center": [0.0, 0.0, 0.0], "scale": 1.0}
    encoded = {"anchor": copy.deepcopy(raw_anchor), "global": {"parts": []}, "local": {"parts": []}}
    # Change units only: preserve every original anchor, local rotation and offset direction.
    for part in encoded["anchor"]["parts"]:
        part["bbox_size"] = (np.asarray(part["bbox_size"], dtype=float) / scale).tolist()
        if part["part_id"] == root:
            part["center"] = ((np.asarray(part["center"]) - np.asarray(normalization["center"])) / scale).tolist()
        else:
            part["offset"] = (np.asarray(part["offset"], dtype=float) / scale).tolist()
    for pid in ids:
        part = index[pid]
        center = np.asarray(part["center"], dtype=float)
        rotation = np.asarray(part["world_rotation"], dtype=float)
        quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]].tolist()
        global_part = {"part_id": pid, "bbox_size": list(part["bbox_size"]), "center": center.tolist(), "rotation": quaternion}
        encoded["global"]["parts"].append(global_part)
        if pid == root:
            local_part = copy.deepcopy(global_part)
        else:
            parent = index[parents[pid]]
            parent_rotation = np.asarray(parent["world_rotation"], dtype=float)
            relative_position = parent_rotation.T @ (center - np.asarray(parent["center"]))
            relative_rotation = Rotation.from_matrix(parent_rotation.T @ rotation).as_quat()[[3, 0, 1, 2]].tolist()
            local_part = {"part_id": pid, "bbox_size": list(part["bbox_size"]), "relative_position": relative_position.tolist(), "relative_rotation": relative_rotation}
        encoded["local"]["parts"].append(local_part)
    errors = {}
    for method in REPRESENTATIONS:
        rebuilt = transform_partnet_representation(encoded[method], spec, method, identity)
        errors[method] = {field: max(float(np.max(np.abs(np.asarray(p[field]) - np.asarray(index[p["part_id"]][field])))) for p in rebuilt["parts"]) for field in ("center", "bbox_size", "world_rotation")}
        if max(errors[method].values()) > tolerance:
            raise ValueError(f"Same-geometry round-trip failed: {method}, {errors[method]}")
    return {"representations": encoded, "normalization": identity, "roundtrip_errors": errors}
def run_partnet_same_geometry_control(cfg: PartNetConfig) -> dict:
    """Offline, Anchor-initialized size intervention on the ORIGINAL complete-case set."""
    evaluation = load_json(cfg.output_path("evaluation_all.json"))
    shared = _index_objects(load_json(cfg.output_path("shared_specs.json")), "shared_specs.json")
    ids = [str(oid) for oid in evaluation["complete_ids"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Missing/duplicate complete-case IDs")
    bases = {str(r["object_id"]): r for r in evaluation["objects"] if r["method"] == "anchor"}
    factors = tuple(float(x) for x in cfg.scale_factors)
    if not factors or len(set(factors)) != len(factors) or any(not np.isfinite(x) or x <= 0 for x in factors):
        raise ValueError("scale_factors must be distinct positive finite numbers")
    tolerance = 1e-9
    records, checks = [], []
    for number, oid in enumerate(ids, 1):
        base = bases[oid]
        spec = shared[oid]["spec"]
        generated = load_json(cfg.output_path(f"generation/anchor/{oid}.json"))
        if generated.get("status") != "valid" or not spec or not spec["connections"]:
            raise ValueError(f"Missing valid source/edges for original complete case: {oid}")
        raw = generated["result"]
        fingerprint = content_hash({"experiment": "anchor-initialized-same-geometry-v1", "raw": raw, "spec": spec, "base_parts": base["layout_parts"], "normalization": base["normalization"], "factors": factors, "roundtrip_tolerance": tolerance})
        path = cfg.output_path(f"same_geometry/objects/{oid}.json")
        cached = _read_cache(path, fingerprint)
        if cached is None:
            bundle = build_partnet_same_geometry_representations(raw, spec, base["layout_parts"], base["normalization"], tolerance)
            initial = {p["part_id"]: p for p in base["layout_parts"]}
            conditions = []
            max_global_local_difference = 0.0
            for factor in factors:
                edge_records = {method: [] for method in REPRESENTATIONS}
                for edge in spec["connections"]:
                    parent, child = edge["parent_part_id"], edge["child_part_id"]
                    before = obb_sat_margin(initial[parent], initial[child])
                    after_margins = {}
                    for method in REPRESENTATIONS:
                        modified = copy.deepcopy(bundle["representations"][method])
                        target = next(p for p in modified["parts"] if p["part_id"] == child)
                        target["bbox_size"] = (np.asarray(target["bbox_size"]) * factor).tolist()
                        layout = transform_partnet_representation(modified, spec, method, bundle["normalization"])
                        index = {p["part_id"]: p for p in layout["parts"]}
                        after = obb_sat_margin(index[parent], index[child])
                        after_margins[method] = after
                        edge_records[method].append({"parent_part_id": parent, "child_part_id": child, "before_margin": before, "after_margin": after, "margin_drift": abs(after - before), "after_deviation": abs(after), "deviation_change": abs(after) - abs(before)})
                    max_global_local_difference = max(max_global_local_difference, abs(after_margins["global"] - after_margins["local"]))
                for method in REPRESENTATIONS:
                    rows = edge_records[method]
                    metrics = {"baseline_contact_deviation": float(np.mean([abs(e["before_margin"]) for e in rows])), "contact_deviation": float(np.mean([e["after_deviation"] for e in rows])), "margin_drift": float(np.mean([e["margin_drift"] for e in rows])), "deviation_change": float(np.mean([e["deviation_change"] for e in rows]))}
                    conditions.append({"method": method, "scale": factor, "metrics": metrics, "edges": rows})
            if max_global_local_difference > tolerance:
                raise ValueError(f"Global/Local resize-only equivalence failed: {oid}")
            check = {"object_id": oid, "category": base["category"], "max_roundtrip_error": max(value for errors in bundle["roundtrip_errors"].values() for value in errors.values()), "global_local_margin_max_difference": max_global_local_difference, "edge_count": len(spec["connections"])}
            cached = {"fingerprint": fingerprint, "object_id": oid, "category": base["category"], "check": check, "initial_representations": bundle["representations"], "normalization": bundle["normalization"], "conditions": conditions}
            save_json(cached, path)
        checks.append(cached["check"])
        for condition in cached["conditions"]:
            records.append({"object_id": oid, "category": base["category"], "method": condition["method"], "scale": condition["scale"], "metrics": condition["metrics"]})
        if number % max(1, cfg.show_every) == 0 or number == len(ids):
            print(f"Same-geometry control: {number}/{len(ids)}")
    metric_names = ("baseline_contact_deviation", "contact_deviation", "margin_drift", "deviation_change")
    summary = _summaries(records, ("method", "scale"), metric_names)
    category_summary = _summaries(records, ("category", "method", "scale"), metric_names)
    statistics = []
    for factor in factors:
        lookup = {m: {r["object_id"]: r for r in records if r["method"] == m and r["scale"] == factor} for m in REPRESENTATIONS}
        for baseline in ("global", "local"):
            result = paired_object_statistics(lookup["anchor"], lookup[baseline], "margin_drift", cfg)
            statistics.append({"experiment": "same_initial_geometry", "baseline": baseline, "metric": "margin_drift", "scale": factor, **result})
    _holm_adjust(statistics)
    protocol = {"experiment": "anchor-initialized same-initial-geometry size intervention", "source": "original Anchor layouts; anchors/offsets not optimized or reselected", "comparators": "Global/Local exact geometric re-encodings, NOT independent LLM generations", "primary_metric": "target-edge margin_drift", "intervention": "one child resized per shared edge; recompute from pristine parameters each time", "normalization": "original Anchor normalized frame fixed for all methods and interventions", "cohort": "original three-method complete cases; no new sample filtering", "scales": list(factors), "statistics": "object-level means; category-stratified bootstrap; Holm over 2 comparisons x scales", "tie_tolerance": tolerance, "bootstrap_samples": cfg.bootstrap_samples, "statistics_seed": cfg.statistics_seed, "limitations": "Conditional on Anchor initial geometry; target-edge OBB proxy; not whole-object collision or GT attachment-location correctness"}
    output = {"protocol": protocol, "object_count": len(ids), "complete_ids": ids, "summary": summary, "category_summary": category_summary, "paired_statistics": statistics, "checks": checks, "object_metrics": records}
    save_json(output, cfg.output_path("same_geometry/results.json"))
    for name in ("summary", "category_summary", "paired_statistics", "checks"):
        _write_csv(output[name], cfg.output_path(f"same_geometry/{name}.csv"))
    print(f"已保存 same_geometry/results.json；{len(ids)} objects，未调用 API。")
    return output
def refresh_partnet_statistics_from_cache(cfg: PartNetConfig) -> dict:
    """Recompute ONLY statistics. Preserve original results and all geometry caches."""
    original = load_json(cfg.output_path("tables/results.json"))
    complete = {str(r["object_id"]) for r in original["generation_status"] if r["complete_case"]}
    if not complete:
        raise ValueError("No complete cases in saved results")
    flat = original["object_metrics"]
    methods = (*REPRESENTATIONS, "random_anchor")
    lookup = {m: {} for m in methods}
    for row in flat:
        oid, method = str(row["object_id"]), row["method"]
        if oid in complete and method in lookup:
            if oid in lookup[method]:
                raise ValueError(f"Duplicate object metrics: {method}/{oid}")
            lookup[method][oid] = {"category": row["category"], "metrics": {k: row.get(k) for k in METRIC_DEFINITIONS}}
    if any(set(lookup[m]) != complete for m in methods):
        raise ValueError("Static/Random Anchor cached cohort is incomplete")
    static_statistics = []
    for baseline in ("global", "local", "random_anchor"):
        for metric in PRIMARY_METRICS:
            result = paired_object_statistics(lookup["anchor"], lookup[baseline], metric, cfg, metric in HIGHER_IS_BETTER)
            static_statistics.append({"experiment": "random_anchor" if baseline == "random_anchor" else "static", "baseline": baseline, "metric": metric, "scale": None, **result})
    _holm_adjust(static_statistics)
    perturbation = load_json(cfg.output_path("perturbation_all.json"))["objects"]
    perturbation = [r for r in perturbation if str(r["object_id"]) in complete]
    scales = sorted({float(r["scale"]) for r in perturbation})
    if not scales:
        raise ValueError("No saved perturbation conditions")
    perturbation_statistics = []
    for factor in scales:
        lookup = {m: {str(r["object_id"]): r for r in perturbation if r["method"] == m and r["scale"] == factor} for m in REPRESENTATIONS}
        if any(set(lookup[m]) != complete for m in REPRESENTATIONS):
            raise ValueError(f"Perturbation cached cohort incomplete at scale={factor}")
        for baseline in ("global", "local"):
            for metric in ("contact_deviation", "margin_drift", "deviation_change"):
                result = paired_object_statistics(lookup["anchor"], lookup[baseline], metric, cfg)
                perturbation_statistics.append({"experiment": "scale_perturbation", "baseline": baseline, "metric": metric, "scale": factor, **result})
    _holm_adjust(perturbation_statistics)
    original["paired_statistics"] = static_statistics + perturbation_statistics
    original["statistics_revision"] = {"version": "numerical-zero-consistent-v1", "tie_tolerance": 1e-9, "rule": "abs(paired difference)<=1e-9 is zero for W/T/L, Wilcoxon and bootstrap", "bootstrap_samples": cfg.bootstrap_samples, "statistics_seed": cfg.statistics_seed, "original_preserved": "tables/results.json", "geometry_recomputed": False, "api_calls": 0}
    save_json(original, cfg.output_path("tables/results_tie_corrected.json"))
    _write_csv(original["paired_statistics"], cfg.output_path("tables/paired_statistics_tie_corrected.csv"))
    print(f"统计已更新：{len(original['paired_statistics'])} tests；原始 tables/results.json 未覆盖。")
    return original
# =============================================================================
# EXPERIMENTS 3: Controlled language-guided parent-part editing
# Append this section once to methods.py AFTER the experiments2 completion patch.
# No PartNet mesh or GT-adjacency oracle is used to select or apply edits.
# =============================================================================

EDIT3_VERSION = "parent-resize-language-v1"


@dataclass
class EditingConfig:
    output_subdir: str = "experiments3"
    task_seed: int = 37
    use_llm_parser: bool = True
    parser_attempts: int = 2
    pilot_count: int = 22
    size_change_options: tuple = (0.8, 1.25)
    axes: tuple = ("x", "y", "z")
    target_margin_threshold: float = 0.003
    tolerance: float = 1e-9


def _edit3_path(cfg: PartNetConfig, ed: EditingConfig, suffix: str = "") -> Path:
    return cfg.output_path(f"{ed.output_subdir}/{suffix}")


def _edit3_tree(spec: dict) -> dict:
    check = validate_shared_part_spec(spec)
    if not check["valid"]:
        raise ValueError(f"Invalid shared structure: {check['errors']}")
    children = {p["part_id"]: [] for p in spec["parts"]}
    parents = {}
    for edge in spec["connections"]:
        parent, child = edge["parent_part_id"], edge["child_part_id"]
        children[parent].append(child)
        parents[child] = parent
    depths = {spec["root_part_id"]: 0}
    queue = [spec["root_part_id"]]
    for node in queue:
        for child in children[node]:
            depths[child] = depths[node] + 1
            queue.append(child)
    return {"children": children, "parents": parents, "depths": depths}


def build_experiments3_edit_tasks(cfg: PartNetConfig, ed: EditingConfig) -> dict:
    """Exactly one reproducible instruction per original complete-case object.

    Choose an internal geometry-bearing parent using ONLY the frozen shared tree;
    not by looking at GT boxes, per-method errors, or physical contact graph.
    """
    summary = load_json(cfg.output_path("evaluation_all.json"))
    shared = _index_objects(load_json(cfg.output_path("shared_specs.json")), "shared_specs.json")
    original_ids = [str(oid) for oid in summary["complete_ids"]]
    if not original_ids or len(set(original_ids)) != len(original_ids):
        raise ValueError("Original complete-case cohort missing or non-unique")
    tasks = []
    for index, oid in enumerate(original_ids):
        row = shared[oid]
        spec = row["spec"]
        if row["status"] != "valid" or spec is None:
            raise ValueError(f"Shared spec missing for {oid}")
        tree = _edit3_tree(spec)
        part_index = {p["part_id"]: p for p in spec["parts"]}
        counts = Counter(p["name"] for p in spec["parts"])
        candidates = [pid for pid, kids in tree["children"].items() if kids]
        if not candidates:
            raise ValueError(f"No geometry-bearing parent candidate: {oid}")
        # Use uniquely named non-root internal parts if possible, else uniquely
        # named root/internal parts. Stable fallback includes instance number.
        unique_internal = [pid for pid in candidates if pid != spec["root_part_id"] and counts[part_index[pid]["name"]] == 1]
        unique_any = [pid for pid in candidates if counts[part_index[pid]["name"]] == 1]
        pool = unique_internal or unique_any or candidates
        pool.sort(key=lambda pid: (-len(tree["children"][pid]), tree["depths"][pid], int(pid.split("_")[-1])))
        target = pool[0]
        part = part_index[target]
        # Balance axes/factors across ordered objects without relying on hash()
        axis = ed.axes[(index + ed.task_seed) % len(ed.axes)]
        factor = ed.size_change_options[(index // len(ed.axes) + ed.task_seed) % len(ed.size_change_options)]
        axis_word = {"x": "width", "y": "depth", "z": "height"}[axis]
        percent = round(abs(float(factor) - 1.0) * 100)
        direction = "Increase" if factor > 1 else "Decrease"
        if counts[part["name"]] == 1:
            label = f"the {part['name']}"
        else:
            label = f"the {part['name']} instance {part['instance_index'] + 1}"
        instruction = (f"{direction} the {axis_word} of {label} by {percent}% along its local {axis}-axis. "
                       "Keep the sizes of all other parts unchanged and preserve the existing attachments.")
        task = {"object_id": oid, "category": row["category"], "task_id": f"parent-edit-{oid}",
                "instruction": instruction,
                "expected_plan": {"target_part_id": target, "axis": axis, "scale_factor": float(factor)},
                "target_is_root": target == spec["root_part_id"], "target_depth": tree["depths"][target],
                "outgoing_edge_count": len(tree["children"][target]), "spec_fingerprint": content_hash(spec)}
        tasks.append(task)
    result = {"version": EDIT3_VERSION, "cohort": "Original three-branch complete cases; no filtering by editing outcome",
              "target_selection": "shared semantic tree only; prefer unique named non-root parent with children",
              "instruction_type": "one controlled natural-language axis-specific parent resize per object",
              "tasks": tasks, "object_count": len(tasks), "task_seed": ed.task_seed,
              "protocol_limitations": ["Parts and shared connections are fixed", "Single-step resizing, not free-form mesh editing", "Target identity/factor oracle used for controlled representation comparison", "LLM language parsing reported separately"]}
    path = _edit3_path(cfg, ed, "edit_tasks.json")
    if path.exists() and load_json(path) != result:
        raise ValueError("Edit tasks changed. Choose a new output_subdir instead of overwriting the frozen task set.")
    save_json(result, path)
    print(f"编辑任务固定：{len(tasks)} 条（内部 target={sum(not t['target_is_root'] for t in tasks)}）。")
    return result


def build_experiments3_parser_messages(instruction: str, spec: dict) -> list:
    parts = [{"part_id": p["part_id"], "name": p["name"], "instance_index": p["instance_index"]} for p in spec["parts"]]
    system = ("Interpret a controlled English 3D part-edit instruction into ONE operation. "
              "Select a part from the supplied immutable part inventory. "
              "The instruction's local x=width, y=depth, z=height; convert percent increase/decrease to an exact scale factor. "
              "Do not predict or modify geometries, connections, anchors or placements. "
              'Return only JSON: {"target_part_id":"part_0","axis":"x","scale_factor":1.25}.')
    user = json.dumps({"instruction": instruction, "fixed_parts": parts}, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_experiments3_parsed_plan(result: dict, spec: dict) -> dict:
    if not isinstance(result, dict) or set(result) != {"target_part_id", "axis", "scale_factor"}:
        return {"valid": False, "errors": ["exactly_three_plan_fields_required"]}
    ids = {p["part_id"] for p in spec["parts"]}
    errors = []
    if result["target_part_id"] not in ids:
        errors.append("target_not_in_fixed_inventory")
    if result["axis"] not in ("x", "y", "z"):
        errors.append("axis_must_be_xyz")
    scale = result["scale_factor"]
    if type(scale) not in (float, int) or not np.isfinite(scale) or not 0.5 <= scale <= 1.6:
        errors.append("invalid_scale_factor")
    return {"valid": not errors, "errors": errors}


def run_experiments3_language_parser(tasks_data: dict, cfg: PartNetConfig, ed: EditingConfig, client=None) -> dict:
    """Use ONE shared parser for the three branches; do not repair semantic errors.

    Formatting/schema errors may retry, but mismatch with intended target is a
    measured language-understanding failure, not a hint sent back to the LLM.
    """
    shared = _index_objects(load_json(cfg.output_path("shared_specs.json")), "shared_specs.json")
    parser_cfg = copy.deepcopy(cfg)
    parser_cfg.max_attempts = ed.parser_attempts
    result_rows, jobs = [], []
    for task in tasks_data["tasks"]:
        oid = task["object_id"]
        spec = shared[oid]["spec"]
        messages = build_experiments3_parser_messages(task["instruction"], spec)
        fp = content_hash({"version": EDIT3_VERSION, "messages": messages, "request": parser_cfg.request_config(), "attempts": ed.parser_attempts})
        path = _edit3_path(cfg, ed, f"language_parse/{oid}.json")
        cached = _read_cache(path, fp)
        if cached is None or cached.get("status") not in ("valid", "exhausted"):
            jobs.append((task, spec, messages, fp, path))
    if jobs and ed.use_llm_parser and cfg.allow_api:
        client = client or create_deepseek_client(base_url=cfg.base_url, timeout=cfg.timeout)
        def worker(job):
            task, spec, messages, fp, path = job
            return _generate_valid_json(client, parser_cfg, path, messages,
                                        lambda value: validate_experiments3_parsed_plan(value, spec), fp)
        _run_jobs([{"object_id": job[0]["object_id"], "job": job} for job in jobs],
                  lambda row: worker(row["job"]), cfg, "Language edit parsing")
    for task in tasks_data["tasks"]:
        oid = task["object_id"]
        spec = shared[oid]["spec"]
        messages = build_experiments3_parser_messages(task["instruction"], spec)
        fp = content_hash({"version": EDIT3_VERSION, "messages": messages, "request": parser_cfg.request_config(), "attempts": ed.parser_attempts})
        cache = _read_cache(_edit3_path(cfg, ed, f"language_parse/{oid}.json"), fp)
        if cache is None:
            status, plan, attempts = "not_run", None, 0
        else:
            status, plan, attempts = cache.get("status"), cache.get("result"), cache.get("attempt_count", 0)
        gold = task["expected_plan"]
        correct = bool(status == "valid" and plan is not None
                       and plan.get("target_part_id") == gold["target_part_id"]
                       and plan.get("axis") == gold["axis"]
                       and abs(float(plan["scale_factor"]) - gold["scale_factor"]) <= 1e-6)
        result_rows.append({"object_id": oid, "category": task["category"], "status": status,
                            "parse_correct": correct, "attempt_count": attempts, "parsed_plan": plan})
    output = {"version": EDIT3_VERSION, "rows": result_rows, "total": len(result_rows),
              "valid_schema": sum(r["status"] == "valid" for r in result_rows),
              "correct": sum(r["parse_correct"] for r in result_rows),
              "parser_mode": "shared_llm" if ed.use_llm_parser else "not_run",
              "note": "LLM sees fixed parts and instruction, not oracle expected plan; semantic mistakes are not retried"}
    save_json(output, _edit3_path(cfg, ed, "language_parse_summary.json"))
    print(f"语言解析：正确 {output['correct']}/{output['total']}，schema valid {output['valid_schema']}/{output['total']}。")
    return output


def _edit3_apply_raw(raw: dict, plan: dict) -> dict:
    output = copy.deepcopy(raw)
    idx = {"x": 0, "y": 1, "z": 2}[plan["axis"]]
    target = next(p for p in output["parts"] if p["part_id"] == plan["target_part_id"])
    target["bbox_size"][idx] *= float(plan["scale_factor"])
    return output


def evaluate_experiments3_edit(before: list, after: list, spec: dict, plan: dict, threshold: float = 0.003, tolerance: float = 1e-9) -> dict:
    """Common geometric proxies for all representations; no anchor-specific metric.

    Primary: change of OBB SAT margin on the target's outgoing shared edges.
    For this parent-resize task, those children are not themselves resized.
    """
    b = {p["part_id"]: p for p in before}
    a = {p["part_id"]: p for p in after}
    if set(a) != set(b):
        raise ValueError("Edit silently changed part identities")
    target_id, axis = plan["target_part_id"], {"x": 0, "y": 1, "z": 2}[plan["axis"]]
    if target_id not in b:
        raise ValueError("Edited target missing")
    initial_size = float(b[target_id]["bbox_size"][axis])
    if initial_size <= 0:
        raise ValueError("Nonpositive edited dimension")
    actual_factor = float(a[target_id]["bbox_size"][axis]) / initial_size
    relatives = _edit3_tree(spec)
    descendants = set()
    stack = list(relatives["children"][target_id])
    while stack:
        child = stack.pop()
        descendants.add(child)
        stack.extend(relatives["children"][child])
    outgoing = [(target_id, child) for child in relatives["children"][target_id]]
    inside = [(e["parent_part_id"], e["child_part_id"]) for e in spec["connections"]
              if e["parent_part_id"] in descendants and e["child_part_id"] in descendants]
    incoming = [(relatives["parents"][target_id], target_id)] if target_id in relatives["parents"] else []
    def margin_info(edges):
        if not edges:
            return {"count": 0, "before_abs": None, "after_abs": None, "margin_drift": None, "after_tolerance_rate": None}
        before_m = np.array([obb_sat_margin(b[u], b[v]) for u, v in edges])
        after_m = np.array([obb_sat_margin(a[u], a[v]) for u, v in edges])
        return {"count": len(edges), "before_abs": float(np.mean(np.abs(before_m))),
                "after_abs": float(np.mean(np.abs(after_m))),
                "margin_drift": float(np.mean(np.abs(after_m - before_m))),
                "after_tolerance_rate": float(np.mean(np.abs(after_m) <= threshold))}
    other_ids = set(b) - descendants - {target_id}
    outside_center_max = max((float(np.linalg.norm(np.asarray(a[pid]["center"]) - np.asarray(b[pid]["center"]))) for pid in other_ids), default=0.0)
    other_size_max = max((float(np.max(np.abs(np.asarray(a[pid]["bbox_size"]) - np.asarray(b[pid]["bbox_size"])))) for pid in b if pid != target_id), default=0.0)
    descendant_moved = sum(float(np.linalg.norm(np.asarray(a[pid]["center"]) - np.asarray(b[pid]["center"]))) > tolerance for pid in descendants)
    outgoing_info = margin_info(outgoing)
    incoming_info = margin_info(incoming)
    inside_info = margin_info(inside)
    return {"outgoing_margin_drift": outgoing_info["margin_drift"],
            "outgoing_after_abs_margin": outgoing_info["after_abs"],
            "outgoing_before_abs_margin": outgoing_info["before_abs"],
            "outgoing_contact_proxy_rate": outgoing_info["after_tolerance_rate"],
            "incoming_margin_drift": incoming_info["margin_drift"],
            "descendant_internal_margin_drift": inside_info["margin_drift"],
            "outside_subtree_max_center_drift": outside_center_max,
            "untargeted_max_size_drift": other_size_max,
            "target_scale_factor_error": abs(actual_factor - float(plan["scale_factor"])),
            "moved_descendant_count": int(descendant_moved),
            "outgoing_edge_count": len(outgoing),
            "subtree_size": 1 + len(descendants)}


def run_experiments3_edit_evaluation(tasks_data: dict, cfg: PartNetConfig, ed: EditingConfig) -> dict:
    """Both independent-generation and Anchor-initialized same-geometry regimes.

    Each uses the template's frozen intended plan. Parser performance is assessed
    separately; no model receives GT attachment locations or GT geometry here.
    """
    main = load_json(cfg.output_path("evaluation_all.json"))
    shared = _index_objects(load_json(cfg.output_path("shared_specs.json")), "shared_specs.json")
    original_ids = [str(oid) for oid in main["complete_ids"]]
    if [t["object_id"] for t in tasks_data["tasks"]] != original_ids:
        raise ValueError("Task cohort/order differs from the original complete cases")
    base_lookup = {(r["object_id"], r["method"]): r for r in main["objects"]}
    task_lookup = {t["object_id"]: t for t in tasks_data["tasks"]}
    all_records = []
    errors = []
    for pos, oid in enumerate(original_ids, 1):
        task = task_lookup[oid]
        spec = shared[oid]["spec"]
        plan = task["expected_plan"]
        source = {method: load_json(cfg.output_path(f"generation/{method}/{oid}.json")) for method in REPRESENTATIONS}
        if any(r.get("status") != "valid" for r in source.values()):
            raise ValueError(f"Original valid-generation cache missing: {oid}")
        # Prespecified descriptor, NOT used to select tasks or define the main cohort.
        # If all outgoing parent anchors have 0 along the edited axis, a dimension
        # change cannot move their attachment locations along that axis.
        axis_idx = {"x": 0, "y": 1, "z": 2}[plan["axis"]]
        anchor_index = {p["part_id"]: p for p in source["anchor"]["result"]["parts"]}
        child_ids = _edit3_tree(spec)["children"][plan["target_part_id"]]
        leverage = float(np.mean([abs(float(anchor_index[child]["parent_anchor"][axis_idx])) for child in child_ids]))
        relevance_group = "attachment_axis_active" if leverage >= 0.25 else "attachment_axis_inactive"
        same_path = cfg.output_path(f"same_geometry/objects/{oid}.json")
        if not same_path.exists():
            raise FileNotFoundError(f"Missing experiments2 mechanism cache: {same_path}")
        same = load_json(same_path)
        if set(same["initial_representations"]) != set(REPRESENTATIONS):
            raise ValueError(f"Missing same-geometry re-encodings: {oid}")
        original = {method: base_lookup[(oid, method)] for method in REPRESENTATIONS}
        fingerprint = content_hash({"experiment": EDIT3_VERSION, "plan": plan, "spec": spec,
                                    "generation": {m: source[m].get("fingerprint") for m in REPRESENTATIONS},
                                    "same_geometry_fp": same["fingerprint"],
                                    "evaluation": {m: original[m].get("fingerprint") for m in REPRESENTATIONS},
                                    "threshold": ed.target_margin_threshold})
        path = _edit3_path(cfg, ed, f"geometry_edit/{oid}.json")
        cache = _read_cache(path, fingerprint)
        if cache is None:
            rows = []
            for regime in ("independent_generation", "same_initial_geometry"):
                for method in REPRESENTATIONS:
                    if regime == "independent_generation":
                        raw = source[method]["result"]
                        before = original[method]["layout_parts"]
                        normal = original[method]["normalization"]
                    else:
                        raw = same["initial_representations"][method]
                        before = original["anchor"]["layout_parts"]
                        normal = same.get("normalization", {"center": [0, 0, 0], "scale": 1.0})
                    after_layout = transform_partnet_representation(_edit3_apply_raw(raw, plan), spec, method, normal)
                    metrics = evaluate_experiments3_edit(before, after_layout["parts"], spec, plan,
                                                         ed.target_margin_threshold, ed.tolerance)
                    rows.append({"object_id": oid, "category": task["category"], "regime": regime,
                                 "method": method, "target_is_root": task["target_is_root"],
                                 "axis": plan["axis"], "factor": plan["scale_factor"],
                                 "anchor_axis_leverage": leverage, "axis_relevance_group": relevance_group,
                                 "metrics": metrics})
            cache = {"object_id": oid, "fingerprint": fingerprint, "task": task, "records": rows}
            save_json(cache, path)
        all_records.extend(cache["records"])
        if pos % max(1, cfg.show_every) == 0 or pos == len(original_ids):
            print(f"语言编辑几何执行：{pos}/{len(original_ids)}")
    result = {"version": EDIT3_VERSION, "object_count": len(original_ids), "records": all_records,
              "protocol": {"edit_plan": "deterministic intended target of controlled language instruction; parser measured separately",
                           "regimes": ["independent_generation", "same_initial_geometry"],
                           "comparison": "native parameter update, no external attachment solver or LLM geometry regeneration",
                           "edit": "resize ONE geometry-bearing parent along one local axis; connected children sizes fixed",
                           "primary": "outgoing_edge OBB SAT margin drift; average outgoing edges within object",
                           "metric_limitations": "OBB SAT is not exact mesh contact; source descriptions do not specify unique GT layout"}}
    save_json(result, _edit3_path(cfg, ed, "geometry_results.json"))
    return result


def summarize_experiments3_edit_results(tasks_data: dict, parser: dict, geometry: dict, cfg: PartNetConfig, ed: EditingConfig) -> dict:
    parse_map = {r["object_id"]: r for r in parser["rows"]}
    complete = {task["object_id"] for task in tasks_data["tasks"]}
    if len(complete) != geometry["object_count"]:
        raise ValueError("Object count mismatch")
    regimes = ("independent_generation", "same_initial_geometry")
    metric_names = ("outgoing_margin_drift", "outgoing_after_abs_margin", "outgoing_contact_proxy_rate",
                    "incoming_margin_drift", "descendant_internal_margin_drift",
                    "outside_subtree_max_center_drift", "untargeted_max_size_drift", "target_scale_factor_error", "moved_descendant_count")
    rows = []
    parsed_correct = {oid for oid, r in parse_map.items() if r["parse_correct"]}
    for row in geometry["records"]:
        rows.append({**row, "parser_correct": row["object_id"] in parsed_correct})
    summary = []
    by_category = []
    axis_relevance_summary = []
    stats = []
    for regime in regimes:
        regime_rows = [r for r in rows if r["regime"] == regime]
        for subset, subset_rows in (("intended_plan_full", regime_rows),
                                    ("correctly_parsed_subset", [r for r in regime_rows if r["parser_correct"]])):
            # Do not interpret the conditional subset as full-population effect.
            for row in _summaries(subset_rows, ("method",), metric_names):
                summary.append({"regime": regime, "subset": subset, **row})
            for row in _summaries(subset_rows, ("category", "method"), metric_names):
                by_category.append({"regime": regime, "subset": subset, **row})
            for row in _summaries(subset_rows, ("axis_relevance_group", "method"), metric_names):
                axis_relevance_summary.append({"regime": regime, "subset": subset, **row})
            lookup = {method: {r["object_id"]: r for r in subset_rows if r["method"] == method} for method in REPRESENTATIONS}
            for baseline in ("global", "local"):
                for metric in ("outgoing_margin_drift", "outgoing_after_abs_margin", "outgoing_contact_proxy_rate"):
                    result = paired_object_statistics(lookup["anchor"], lookup[baseline], metric, cfg,
                                                      higher_is_better=(metric == "outgoing_contact_proxy_rate"))
                    stats.append({"regime": regime, "subset": subset, "baseline": baseline, "metric": metric, **result})
    for regime in regimes:
        for subset in ("intended_plan_full", "correctly_parsed_subset"):
            family = [r for r in stats if r["regime"] == regime and r["subset"] == subset]
            _holm_adjust(family)
    output = {"version": EDIT3_VERSION, "cohort_size": len(complete), "parser_success": parser["correct"],
              "parser_schema_valid": parser["valid_schema"], "summary": summary,
              "category_summary": by_category, "axis_relevance_summary": axis_relevance_summary,
              "paired_statistics": stats,
              "protocol": geometry["protocol"], "sample_policy": "full original complete-case cohort for intended-plan geometry; parsed-correct subset separately labeled",
              "metrics_note": "Size-change accuracy and untouched-size preservation are deterministic invariants, NOT evidence of superior LLM understanding."}
    save_json(output, _edit3_path(cfg, ed, "final_results.json"))
    _write_csv(summary, _edit3_path(cfg, ed, "summary.csv"))
    _write_csv(by_category, _edit3_path(cfg, ed, "category_summary.csv"))
    _write_csv(axis_relevance_summary, _edit3_path(cfg, ed, "axis_relevance_summary.csv"))
    _write_csv(stats, _edit3_path(cfg, ed, "paired_statistics.csv"))
    print(f"Experiments3：{len(complete)} objects；解析正确 {parser['correct']}/{len(complete)}；两种初始布局条件统计已保存。")
    return output
# =============================================================================
# Experiments 4: GT-conditioned PartNet-Mobility kinematic representation study
# This is NOT LLM-predicted joint discovery or a physics simulator.
# =============================================================================
from pathlib import Path as _P4Path
from collections import defaultdict as _p4_defaultdict
from scipy.spatial.transform import Rotation as _P4Rotation
import xml.etree.ElementTree as _p4_ET
import hashlib as _p4_hashlib


def _p4_vec(text, default=(0., 0., 0.)):
    if text is None:
        return np.asarray(default, dtype=float)
    arr = np.fromstring(str(text).replace(',', ' '), sep=' ', dtype=float)
    if len(arr) != 3 or not np.isfinite(arr).all():
        raise ValueError('Invalid 3-vector: ' + str(text))
    return arr


def _p4_tf(origin):
    matrix = np.eye(4, dtype=float)
    if origin is not None:
        matrix[:3, 3] = _p4_vec(origin.get('xyz'))
        matrix[:3, :3] = _P4Rotation.from_euler('xyz', _p4_vec(origin.get('rpy'))).as_matrix()
    return matrix


def _p4_rigid(rotation=None, translation=None):
    matrix = np.eye(4, dtype=float)
    if rotation is not None:
        matrix[:3, :3] = rotation
    if translation is not None:
        matrix[:3, 3] = translation
    return matrix


def _p4_mesh_path(folder, filename):
    name = (filename or '').replace('package://', '').replace('file://', '').replace('\\', '/')
    candidates = [folder / name, folder / 'textured_objs' / _P4Path(name).name, folder / 'new_objs' / _P4Path(name).name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError('Mesh not found: ' + str(filename))


def _p4_obj_vertices(path):
    """OBJ face-referenced vertices only, incl. negative indices; independent per-file tables."""
    vertices, indices = [], set()
    with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
        for row in handle:
            if row.startswith('v '):
                xyz = row.strip().split()
                vertices.append([float(xyz[1]), float(xyz[2]), float(xyz[3])])
            elif row.startswith('f '):
                for item in row.strip().split()[1:]:
                    token = item.split('/')[0]
                    if not token:
                        continue
                    idx = int(token)
                    indices.add(idx - 1 if idx > 0 else len(vertices) + idx)
    if not indices:
        raise ValueError('Mesh has no faces: ' + str(path))
    points = np.asarray(vertices, dtype=float)
    ids = sorted(indices)
    if min(ids) < 0 or max(ids) >= len(points):
        raise ValueError('OBJ references an invalid vertex: ' + str(path))
    return points[ids]


def _p4_meta_category(folder):
    meta = folder / 'meta.json'
    if meta.is_file():
        try:
            val = json.loads(meta.read_text(encoding='utf-8'))
            for name in ('model_cat', 'category', 'cat'):
                if val.get(name):
                    return str(val[name])
        except (ValueError, TypeError):
            pass
    metadata = folder / 'metadata.txt'
    if metadata.is_file():
        for line in metadata.read_text(encoding='utf-8', errors='ignore').splitlines():
            if 'model_cat' in line or 'category' in line:
                return line.split(':', 1)[-1].strip() or 'Unknown'
    if not folder.name.isdigit():
        return folder.name
    return 'Unknown'


def discover_experiments4_assets(mobility_root, max_depth=4):
    """Find native mobility.urdf without modifying any dataset files."""
    root = _P4Path(mobility_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f'PartNet-Mobility data missing: {root}. Set mob_root in Cell 2 to the dataset root containing mobility.urdf.')
    rows = []
    for folder, dirs, files in os.walk(root):
        rel_depth = len(_P4Path(folder).relative_to(root).parts)
        if 'mobility.urdf' in files:
            path = _P4Path(folder)
            rows.append({'object_id': path.name, 'category': _p4_meta_category(path), 'asset_dir': str(path.resolve())})
            dirs[:] = []
        elif rel_depth >= max_depth:
            dirs[:] = []
    rows.sort(key=lambda r: (r['category'], r['object_id'], r['asset_dir']))
    if not rows:
        raise FileNotFoundError(f'No mobility.urdf found up to depth {max_depth} beneath {root}. This is a separate dataset from static PartNet.')
    return rows


def _p4_parse_urdf(folder):
    document = _p4_ET.parse(folder / 'mobility.urdf')
    robot = document.getroot()
    link_meshes, joints = {}, []
    for link in robot.findall('link'):
        name = link.get('name')
        if not name or name in link_meshes:
            raise ValueError('Duplicate/missing link name')
        visuals = []
        for visual in link.findall('visual'):
            mesh = visual.find('./geometry/mesh')
            if mesh is None or not mesh.get('filename'):
                continue
            visuals.append({'filename': mesh.get('filename'), 'origin': _p4_tf(visual.find('origin')), 'scale': _p4_vec(mesh.get('scale'), (1, 1, 1))})
        link_meshes[name] = visuals
    for joint in robot.findall('joint'):
        kind = joint.get('type', '').lower()
        if kind not in ('fixed', 'continuous', 'revolute', 'prismatic'):
            raise ValueError('Unsupported joint type: ' + kind)
        parent, child = joint.find('parent'), joint.find('child')
        if parent is None or child is None:
            raise ValueError('Incomplete URDF joint')
        parent_name, child_name = parent.get('link'), child.get('link')
        if parent_name not in link_meshes or child_name not in link_meshes:
            raise ValueError('URDF joint links missing from links')
        axis_elem, limits = joint.find('axis'), joint.find('limit')
        axis = _p4_vec(axis_elem.get('xyz') if axis_elem is not None else None, (1, 0, 0))
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError('Joint has zero axis')
        lower = float(limits.get('lower')) if limits is not None and limits.get('lower') is not None else None
        upper = float(limits.get('upper')) if limits is not None and limits.get('upper') is not None else None
        if kind in ('revolute', 'prismatic') and (lower is None or upper is None):
            raise ValueError('Joint missing limits')
        joints.append({'name': joint.get('name') or f'joint_{len(joints)}', 'type': kind, 'parent': parent_name, 'child': child_name, 'origin': _p4_tf(joint.find('origin')), 'axis': axis/norm, 'lower': lower, 'upper': upper})
    return link_meshes, joints


def _p4_link_bounds(folder, link_meshes):
    cache, bounds = {}, {}
    for link, visuals in link_meshes.items():
        candidates = []
        for visual in visuals:
            path = _p4_mesh_path(folder, visual['filename'])
            if path not in cache:
                cache[path] = _p4_obj_vertices(path)
            pts = cache[path] * visual['scale']
            pts = pts @ visual['origin'][:3, :3].T + visual['origin'][:3, 3]
            candidates.append(pts)
        if candidates:
            full = np.concatenate(candidates)
            lo, hi = full.min(axis=0), full.max(axis=0)
            bounds[link] = {'center': (lo+hi)/2, 'size': hi-lo}
    return bounds


def _p4_fk(links, joints, states):
    by_child = {}
    for joint in joints:
        if joint['child'] in by_child:
            raise ValueError('Multiple incoming joints for a link')
        by_child[joint['child']] = joint
    world, visiting = {}, set()
    def solve(name):
        if name in world:
            return world[name]
        if name in visiting:
            raise ValueError('Cyclic URDF links')
        visiting.add(name)
        joint = by_child.get(name)
        if joint is None:
            mat = np.eye(4)
        else:
            q = float(states.get(joint['name'], 0.0))
            motion = np.eye(4)
            if joint['type'] in ('revolute', 'continuous'):
                motion[:3,:3] = _P4Rotation.from_rotvec(joint['axis'] * q).as_matrix()
            elif joint['type'] == 'prismatic':
                motion[:3,3] = joint['axis'] * q
            mat = solve(joint['parent']) @ joint['origin'] @ motion
        visiting.remove(name)
        world[name] = mat
        return mat
    for link in links:
        solve(link)
    return world


def _p4_base_q(joint):
    low, high = joint['lower'], joint['upper']
    if low is None or high is None:
        return 0.0
    return float(np.clip(0., low, high))


def _p4_motion_step(joint, object_scale, max_angle=0.7, fraction=0.55):
    q0 = _p4_base_q(joint)
    if joint['type'] == 'continuous':
        return q0, q0 + max_angle
    lo, hi = joint['lower'], joint['upper']
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
        return None
    positive, negative = hi-q0, q0-lo
    amount = positive if positive >= negative else -negative
    if joint['type'] == 'revolute':
        delta = np.sign(amount) * min(abs(amount)*fraction, max_angle)
        return (q0, q0 + delta) if abs(delta) >= 0.08 else None
    delta = np.sign(amount) * min(abs(amount)*fraction, object_scale * 0.25)
    return (q0, q0 + delta) if abs(delta)/object_scale >= 0.015 else None


def prepare_experiments4_asset(asset, max_joints=50):
    """Extract TRUE URDF kinematics; no semantic tree is treated as physical GT."""
    folder = _P4Path(asset['asset_dir'])
    links, joints = _p4_parse_urdf(folder)
    if not joints or len(joints) > max_joints:
        raise ValueError('Empty/too-many-joints object')
    bounds = _p4_link_bounds(folder, links)
    q_base = {j['name']: _p4_base_q(j) for j in joints}
    fk0 = _p4_fk(links, joints, q_base)
    boxes = {}
    for link, bb in bounds.items():
        tf = fk0[link]
        boxes[link] = {'center': tf[:3,:3] @ bb['center'] + tf[:3,3], 'size': bb['size'], 'rotation': tf[:3,:3]}
    if len(boxes) < 2:
        raise ValueError('Insufficient visual mesh links')
    signs = np.asarray(list(itertools.product((-0.5,0.5),repeat=3)))
    points = np.concatenate([box['center']+(signs*box['size']) @ box['rotation'].T for box in boxes.values()])
    lo, hi = points.min(axis=0), points.max(axis=0)
    scale = float(np.max(hi-lo))
    if scale <= 1e-10:
        raise ValueError('Degenerate object normalization')
    shift = (hi+lo)/2
    descendants = _p4_defaultdict(set)
    for joint in joints:
        descendants[joint['parent']].add(joint['child'])
    def subtree(link):
        seen, todo = set(), [link]
        while todo:
            k = todo.pop()
            if k not in seen:
                seen.add(k)
                todo.extend(descendants[k])
        return sorted(seen)
    rows = []
    for joint in joints:
        if joint['type'] == 'fixed' or joint['parent'] not in boxes or joint['child'] not in boxes:
            continue
        step = _p4_motion_step(joint, scale)
        if step is None:
            continue
        q0, q1 = step
        after_q = dict(q_base)
        after_q[joint['name']] = q1
        fk1 = _p4_fk(links, joints, after_q)
        changed = [name for name in subtree(joint['child']) if name in boxes]
        if not changed:
            continue
        parent_box, child_box = boxes[joint['parent']], boxes[joint['child']]
        origin_world = fk0[joint['parent']] @ joint['origin']
        pivot = origin_world[:3,3]
        axis_world = origin_world[:3,:3] @ joint['axis']
        pa = parent_box['rotation'].T @ (pivot-parent_box['center'])/np.maximum(parent_box['size'], 1e-10)
        ca = child_box['rotation'].T @ (pivot-child_box['center'])/np.maximum(child_box['size'], 1e-10)
        base = {}
        target = {}
        for link in changed:
            orig = boxes[link]
            tf1 = fk1[link]
            geom = bounds[link]
            base[link] = {'center': ((orig['center']-shift)/scale).tolist(), 'rotation': orig['rotation'].tolist(), 'size': (orig['size']/scale).tolist()}
            target[link] = {'center': ((tf1[:3,:3]@geom['center']+tf1[:3,3]-shift)/scale).tolist(), 'rotation': tf1[:3,:3].tolist(), 'size': (orig['size']/scale).tolist()}
        rows.append({'joint_name': joint['name'], 'kind': 'hinge' if joint['type'] in ('revolute','continuous') else 'slider', 'parent': joint['parent'], 'child': joint['child'], 'q_start': q0, 'q_end': q1, 'delta': q1-q0, 'angle_or_distance_units': 'radians' if joint['type'] != 'prismatic' else 'native_asset_length', 'pivot_normalized': ((pivot-shift)/scale).tolist(), 'axis_world': axis_world.tolist(), 'parent_anchor': pa.tolist(), 'child_anchor': ca.tolist(), 'rest_parent': {'center':((parent_box['center']-shift)/scale).tolist(),'size':(parent_box['size']/scale).tolist(),'rotation':parent_box['rotation'].tolist()}, 'normalization_scale':scale, 'parent_anchor_inside': bool(np.all(np.abs(pa)<=0.500001)), 'child_anchor_inside': bool(np.all(np.abs(ca)<=0.500001)), 'changed_links': changed, 'initial': base, 'reference': target, 'translation_normalized': ((axis_world*(q1-q0)/scale) if joint['type']=='prismatic' else np.zeros(3)).tolist()})
    if not rows:
        raise ValueError('No eligible hinge/slider joints with visible parent and child')
    return {'object_id': asset['object_id'], 'category': asset['category'], 'asset_dir': asset['asset_dir'], 'link_count': len(links), 'geometry_link_count': len(boxes), 'normalization_scale': scale, 'events': rows}


def build_experiments4_manifest(mobility_root, output_dir, seed=42, max_per_category=20, max_categories=10, min_per_category=3):
    """Frozen deterministic category-balanced eligible object cohort."""
    dest = _P4Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/'benchmark_manifest.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf8'))
        if str(_P4Path(old['root']).resolve()) != str(_P4Path(mobility_root).resolve()):
            raise ValueError('Frozen manifest root differs; use another output_dir')
        print('Reuse frozen Mobility manifest:', len(old['objects']), 'objects')
        return old
    assets = discover_experiments4_assets(mobility_root)
    grouped = _p4_defaultdict(list)
    for item in assets:
        grouped[item['category']].append(item)
    # Selection is based on native data eligibility, not geometry or method scores.
    eligible = _p4_defaultdict(list)
    skipped = _p4_defaultdict(int)
    preferred = ['Door','StorageFurniture','Refrigerator','Microwave','Laptop','Faucet','WashingMachine','Dishwasher','Scissors','Safe','Window','Drawer']
    ranking = lambda cat: (preferred.index(cat) if cat in preferred else len(preferred), -len(grouped[cat]), cat)
    for category in sorted(grouped, key=ranking):
        category_assets = grouped[category]
        if sum(len(v)>=min_per_category for v in eligible.values())>=max_categories:
            break
        ordered = sorted(category_assets, key=lambda r: _p4_hashlib.sha256(f"{seed}|{r['asset_dir']}".encode()).hexdigest())
        for item in ordered:
            if len(eligible[category]) >= max_per_category:
                break
            try:
                prepared = prepare_experiments4_asset(item)
                cache_file = dest/'prepared_cache'/f"{category}_{item['object_id']}.json"
                cache_file.parent.mkdir(parents=True,exist_ok=True)
                cache_file.write_text(json.dumps(prepared,ensure_ascii=False),encoding='utf8')
                eligible[category].append({'object_id': item['object_id'], 'category': category, 'asset_dir': item['asset_dir'], 'joint_count': len(prepared['events']), 'hinge_count': sum(e['kind']=='hinge' for e in prepared['events']), 'slider_count': sum(e['kind']=='slider' for e in prepared['events'])})
            except (ValueError, FileNotFoundError, OSError, _p4_ET.ParseError):
                skipped[category] += 1
    accepted_categories = [k for k in sorted(eligible, key=ranking) if len(eligible[k]) >= min_per_category][:max_categories]
    chosen = [item for category in accepted_categories for item in eligible[category]]
    if not chosen:
        raise RuntimeError('No eligible Mobility assets. Check URDF mesh filenames, chosen root, and source data integrity.')
    manifest = {'protocol':'GT-conditioned native URDF joint intervention; not LLM joint discovery', 'root':str(_P4Path(mobility_root).resolve()), 'seed':seed, 'max_per_category':max_per_category, 'max_categories':max_categories, 'min_per_category':min_per_category, 'categories':accepted_categories, 'objects':chosen, 'discovered_assets':len(assets), 'skipped_by_category':dict(skipped)}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf8')
    print('Frozen Mobility:', len(chosen), 'objects;', sum(i['joint_count'] for i in chosen), 'candidate joint events;', accepted_categories)
    return manifest


def evaluate_experiments4_event(event):
    """Fair base pose + same URDF action, compare native update rules."""
    pivot = np.asarray(event['pivot_normalized'])
    axis = np.asarray(event['axis_world'])
    angle = event['delta'] if event['kind']=='hinge' else 0.
    Q = _P4Rotation.from_rotvec(axis * angle).as_matrix()
    displacement = np.asarray(event['translation_normalized'])
    child_center0 = np.asarray(event['initial'][event['child']]['center'])
    result = []
    for method in ('anchor_joint','local_center','local_pivot_aware','global_frozen'):
        errors, orientation_errors = [], []
        child_actual = None
        for name in event['changed_links']:
            bb = event['initial'][name]
            c = np.asarray(bb['center'])
            R = np.asarray(bb['rotation'])
            if method in ('anchor_joint','local_pivot_aware'):
                c2 = pivot + Q @ (c-pivot) + displacement
                R2 = Q @ R
            elif method=='local_center':
                c2 = child_center0 + Q @ (c - child_center0) if event['kind']=='hinge' else c+displacement
                R2 = Q @ R
            else:
                c2, R2 = c, R
            ref = event['reference'][name]
            errors.append(float(np.linalg.norm(c2 - np.asarray(ref['center']))))
            trace = (np.trace(R2.T @ np.asarray(ref['rotation']))-1.)/2.
            orientation_errors.append(float(np.arccos(np.clip(trace,-1.,1.))))
            if name==event['child']:
                child_actual = (c2, R2)
        initial_child = event['initial'][event['child']]
        joint_local = np.asarray(event['child_anchor'])*np.asarray(initial_child['size'])
        actual_pivot = child_actual[0] + child_actual[1] @ joint_local
        expected_pivot = pivot + displacement
        pivot_error = float(np.linalg.norm(actual_pivot-expected_pivot))
        result.append({'method':method, 'center_error':float(np.mean(errors)), 'rotation_error_rad':float(np.mean(orientation_errors)), 'joint_point_error':pivot_error, 'descendant_count':len(errors)})
    return result


def run_experiments4_mobility(manifest, output_dir):
    dest = _P4Path(output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    records=[]
    for i, entry in enumerate(manifest['objects'],1):
        path = dest/'objects'/f"{entry['category']}_{entry['object_id']}.json"
        if path.is_file():
            obj = json.loads(path.read_text(encoding='utf8'))
        else:
            prepared_path = dest/'prepared_cache'/f"{entry['category']}_{entry['object_id']}.json"
            obj = json.loads(prepared_path.read_text(encoding='utf8')) if prepared_path.is_file() else prepare_experiments4_asset(entry)
            path.parent.mkdir(parents=True,exist_ok=True)
            # Store the source motions and per-object metrics for diagnostic traceability.
            rows = []
            for evt in obj['events']:
                for score in evaluate_experiments4_event(evt):
                    rows.append({'object_id': entry['object_id'], 'category':entry['category'], 'joint_name':evt['joint_name'], 'kind':evt['kind'], 'parent_anchor_inside':evt['parent_anchor_inside'], 'child_anchor_inside':evt['child_anchor_inside'], 'delta':evt['delta'], **score})
            obj = {'object_id':entry['object_id'], 'category':entry['category'], 'rows': rows}
            path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
        records.extend(obj['rows'])
        if i%25==0 or i==len(manifest['objects']):
            print('Mobility kinematic objects:',i,'/',len(manifest['objects']))
    if not records:
        raise RuntimeError('No mobility results')
    output = {'protocol':manifest['protocol'], 'object_count':len(manifest['objects']), 'event_count':len(records)//4, 'records':records}
    (dest/'kinematics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    return output


def summarize_experiments4_mobility(kinematics, output_dir, bootstrap_samples=2000, seed=42):
    """Object-first aggregation: multiple joints in one object are NOT iid objects."""
    import pandas as pd
    df = pd.DataFrame(kinematics['records'])
    metrics = ('center_error','rotation_error_rad','joint_point_error')
    object_rows=[]
    for (oid,cat,kind,method),frame in df.groupby(['object_id','category','kind','method']):
        object_rows.append({'object_id':oid,'category':cat,'kind':kind,'method':method,'n_joints':len(frame),**{m:float(frame[m].mean()) for m in metrics},'both_anchors_inside_fraction':float((frame['parent_anchor_inside'] & frame['child_anchor_inside']).mean())})
    object_df=pd.DataFrame(object_rows)
    summaries=[]
    for (kind,method),frame in object_df.groupby(['kind','method']):
        summaries.append({'kind':kind,'method':method,'n_objects':len(frame),'n_joint_records':int(frame['n_joints'].sum()),**{m:float(frame[m].mean()) for m in metrics},'both_anchors_inside_fraction':float(frame['both_anchors_inside_fraction'].mean())})
    rng = np.random.default_rng(seed)
    paired=[]
    for kind in sorted(object_df['kind'].unique()):
        for comparator in ('local_center','local_pivot_aware','global_frozen'):
            for metric in ('joint_point_error','center_error'):
                left=object_df[(object_df.kind==kind)&(object_df.method=='anchor_joint')][['object_id','category',metric]].rename(columns={metric:'anchor'})
                right=object_df[(object_df.kind==kind)&(object_df.method==comparator)][['object_id',metric]].rename(columns={metric:'baseline'})
                pair=left.merge(right,on='object_id',validate='one_to_one')
                dif=(pair['baseline']-pair['anchor']).to_numpy()
                dif[np.abs(dif)<=1e-9]=0.
                if len(dif)==0: continue
                indices=[group.index.to_numpy() for _,group in pair.reset_index(drop=True).groupby('category')]
                boot=np.zeros(bootstrap_samples)
                for ix in indices:
                    d=dif[ix]
                    if len(d):
                        boot+=d[rng.integers(0,len(d),size=(bootstrap_samples,len(d)))].sum(axis=1)/len(dif)
                ci=np.quantile(boot,[.025,.975]) if bootstrap_samples else (float('nan'),float('nan'))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    p=1. if not np.any(dif) else float(wilcoxon(dif,zero_method='wilcox').pvalue)
                paired.append({'kind':kind,'baseline':comparator,'metric':metric,'n_objects':len(dif),'anchor_mean':float(pair['anchor'].mean()),'baseline_mean':float(pair['baseline'].mean()),'mean_improvement':float(dif.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'wilcoxon_p':p,'wins':int(np.sum(dif>0)),'ties':int(np.sum(dif==0)),'losses':int(np.sum(dif<0))})
    # Holm separately for each kind, preventing exploratory p-value inflation.
    for kind in sorted(set(x['kind'] for x in paired)):
        group=sorted([x for x in paired if x['kind']==kind],key=lambda x:x['wilcoxon_p'])
        running=0.
        for i,row in enumerate(group):
            running=max(running,min(1.,(len(group)-i)*row['wilcoxon_p']))
            row['holm_p']=running
    dest = _P4Path(output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    output={'protocol':{'benchmark':'PartNet-Mobility native URDF','input':'GT-conditioned link poses, joint axes/pivots/limits; NO LLM generation','normalization':'rest-pose object-longest-extent = 1; source URDF orientation retained','unit':'object, multiple joints averaged within object; paired by object','intervention':'one native joint per trial; all descendant visuals propagated rigidly','methods':{'anchor_joint':'explicit joint pivot/axis update','local_center':'hinge rotates child group about its own center; slider applies correct axis displacement','local_pivot_aware':'kinematically-equivalent strong control using pivot-aware local frame','global_frozen':'global poses held fixed when joint command changes'},'meaning':'joint-aware local can reproduce anchor exactly; no claim of inherent impossibility for baseline','limitations':['No LLM pivot prediction or natural-language joint inference','No dynamics, force/torque, physics contact, or collision-free trajectory','This is GT-conditioned native joint-kinematics capacity evaluation, NOT a PartNet static-generation comparison','Out-of-box anchors are retained (not clipped), separately quantified']},'object_count':kinematics['object_count'],'event_count':kinematics['event_count'],'summary':summaries,'paired_statistics':paired,'object_metrics':object_rows}
    (dest/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    pd.DataFrame(summaries).to_csv(dest/'summary.csv',index=False)
    pd.DataFrame(paired).to_csv(dest/'paired_statistics.csv',index=False)
    pd.DataFrame(object_rows).to_csv(dest/'object_metrics.csv',index=False)
    return output


def _p4_semantic_labels(folder):
    path = _P4Path(folder)/'semantics.txt'
    mapping={}
    if path.is_file():
        for row in path.read_text(encoding='utf8',errors='ignore').splitlines():
            items=row.strip().split()
            if len(items)>=3:
                mapping[items[0]]=' '.join(items[2:]).replace('_',' ')
    return mapping


def run_experiments4_llm_joint_prediction(manifest, output_dir, cfg, max_attempts=2, client=None):
    """One shared LLM per joint; NO GT pivot/axis is shown in the prompt."""
    dest=_P4Path(output_dir)
    result_path=dest/'llm_joint_predictions.json'
    existing={}
    if result_path.is_file():
        existing={x['event_id']:x for x in json.loads(result_path.read_text(encoding='utf8'))['events']}
    if client is None and cfg.allow_api:
        client=create_deepseek_client(base_url=cfg.base_url,timeout=cfg.timeout)
    rows=[]
    for item in manifest['objects']:
        prepared_path=dest/'prepared_cache'/f"{item['category']}_{item['object_id']}.json"
        prepared=json.loads(prepared_path.read_text(encoding='utf8')) if prepared_path.is_file() else prepare_experiments4_asset(item)
        semantics=_p4_semantic_labels(item['asset_dir'])
        for event in prepared['events']:
            event_id=f"{item['category']}|{item['object_id']}|{event['joint_name']}"
            if event_id in existing:
                rows.append(existing[event_id])
                continue
            if not cfg.allow_api:
                raise RuntimeError('LLM prediction cache incomplete and cfg.allow_api=False; enable API for Cell 7 or supply previous predictions')
            parent=event['rest_parent']
            child=event['initial'][event['child']]
            message={'category':item['category'],'parent_link':event['parent'],'parent_semantics':semantics.get(event['parent'],'unknown'),'child_link':event['child'],'child_semantics':semantics.get(event['child'],'unknown'),'joint_kind':event['kind'],'parent_box':parent,'child_box':child}
            prompt=("Estimate a physical articulation joint from a REST-POSE structured OBB model. "
                "Output a JSON object with two world-coordinate fields ONLY: "
                "pivot [x,y,z] (normalized object frame) and axis [x,y,z] (unit direction). "
                "For hinge, pivot is any point on the rotation axis. For slider, pivot is not scored, but supply [0,0,0]. "
                "The joint kind is given, but the axis/pivot is not. Do not return IDs, explanations, or extra fields. "
                "The sign convention of the ground-truth joint axis is NOT supplied, so axial sign ambiguity will be reported. "
                "Input: "+json.dumps(message,ensure_ascii=False))
            attempts=[]
            prediction=None
            for k in range(max_attempts):
                try:
                    reply=client.chat.completions.create(model=cfg.model,temperature=cfg.temperature,max_tokens=300,messages=[{'role':'user','content':prompt if k==0 else prompt+' Return strict JSON, previous response was malformed.'}])
                    raw=reply.choices[0].message.content.strip()
                    if raw.startswith('```'):
                        raw='\n'.join(raw.splitlines()[1:-1]).strip()
                    obj=json.loads(raw)
                    pivot=_p4_vec(obj['pivot']) if isinstance(obj['pivot'],str) else np.asarray(obj['pivot'],dtype=float)
                    axis=_p4_vec(obj['axis']) if isinstance(obj['axis'],str) else np.asarray(obj['axis'],dtype=float)
                    if pivot.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(pivot).all() or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:
                        raise ValueError('Invalid pivot/axis vector')
                    prediction={'pivot':pivot.tolist(),'axis':(axis/np.linalg.norm(axis)).tolist()}
                    attempts.append({'attempt':k+1,'valid':True})
                    break
                except (ValueError,KeyError,TypeError,IndexError,AttributeError) as error:
                    attempts.append({'attempt':k+1,'valid':False,'error':str(error)[:150]})
            record={'event_id':event_id,'object_id':item['object_id'],'category':item['category'],'joint_name':event['joint_name'],'kind':event['kind'],'status':'valid' if prediction else 'invalid','prediction':prediction,'attempts':attempts}
            rows.append(record)
            # One-event checkpoint: rerun consumes no API for existing entries.
            result_path.write_text(json.dumps({'model':cfg.model,'temperature':cfg.temperature,'events':rows},ensure_ascii=False,indent=2),encoding='utf8')
    output={'model':cfg.model,'temperature':cfg.temperature,'events':rows}
    result_path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    return output


def score_experiments4_llm_joint_prediction(manifest, predictions, output_dir):
    """Report measured GT relative pose error; sign of positive-angle axis is ambiguous."""
    dest=_P4Path(output_dir)
    prediction_index={row['event_id']:row for row in predictions['events']}
    scored=[]
    for item in manifest['objects']:
        prepared_path=dest/'prepared_cache'/f"{item['category']}_{item['object_id']}.json"
        prepared=json.loads(prepared_path.read_text(encoding='utf8')) if prepared_path.is_file() else prepare_experiments4_asset(item)
        for evt in prepared['events']:
            key=f"{item['category']}|{item['object_id']}|{evt['joint_name']}"
            pred=prediction_index.get(key)
            if pred is None or pred['status']!='valid':
                continue
            axis=np.asarray(pred['prediction']['axis'])
            pivot=np.asarray(pred['prediction']['pivot'])
            gt_axis=np.asarray(evt['axis_world'])
            gt_pivot=np.asarray(evt['pivot_normalized'])
            dot=float(np.clip(np.dot(axis,gt_axis),-1,1))
            line_error = float(np.linalg.norm(np.cross(pivot-gt_pivot,gt_axis))) if evt['kind']=='hinge' else None
            guess=copy.deepcopy(evt)
            guess['axis_world']=axis.tolist()
            guess['pivot_normalized']=pivot.tolist()
            guess['translation_normalized']=(axis*evt['delta']/evt['normalization_scale']).tolist() if evt['kind']=='slider' else [0,0,0]
            models=evaluate_experiments4_event(guess)
            for row in models:
                scored.append({'event_id':key,'object_id':item['object_id'],'category':item['category'],'kind':evt['kind'],'method':row['method'],'axis_cosine_signed':dot,'axis_cosine_unsigned':abs(dot),'hinge_axis_line_error':line_error,**row})
    df=__import__('pandas').DataFrame(scored)
    if df.empty:
        raise RuntimeError('No valid LLM predictions for any joint event')
    obj=df.groupby(['object_id','category','kind','method'],as_index=False).agg({'center_error':'mean','rotation_error_rad':'mean','joint_point_error':'mean','axis_cosine_signed':'mean','axis_cosine_unsigned':'mean','hinge_axis_line_error':'mean'})
    summary=obj.groupby(['kind','method'],as_index=False).agg({'center_error':'mean','rotation_error_rad':'mean','joint_point_error':'mean','axis_cosine_signed':'mean','axis_cosine_unsigned':'mean','hinge_axis_line_error':'mean'})
    output={'protocol':{'input':'semantic labels + rest OBB, joint kind; GT pivot and axis hidden','evaluation':'GT URDF q-command pose and pivot/axis consistency','caveats':['Axis sign is not identifiable from unlabeled positive/negative joint convention','Rest OBBs and semantics often do not uniquely determine the hinge line','All methods receive the same inferred pivot/axis; local_pivot_aware is equivalent to Anchor when joint parameters are equal','No claim of general text-to-3D generative success']},'predicted_joint_events':len(set(r['event_id'] for r in scored)),'summary':summary.to_dict('records'),'object_metrics':obj.to_dict('records')}
    (dest/'llm_pose_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    summary.to_csv(dest/'llm_pose_summary.csv',index=False)
    return output
# =============================================================================
# Experiments 5: Formal Open-world Generalization
# Requires the cleaned PartNet/Editing methods.py from Experiments 2-4.
# No PartNet GT is used in this module.
# =============================================================================

OPENWORLD_EVALUATION_VERSION = "openworld-obb-eval-v1"
OPENWORLD_SHARED_VERSION = "openworld-free-decomposition-v1"
OPENWORLD_METRIC_DEFINITIONS = {
    "shared_edge_obb_sat_deviation": "Mean absolute signed OBB SAT margin on the shared inferred tree edges; lower is better and penalizes both gap and penetration.",
    "shared_edge_obb_sat_gap": "Mean positive OBB SAT margin on shared inferred edges; lower is better.",
    "shared_edge_obb_sat_penetration": "Mean positive penetration depth proxy max(-SAT margin,0) on shared inferred edges; lower is better.",
    "obb_overlap_pair_fraction": "Fraction of all unordered predicted OBB pairs with strict SAT overlap; lower is better as a collision proxy, not exact mesh collision.",
    "part_volume_sum": "Sum of normalized local OBB volumes; descriptive compactness quantity, not a universal quality score."
}
OPENWORLD_PRIMARY_METRICS = (
    "shared_edge_obb_sat_deviation",
    "shared_edge_obb_sat_gap",
    "obb_overlap_pair_fraction",
)


def load_openworld_inputs(source_path: str | Path, cfg: PartNetConfig) -> dict:
    """Freeze only object_id/category/description from a legacy open-world JSON.

    Old part_spec / anchor_tree / global_tree / local_tree fields are intentionally ignored.
    Supported top-level shapes: list, or dict containing objects/data/samples/items/results.
    """
    path = Path(source_path)
    if not path.is_absolute():
        path = cfg.project_path(str(path))
    data = load_json(path)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = None
        for key in ("objects", "data", "samples", "items", "results"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
        if rows is None and data and all(isinstance(v, dict) and "description" in v for v in data.values()):
            rows = list(data.values())
        if rows is None:
            raise ValueError("Open-world source must be a list or contain objects/data/samples/items/results list")
    else:
        raise ValueError("Unsupported open-world JSON top level")

    objects = []
    seen_ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} is not an object")
        description = row.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Row {index} missing non-empty description")
        object_id = str(row.get("object_id", row.get("id", f"ow_{index:04d}")))
        if object_id in seen_ids:
            raise ValueError(f"Duplicate open-world object_id: {object_id}")
        seen_ids.add(object_id)
        category = row.get("category", row.get("group", row.get("type", "unknown")))
        category = str(category) if category is not None else "unknown"
        objects.append({"object_id": object_id, "category": category, "description": description.strip()})

    if cfg.object_limit is not None:
        objects = objects[: int(cfg.object_limit)]
    if not objects:
        raise ValueError("Open-world source contains no objects")

    frozen = {
        "protocol": "free natural-language descriptions only; legacy structures/generations ignored",
        "source_file": str(path),
        "object_count": len(objects),
        "objects": objects,
    }
    manifest_path = cfg.output_path("openworld_manifest.json")
    if manifest_path.exists():
        old = load_json(manifest_path)
        if old.get("objects") != objects:
            raise ValueError("Open-world manifest differs from the frozen output. Use a new output_dir for a changed dataset.")
    else:
        save_json(frozen, manifest_path)
    print(f"Open-world descriptions: {len(objects)} objects; frozen manifest={manifest_path}")
    return frozen


def build_openworld_shared_messages(description: str) -> list:
    system = """
You infer a shared structured physical-part specification from ONE free-form natural-language object description.
This shared specification will be used unchanged by several 3D representations, so do the semantic decomposition only once.

Return ONLY one JSON object with this exact schema:
{
  "object_category": "short category name",
  "root_part_id": "part_0",
  "parts": [
    {"part_id":"part_0", "name":"singular physical part name", "instance_index":0}
  ],
  "connections": [
    {"parent_part_id":"part_0", "child_part_id":"part_1"}
  ]
}

Rules:
- Include meaningful geometry-bearing physical components needed to realize the described object.
- Do NOT create pure semantic/group nodes such as "chair back group", "base assembly", "screen side", or "handle side" unless the text clearly describes that item itself as a physical geometry-bearing component.
- Do not invent screws, bolts, decorations, or hidden internals unless explicitly required by the description.
- Preserve explicit multiplicity. If the description says four legs, create four separate leg instances.
- Repeated instances use the SAME singular name and consecutive instance_index values starting from 0.
- Assign part_id values exactly part_0, part_1, ... with no gaps or duplicates.
- Use concise human-readable names. Do not add geometry numbers, coordinates, dimensions, rotations, anchors, or bounding boxes.
- Choose exactly ONE geometry-bearing part as root.
- connections must form one directed rooted tree over all parts: N parts -> exactly N-1 edges; root has no parent; every other part has exactly one parent; connected and acyclic.
- Use parent-child edges to represent plausible direct physical support/attachment/dependency, not merely semantic similarity.
- Prefer the structurally central/supporting physical part as parent when reasonable.
- Keep the decomposition practical: normally 2-30 physical parts.

Return JSON only. No markdown and no explanation.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": description.strip()}]


def validate_openworld_shared_spec(spec: dict) -> dict:
    base = validate_shared_part_spec(spec)
    errors = list(base.get("errors", []))
    if not isinstance(spec, dict):
        return {"valid": False, "errors": errors or ["spec_not_dict"]}
    category = spec.get("object_category")
    if not isinstance(category, str) or not category.strip():
        errors.append("missing_object_category")
    parts = spec.get("parts", [])
    if not (2 <= len(parts) <= 30):
        errors.append("part_count_must_be_between_2_and_30")
    for part in parts:
        if not isinstance(part, dict):
            continue
        name = part.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{part.get('part_id','?')}:missing_part_name")
        if set(part) != {"part_id", "name", "instance_index"}:
            errors.append(f"{part.get('part_id','?')}:part_fields_must_be_exact")
        if type(part.get("instance_index")) is not int or part.get("instance_index", -1) < 0:
            errors.append(f"{part.get('part_id','?')}:invalid_instance_index")
    return {"valid": not errors, "errors": errors}


def _openworld_shared_fingerprint(row: dict, cfg: PartNetConfig) -> str:
    return content_hash({
        "schema": OPENWORLD_SHARED_VERSION,
        "messages": build_openworld_shared_messages(row["description"]),
        "request": cfg.request_config(),
        "max_attempts": cfg.max_attempts,
    })


def run_openworld_shared_specs(inputs: dict, cfg: PartNetConfig, client=None) -> dict:
    jobs = []
    for row in inputs["objects"]:
        fingerprint = _openworld_shared_fingerprint(row, cfg)
        cached = _read_cache(cfg.output_path(f"shared/{row['object_id']}.json"), fingerprint)
        if cached is None or cached.get("status") not in ("valid", "exhausted"):
            jobs.append(row)
    if jobs and cfg.allow_api:
        client = client or create_deepseek_client(base_url=cfg.base_url, timeout=cfg.timeout)
        def worker(row):
            path = cfg.output_path(f"shared/{row['object_id']}.json")
            return _generate_valid_json(
                client,
                cfg,
                path,
                build_openworld_shared_messages(row["description"]),
                validate_openworld_shared_spec,
                _openworld_shared_fingerprint(row, cfg),
            )
        _run_jobs(jobs, worker, cfg, "Open-world shared decomposition")

    results = {}
    for row in inputs["objects"]:
        record = _read_cache(cfg.output_path(f"shared/{row['object_id']}.json"), _openworld_shared_fingerprint(row, cfg))
        if record is None:
            results[row["object_id"]] = {**row, "status": "missing", "spec": None, "attempt_count": 0, "first_attempt_valid": False}
            continue
        spec = record.get("result") if record.get("status") == "valid" else None
        results[row["object_id"]] = {
            **row,
            "status": record.get("status", "missing"),
            "spec": spec,
            "attempt_count": int(record.get("attempt_count", 0)),
            "first_attempt_valid": bool(record.get("attempts") and record["attempts"][0].get("validation", {}).get("valid")),
        }
    save_json({"protocol": OPENWORLD_SHARED_VERSION, "objects": list(results.values())}, cfg.output_path("shared_specs.json"))
    print(f"Open-world shared specs: valid={sum(r['status']=='valid' for r in results.values())}/{len(results)}")
    return results


def run_openworld_generation(shared: dict, cfg: PartNetConfig, client=None) -> dict:
    """Reuse the current fixed-part representation prompts/validators with a separate output_dir."""
    return run_partnet_generation(shared, cfg, client=client)


def _openworld_tree_depth(spec: dict) -> int:
    children = defaultdict(list)
    for edge in spec["connections"]:
        children[edge["parent_part_id"]].append(edge["child_part_id"])
    def depth(pid):
        return 0 if not children[pid] else 1 + max(depth(child) for child in children[pid])
    return int(depth(spec["root_part_id"]))


def evaluate_openworld_layout(layout: dict, spec: dict, zero_tolerance: float = 1e-9) -> dict:
    parts = layout["parts"]
    pindex = {p["part_id"]: p for p in parts}
    if set(pindex) != {p["part_id"] for p in spec["parts"]}:
        raise ValueError("Layout part IDs differ from shared specification")
    pair_margins = [obb_sat_margin(a, b) for a, b in itertools.combinations(parts, 2)]
    shared_margins = [obb_sat_margin(pindex[e["parent_part_id"]], pindex[e["child_part_id"]]) for e in spec["connections"]]
    metrics = {
        "shared_edge_obb_sat_deviation": _mean_or_none(np.abs(shared_margins)),
        "shared_edge_obb_sat_gap": _mean_or_none(np.maximum(shared_margins, 0)),
        "shared_edge_obb_sat_penetration": _mean_or_none(np.maximum(-np.asarray(shared_margins), 0)),
        "obb_overlap_pair_fraction": _mean_or_none(np.asarray(pair_margins) < -zero_tolerance),
        "part_volume_sum": float(sum(np.prod(p["bbox_size"]) for p in parts)),
    }
    return {
        "metrics": metrics,
        "part_count": len(parts),
        "shared_edge_count": len(spec["connections"]),
        "tree_depth": _openworld_tree_depth(spec),
    }


def run_openworld_evaluation(inputs: dict, shared: dict, generation: dict, cfg: PartNetConfig) -> dict:
    records, failures = [], []
    for representation in REPRESENTATIONS:
        for item in inputs["objects"]:
            oid = item["object_id"]
            if shared[oid]["status"] != "valid":
                continue
            generated = generation[representation].get(oid, {})
            if generated.get("status") != "valid":
                continue
            fingerprint = content_hash({
                "evaluation": OPENWORLD_EVALUATION_VERSION,
                "generation": generated["fingerprint"],
                "spec": shared[oid]["spec"],
                "zero_tolerance": cfg.zero_tolerance,
            })
            path = cfg.output_path(f"evaluation/{representation}/{oid}.json")
            try:
                cached = _read_cache(path, fingerprint)
                if cached is None:
                    spec = shared[oid]["spec"]
                    layout = transform_partnet_representation(generated["result"], spec, representation)
                    evaluation = evaluate_openworld_layout(layout, spec, cfg.zero_tolerance)
                    cached = {
                        "fingerprint": fingerprint,
                        "object_id": oid,
                        "category": item["category"],
                        "method": representation,
                        "normalization": layout["normalization"],
                        "layout_parts": layout["parts"],
                        **evaluation,
                    }
                    save_json(cached, path)
                records.append(cached)
            except (ValueError, KeyError, TypeError, FloatingPointError) as error:
                failures.append({"object_id": oid, "method": representation, "error": _safe_error(error)})
    complete = {row["object_id"] for row in inputs["objects"] if shared[row["object_id"]]["status"] == "valid"}
    for method in REPRESENTATIONS:
        complete &= {r["object_id"] for r in records if r["method"] == method}
    ordered = [row["object_id"] for row in inputs["objects"] if row["object_id"] in complete]
    result = {
        "version": OPENWORLD_EVALUATION_VERSION,
        "objects": records,
        "complete_ids": ordered,
        "failures": failures,
        "metric_definitions": OPENWORLD_METRIC_DEFINITIONS,
    }
    save_json(result, cfg.output_path("evaluation_all.json"))
    print(f"Open-world static evaluation: complete={len(ordered)}/{len(inputs['objects'])}; evaluation failures={len(failures)}")
    return result


def run_openworld_random_anchor(inputs: dict, shared: dict, generation: dict, evaluation: dict, cfg: PartNetConfig) -> dict:
    records = []
    bases = {r["object_id"]: r for r in evaluation["objects"] if r["method"] == "anchor"}
    for item in inputs["objects"]:
        oid = item["object_id"]
        if oid not in bases or oid not in set(evaluation["complete_ids"]):
            continue
        base = bases[oid]
        fingerprint = content_hash({
            "base": base["fingerprint"],
            "random_anchor": "uniform-face-v1",
            "seeds": cfg.random_anchor_seeds,
            "normalization": "fixed_to_base",
        })
        path = cfg.output_path(f"random_anchor/{oid}.json")
        cached = _read_cache(path, fingerprint)
        if cached is None:
            spec = shared[oid]["spec"]
            raw = generation["anchor"][oid]["result"]
            seed_results = []
            for seed in cfg.random_anchor_seeds:
                modified = randomize_partnet_anchors(raw, spec, seed)
                layout = transform_partnet_representation(modified, spec, "anchor", base["normalization"])
                result = evaluate_openworld_layout(layout, spec, cfg.zero_tolerance)
                seed_results.append({"seed": seed, "metrics": result["metrics"]})
            metrics = {key: _mean_or_none([r["metrics"][key] for r in seed_results if r["metrics"][key] is not None]) for key in OPENWORLD_METRIC_DEFINITIONS}
            cached = {
                "fingerprint": fingerprint,
                "object_id": oid,
                "category": item["category"],
                "method": "random_anchor",
                "metrics": metrics,
                "seed_results": seed_results,
                "normalization_policy": "base_anchor_fixed",
            }
            save_json(cached, path)
        records.append(cached)
    result = {"objects": records, "seeds": list(cfg.random_anchor_seeds), "aggregation": "seed mean within object; objects are statistical units"}
    save_json(result, cfg.output_path("random_anchor_all.json"))
    print(f"Open-world Random Anchor: {len(records)} objects x {len(cfg.random_anchor_seeds)} seeds")
    return result


def export_openworld_results(inputs: dict, shared: dict, generation: dict, evaluation: dict, random_anchor: dict, perturbation: dict, cfg: PartNetConfig) -> dict:
    complete = set(evaluation["complete_ids"])
    paired_static = [r for r in evaluation["objects"] if r["object_id"] in complete]
    paired_static += [r for r in random_anchor["objects"] if r["object_id"] in complete]
    summary = _summaries(paired_static, ("method",), tuple(OPENWORLD_METRIC_DEFINITIONS))
    category_summary = _summaries(paired_static, ("category", "method"), tuple(OPENWORLD_METRIC_DEFINITIONS))

    lookup = {method: {r["object_id"]: r for r in paired_static if r["method"] == method} for method in (*REPRESENTATIONS, "random_anchor")}
    static_stats = []
    for baseline in ("global", "local", "random_anchor"):
        for metric in OPENWORLD_PRIMARY_METRICS:
            row = paired_object_statistics(lookup["anchor"], lookup[baseline], metric, cfg)
            static_stats.append({"experiment": "random_anchor" if baseline == "random_anchor" else "static", "baseline": baseline, "metric": metric, "scale": None, **row})
    _holm_adjust(static_stats)

    perturb_rows = [r for r in perturbation["objects"] if r["object_id"] in complete]
    perturb_summary = _summaries(perturb_rows, ("method", "scale"), ("contact_deviation", "baseline_contact_deviation", "margin_drift", "deviation_change"))
    perturb_stats = []
    for scale in cfg.scale_factors:
        by_method = {m: {r["object_id"]: r for r in perturb_rows if r["method"] == m and r["scale"] == scale} for m in REPRESENTATIONS}
        for baseline in ("global", "local"):
            for metric in ("contact_deviation", "margin_drift", "deviation_change"):
                result = paired_object_statistics(by_method["anchor"], by_method[baseline], metric, cfg)
                perturb_stats.append({"experiment": "scale_perturbation", "baseline": baseline, "metric": metric, "scale": scale, **result})
    _holm_adjust(perturb_stats)

    statuses = []
    for item in inputs["objects"]:
        oid = item["object_id"]
        row = {
            "object_id": oid,
            "category": item["category"],
            "shared_status": shared[oid]["status"],
            "shared_attempts": shared[oid].get("attempt_count", 0),
            "shared_first_attempt_valid": shared[oid].get("first_attempt_valid", False),
            "complete_case": oid in complete,
        }
        for method in REPRESENTATIONS:
            generated = generation[method].get(oid, {})
            row[method + "_status"] = generated.get("status", "blocked_by_shared")
            row[method + "_attempts"] = generated.get("attempt_count", 0)
            row[method + "_first_attempt_valid"] = bool(generated.get("attempts") and generated["attempts"][0].get("validation", {}).get("valid"))
        statuses.append(row)

    structure_rows = []
    for oid, row in shared.items():
        if row["status"] != "valid":
            continue
        structure_rows.append({
            "object_id": oid,
            "category": row["category"],
            "part_count": len(row["spec"]["parts"]),
            "edge_count": len(row["spec"]["connections"]),
            "tree_depth": _openworld_tree_depth(row["spec"]),
        })

    object_metrics = []
    for r in paired_static:
        object_metrics.append({"object_id": r["object_id"], "category": r["category"], "method": r["method"], **r["metrics"]})

    tables = {
        "summary": summary,
        "category_summary": category_summary,
        "perturbation_summary": perturb_summary,
        "paired_statistics": static_stats + perturb_stats,
        "generation_status": statuses,
        "structure_summary": structure_rows,
        "object_metrics": object_metrics,
    }
    for name, rows in tables.items():
        _write_csv(rows, cfg.output_path(f"tables/{name}.csv"))
    save_json(tables, cfg.output_path("tables/results.json"))
    protocol = {
        "setting": "formal open-world free natural language; no GT part inventory",
        "shared_decomposition": "one LLM decomposition reused by Anchor/Global/Hierarchy Local",
        "representation_generation": "same model/config and fixed shared part IDs/connections",
        "main_sample_policy": "complete cases shared across Anchor/Global/Local; failures reported separately",
        "static_metrics": OPENWORLD_METRIC_DEFINITIONS,
        "random_anchor": "same Anchor raw parameters except randomized surface anchors; 5-seed object mean by default",
        "perturbation": "all shared edges; one child bbox scaled per intervention; base normalization fixed",
        "statistics": "paired object differences; category-stratified bootstrap; numerical-zero-consistent Wilcoxon; Holm correction within static/ablation and perturbation families",
        "important_limitation": "No GT 3D instance is available. Structural proxy metrics do not measure semantic/text fidelity; human evaluation is a separate next step.",
        "planned_objects": len(inputs["objects"]),
        "shared_valid": sum(r["status"] == "valid" for r in shared.values()),
        "complete_object_count": len(complete),
    }
    save_json(protocol, cfg.output_path("evaluation_protocol.json"))
    print(f"Open-world results exported: complete={len(complete)}/{len(inputs['objects'])}")
    return tables


def openworld_summary_view(tables: dict, kind: str = "summary"):
    import pandas as pd
    frame = pd.DataFrame(tables[kind])
    if frame.empty:
        return frame
    if kind == "summary":
        return frame.pivot(index="method", columns="metric", values="mean").round(6)
    if kind == "perturbation_summary":
        return frame.pivot(index=["scale", "method"], columns="metric", values="mean").round(6)
    return frame

# =============================================================================
# Experiments 6: Second-LLM Backbone Robustness
# Reuses frozen shared structures from Experiments 2 and 5.
# Only Anchor / Global / Hierarchy Local geometry generation changes backbone.
# =============================================================================

EXPERIMENTS6_VERSION = "second-backbone-v1"
EXPERIMENTS6_STATIC_METRICS = (
    "shared_edge_obb_sat_deviation",
    "obb_overlap_pair_fraction",
)
EXPERIMENTS6_PERTURB_METRICS = ("margin_drift",)


@dataclass
class Experiments6Config:
    project_dir: str = "."
    output_dir: str = "experiments6_results"
    primary_partnet_dir: str = r"partnet_results\experiments2"
    primary_openworld_dir: str = "openworld_results"
    secondary_label: str = "qwen3.7-plus"
    secondary_model: str = "qwen3.7-plus"
    secondary_base_url: str = ""
    secondary_api_key_env: str = "DASHSCOPE_API_KEY"
    secondary_extra_body: dict = field(default_factory=lambda: {"enable_thinking": False})
    temperature: float = 0.0
    max_tokens: int = 8192
    max_attempts: int = 3
    workers: int = 4
    timeout: float = 180.0
    transport_retries: int = 2
    partnet_per_category: int = 11
    subset_seed: int = 606
    scale_factors: tuple = (0.8, 1.2, 1.5)
    bootstrap_samples: int = 10000
    statistics_seed: int = 42
    show_every: int = 20
    allow_api: bool = True

    def project_path(self, name: str | Path) -> Path:
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = Path(self.project_dir).expanduser().resolve() / path
        return path.resolve()

    def output_path(self, name: str | Path = "") -> Path:
        base = self.project_path(self.output_dir)
        return base / name


def create_openai_compatible_client(
    api_key_env: str,
    base_url: str,
    timeout: float = 180.0,
):
    """Create a generic OpenAI-compatible client without storing the key in artifacts."""
    from openai import OpenAI
    from getpass import getpass

    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError(
            "secondary_base_url 不能为空。请填写第二模型服务商的 OpenAI-compatible base_url。"
        )

    key = os.environ.get(api_key_env)
    if not key:
        key = getpass(f"{api_key_env}（隐藏输入，仅保存在当前进程）：").strip()
        if key:
            os.environ[api_key_env] = key
    if not key:
        raise ValueError(f"缺少环境变量/API Key: {api_key_env}")

    return OpenAI(
        api_key=key,
        base_url=base_url.strip(),
        timeout=timeout,
        max_retries=0,
    )


def _experiments6_load_primary_dataset(source_dir: Path, dataset: str) -> dict:
    required = (
        "shared_specs.json",
        "evaluation_all.json",
        "perturbation_all.json",
    )
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{dataset}: source_dir={source_dir} 缺少 {missing}"
        )

    shared_data = load_json(source_dir / "shared_specs.json")
    evaluation = load_json(source_dir / "evaluation_all.json")
    perturbation = load_json(source_dir / "perturbation_all.json")

    if not isinstance(shared_data, dict) or not isinstance(shared_data.get("objects"), list):
        raise ValueError(f"{dataset}: shared_specs.json 缺少 objects list")

    shared_index = {}
    order = []
    for row in shared_data["objects"]:
        oid = str(row["object_id"])
        if oid in shared_index:
            raise ValueError(f"{dataset}: duplicate object_id={oid}")
        shared_index[oid] = row
        order.append(oid)

    complete_ids = [str(x) for x in evaluation.get("complete_ids", [])]
    complete_set = set(complete_ids)
    if not complete_ids:
        raise ValueError(f"{dataset}: primary evaluation has no complete_ids")

    for oid in complete_ids:
        row = shared_index.get(oid)
        if row is None or row.get("status") != "valid" or not isinstance(row.get("spec"), dict):
            raise ValueError(f"{dataset}: primary complete object lacks valid shared spec: {oid}")

    return {
        "dataset": dataset,
        "source_dir": str(source_dir),
        "shared_index": shared_index,
        "shared_order": order,
        "primary_complete_ids": complete_ids,
        "primary_complete_set": complete_set,
        "primary_evaluation": evaluation,
        "primary_perturbation": perturbation,
    }


def build_experiments6_manifest(cfg: Experiments6Config) -> dict:
    """Freeze the exact cross-backbone evaluation cohort.

    PartNet:
        deterministic balanced sample,
        `partnet_per_category` per category.

    OpenWorld:
        every Experiments-5 primary complete case.

    API transport settings such as base_url are deliberately excluded
    from the scientific manifest because they do not change the cohort
    or generation protocol.
    """

    partnet_source = _experiments6_load_primary_dataset(
        cfg.project_path(cfg.primary_partnet_dir),
        "partnet",
    )

    openworld_source = _experiments6_load_primary_dataset(
        cfg.project_path(cfg.primary_openworld_dir),
        "openworld",
    )

    if cfg.partnet_per_category <= 0:
        raise ValueError(
            "partnet_per_category must be positive"
        )

    by_category = defaultdict(list)

    for oid in partnet_source["primary_complete_ids"]:
        row = partnet_source["shared_index"][oid]

        by_category[
            str(row["category"])
        ].append(oid)

    rng = np.random.default_rng(
        cfg.subset_seed
    )

    selected_set = set()

    category_counts = {}

    for category in sorted(by_category):
        ids = sorted(
            by_category[category]
        )

        if len(ids) < cfg.partnet_per_category:
            raise ValueError(
                f"PartNet category {category} "
                f"only has {len(ids)} complete cases; "
                f"requested {cfg.partnet_per_category}"
            )

        chosen_indices = rng.choice(
            len(ids),
            size=cfg.partnet_per_category,
            replace=False,
        )

        chosen = {
            ids[int(i)]
            for i in chosen_indices
        }

        selected_set.update(
            chosen
        )

        category_counts[
            category
        ] = len(chosen)

    partnet_ids = [
        oid
        for oid in partnet_source["shared_order"]
        if oid in selected_set
    ]

    openworld_ids = [
        oid
        for oid in openworld_source["shared_order"]
        if oid
        in openworld_source["primary_complete_set"]
    ]

    def object_rows(source, ids):
        rows = []

        for oid in ids:
            row = source[
                "shared_index"
            ][oid]

            rows.append({
                "object_id":
                    oid,
                "category":
                    row["category"],
                "description":
                    row["description"],
                "spec_hash":
                    content_hash(
                        row["spec"]
                    ),
                "part_count":
                    len(
                        row["spec"]["parts"]
                    ),
                "edge_count":
                    len(
                        row["spec"][
                            "connections"
                        ]
                    ),
            })

        return rows

    manifest = {
        "version":
            EXPERIMENTS6_VERSION,

        "protocol": (
            "Frozen shared specs from primary "
            "DeepSeek experiments; only "
            "representation geometry-generation "
            "backbone changes."
        ),

        "secondary": {
            "label":
                cfg.secondary_label,

            "model":
                cfg.secondary_model,

            "temperature":
                cfg.temperature,

            "max_tokens":
                cfg.max_tokens,

            "max_attempts":
                cfg.max_attempts,

            "extra_body":
                copy.deepcopy(
                    cfg.secondary_extra_body
                ),
        },

        "selection": {
            "subset_seed":
                cfg.subset_seed,

            "partnet_per_category":
                cfg.partnet_per_category,

            "partnet_count":
                len(partnet_ids),

            "partnet_category_counts":
                category_counts,

            "openworld_count":
                len(openworld_ids),

            "openworld_policy":
                (
                    "all Experiments-5 "
                    "primary complete cases"
                ),
        },

        "partnet_objects":
            object_rows(
                partnet_source,
                partnet_ids
            ),

        "openworld_objects":
            object_rows(
                openworld_source,
                openworld_ids
            ),
    }

    path = cfg.output_path(
        "experiments6_manifest.json"
    )

    if path.is_file():
        old = load_json(
            path
        )

        old_compare = copy.deepcopy(
            old
        )

        new_compare = copy.deepcopy(
            manifest
        )

        old_compare.setdefault(
            "secondary",
            {}
        ).pop(
            "base_url",
            None
        )

        new_compare.setdefault(
            "secondary",
            {}
        ).pop(
            "base_url",
            None
        )

        if old_compare != new_compare:
            raise ValueError(
                "Experiments 6 scientific "
                "manifest/config changed. "
                "Use a new output_dir instead "
                "of overwriting the frozen cohort."
            )

        if old != manifest:
            save_json(
                manifest,
                path
            )

    else:
        save_json(
            manifest,
            path
        )

    print(
        "Experiments 6 manifest:",
        f"PartNet={len(partnet_ids)},",
        f"OpenWorld={len(openworld_ids)},",
        f"secondary={cfg.secondary_model}",
    )

    return manifest


def prepare_experiments6_sources(cfg: Experiments6Config, manifest: dict | None = None) -> dict:
    manifest = manifest or build_experiments6_manifest(cfg)
    sources = {}

    source_specs = {
        "partnet": (
            cfg.project_path(cfg.primary_partnet_dir),
            [row["object_id"] for row in manifest["partnet_objects"]],
        ),
        "openworld": (
            cfg.project_path(cfg.primary_openworld_dir),
            [row["object_id"] for row in manifest["openworld_objects"]],
        ),
    }

    for dataset, (source_dir, ids) in source_specs.items():
        primary = _experiments6_load_primary_dataset(source_dir, dataset)
        id_set = set(ids)
        ordered_ids = [oid for oid in primary["shared_order"] if oid in id_set]
        if ordered_ids != ids:
            raise ValueError(f"{dataset}: manifest object order/source mismatch")

        shared = {}
        objects = []
        for oid in ids:
            row = copy.deepcopy(primary["shared_index"][oid])
            if row.get("status") != "valid":
                raise ValueError(f"{dataset}: selected shared row is not valid: {oid}")
            shared[oid] = row
            objects.append({
                "object_id": oid,
                "category": row["category"],
                "description": row["description"],
            })

        sources[dataset] = {
            "inputs": {
                "protocol": "Experiments-6 frozen subset",
                "object_count": len(objects),
                "objects": objects,
            },
            "shared": shared,
            "primary_evaluation": primary["primary_evaluation"],
            "primary_perturbation": primary["primary_perturbation"],
            "source_dir": str(source_dir),
        }

    return sources


def make_experiments6_generation_cfg(
    cfg: Experiments6Config,
    dataset: str,
) -> PartNetConfig:
    if dataset not in ("partnet", "openworld"):
        raise ValueError(f"Unknown dataset: {dataset}")
    run = PartNetConfig()
    run.project_dir = cfg.project_dir
    run.output_dir = str(cfg.output_path(dataset))
    run.model = cfg.secondary_model
    run.temperature = cfg.temperature
    run.max_tokens = cfg.max_tokens
    run.max_attempts = cfg.max_attempts
    run.workers = cfg.workers
    run.timeout = cfg.timeout
    run.transport_retries = cfg.transport_retries
    run.base_url = cfg.secondary_base_url
    run.extra_body = copy.deepcopy(cfg.secondary_extra_body)
    run.zero_tolerance = 1e-9
    run.scale_factors = tuple(cfg.scale_factors)
    run.bootstrap_samples = cfg.bootstrap_samples
    run.statistics_seed = cfg.statistics_seed
    run.expected_objects = None
    run.object_limit = None
    run.show_every = cfg.show_every
    run.allow_api = cfg.allow_api
    return run


def run_experiments6_generation(
    sources: dict,
    cfg: Experiments6Config,
    client=None,
) -> dict:
    if client is None and cfg.allow_api:
        client = create_openai_compatible_client(
            api_key_env=cfg.secondary_api_key_env,
            base_url=cfg.secondary_base_url,
            timeout=cfg.timeout,
        )

    generations = {}
    runtime = {
        "version": EXPERIMENTS6_VERSION,
        "requested_model": cfg.secondary_model,
        "secondary_label": cfg.secondary_label,
        "base_url": cfg.secondary_base_url,
        "datasets": {},
    }

    for dataset in ("partnet", "openworld"):
        run_cfg = make_experiments6_generation_cfg(cfg, dataset)
        generations[dataset] = run_partnet_generation(
            sources[dataset]["shared"],
            run_cfg,
            client=client,
        )

        returned_models = Counter()
        status_counts = {}
        for method in REPRESENTATIONS:
            rows = generations[dataset][method]
            status_counts[method] = Counter(
                row.get("status", "missing")
                for row in rows.values()
            )
            for row in rows.values():
                for attempt in row.get("attempts", []):
                    model = attempt.get("returned_model")
                    if model:
                        returned_models[str(model)] += 1

        runtime["datasets"][dataset] = {
            "status_counts": {
                method: dict(counts)
                for method, counts in status_counts.items()
            },
            "returned_model_counts": dict(returned_models),
        }

    save_json(runtime, cfg.output_path("runtime_metadata.json"))
    return generations


def run_experiments6_evaluation(
    sources: dict,
    generations: dict,
    cfg: Experiments6Config,
) -> dict:
    evaluations = {}
    for dataset in ("partnet", "openworld"):
        run_cfg = make_experiments6_generation_cfg(cfg, dataset)
        run_cfg.allow_api = False
        evaluations[dataset] = run_openworld_evaluation(
            sources[dataset]["inputs"],
            sources[dataset]["shared"],
            generations[dataset],
            run_cfg,
        )
    return evaluations


def run_experiments6_perturbation(
    sources: dict,
    generations: dict,
    evaluations: dict,
    cfg: Experiments6Config,
) -> dict:
    outputs = {}
    for dataset in ("partnet", "openworld"):
        run_cfg = make_experiments6_generation_cfg(cfg, dataset)
        run_cfg.allow_api = False
        outputs[dataset] = run_partnet_scale_perturbation(
            sources[dataset]["inputs"],
            sources[dataset]["shared"],
            generations[dataset],
            evaluations[dataset],
            run_cfg,
        )
    return outputs


def _experiments6_category_direction(
    rows: list,
    metric: str,
    baseline: str,
    scale: float | None = None,
) -> dict:
    filtered = rows
    if scale is not None:
        filtered = [r for r in filtered if float(r["scale"]) == float(scale)]

    by_cat_method = defaultdict(list)
    for row in filtered:
        value = row["metrics"].get(metric)
        if value is None or not np.isfinite(value):
            continue
        by_cat_method[(row["category"], row["method"])].append(float(value))

    categories = sorted({cat for cat, _ in by_cat_method})
    wins = ties = losses = 0
    detail = []
    for category in categories:
        a = by_cat_method.get((category, "anchor"), [])
        b = by_cat_method.get((category, baseline), [])
        if not a or not b:
            continue
        am = float(np.mean(a))
        bm = float(np.mean(b))
        difference = bm - am
        if difference > 1e-9:
            wins += 1
        elif difference < -1e-9:
            losses += 1
        else:
            ties += 1
        detail.append({
            "category": category,
            "anchor_mean": am,
            "baseline_mean": bm,
            "improvement": difference,
        })
    return {
        "categories": len(detail),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "detail": detail,
    }


def summarize_experiments6_results(
    sources: dict,
    generations: dict,
    evaluations: dict,
    perturbations: dict,
    cfg: Experiments6Config,
) -> dict:
    """Compare primary DeepSeek vs secondary backbone on the identical secondary-complete cohort."""
    summary_rows = []
    paired_rows = []
    category_rows = []
    generation_rows = []
    concordance_rows = []

    for dataset in ("partnet", "openworld"):
        planned_ids = [r["object_id"] for r in sources[dataset]["inputs"]["objects"]]
        secondary_complete = set(evaluations[dataset]["complete_ids"])
        common_ids = [oid for oid in planned_ids if oid in secondary_complete]
        common_set = set(common_ids)

        primary_eval_rows = [
            r
            for r in sources[dataset]["primary_evaluation"]["objects"]
            if r["object_id"] in common_set and r["method"] in REPRESENTATIONS
        ]
        secondary_eval_rows = [
            r
            for r in evaluations[dataset]["objects"]
            if r["object_id"] in common_set and r["method"] in REPRESENTATIONS
        ]

        expected_records = len(common_ids) * len(REPRESENTATIONS)
        if len(primary_eval_rows) != expected_records:
            raise ValueError(
                f"{dataset}: primary static rows {len(primary_eval_rows)} != expected {expected_records}"
            )
        if len(secondary_eval_rows) != expected_records:
            raise ValueError(
                f"{dataset}: secondary static rows {len(secondary_eval_rows)} != expected {expected_records}"
            )

        for backbone, rows in (
            ("primary_deepseek", primary_eval_rows),
            (cfg.secondary_label, secondary_eval_rows),
        ):
            for row in _summaries(rows, ("method",), EXPERIMENTS6_STATIC_METRICS):
                summary_rows.append({
                    "dataset": dataset,
                    "backbone": backbone,
                    "scale": None,
                    **row,
                })

            lookup = {
                method: {
                    r["object_id"]: r
                    for r in rows
                    if r["method"] == method
                }
                for method in REPRESENTATIONS
            }
            family = []
            for baseline in ("global", "local"):
                for metric in EXPERIMENTS6_STATIC_METRICS:
                    result = paired_object_statistics(
                        lookup["anchor"],
                        lookup[baseline],
                        metric,
                        make_experiments6_generation_cfg(cfg, dataset),
                    )
                    result_row = {
                        "dataset": dataset,
                        "backbone": backbone,
                        "experiment": "static",
                        "baseline": baseline,
                        "metric": metric,
                        "scale": None,
                        **result,
                    }
                    family.append(result_row)
                    category = _experiments6_category_direction(
                        rows,
                        metric,
                        baseline,
                    )
                    category_rows.append({
                        "dataset": dataset,
                        "backbone": backbone,
                        "experiment": "static",
                        "baseline": baseline,
                        "metric": metric,
                        "scale": None,
                        "category_count": category["categories"],
                        "category_wins": category["wins"],
                        "category_ties": category["ties"],
                        "category_losses": category["losses"],
                    })
            _holm_adjust(family)
            paired_rows.extend(family)

        primary_perturb_rows = [
            r
            for r in sources[dataset]["primary_perturbation"]["objects"]
            if r["object_id"] in common_set and r["method"] in REPRESENTATIONS
        ]
        secondary_perturb_rows = [
            r
            for r in perturbations[dataset]["objects"]
            if r["object_id"] in common_set and r["method"] in REPRESENTATIONS
        ]

        for backbone, rows in (
            ("primary_deepseek", primary_perturb_rows),
            (cfg.secondary_label, secondary_perturb_rows),
        ):
            for row in _summaries(rows, ("method", "scale"), EXPERIMENTS6_PERTURB_METRICS):
                summary_rows.append({
                    "dataset": dataset,
                    "backbone": backbone,
                    **row,
                })

            family = []
            for scale in cfg.scale_factors:
                by_method = {
                    method: {
                        r["object_id"]: r
                        for r in rows
                        if r["method"] == method and float(r["scale"]) == float(scale)
                    }
                    for method in REPRESENTATIONS
                }
                for baseline in ("global", "local"):
                    result = paired_object_statistics(
                        by_method["anchor"],
                        by_method[baseline],
                        "margin_drift",
                        make_experiments6_generation_cfg(cfg, dataset),
                    )
                    result_row = {
                        "dataset": dataset,
                        "backbone": backbone,
                        "experiment": "scale_perturbation",
                        "baseline": baseline,
                        "metric": "margin_drift",
                        "scale": float(scale),
                        **result,
                    }
                    family.append(result_row)
                    category = _experiments6_category_direction(
                        rows,
                        "margin_drift",
                        baseline,
                        scale=float(scale),
                    )
                    category_rows.append({
                        "dataset": dataset,
                        "backbone": backbone,
                        "experiment": "scale_perturbation",
                        "baseline": baseline,
                        "metric": "margin_drift",
                        "scale": float(scale),
                        "category_count": category["categories"],
                        "category_wins": category["wins"],
                        "category_ties": category["ties"],
                        "category_losses": category["losses"],
                    })
            _holm_adjust(family)
            paired_rows.extend(family)

        for oid in planned_ids:
            row = {
                "dataset": dataset,
                "object_id": oid,
                "category": sources[dataset]["shared"][oid]["category"],
                "secondary_complete": oid in secondary_complete,
            }
            for method in REPRESENTATIONS:
                generated = generations[dataset][method].get(oid, {})
                row[method + "_status"] = generated.get("status", "missing")
                row[method + "_attempt_count"] = int(generated.get("attempt_count", 0))
                row[method + "_first_attempt_valid"] = bool(
                    generated.get("attempts")
                    and generated["attempts"][0].get("validation", {}).get("valid")
                )
            generation_rows.append(row)

        dataset_pairs = [r for r in paired_rows if r["dataset"] == dataset]
        index = {
            (
                r["backbone"],
                r["experiment"],
                r["baseline"],
                r["metric"],
                r["scale"],
            ): r
            for r in dataset_pairs
        }
        secondary_backbone = cfg.secondary_label
        for experiment, metric, scales in (
            ("static", "shared_edge_obb_sat_deviation", [None]),
            ("static", "obb_overlap_pair_fraction", [None]),
            ("scale_perturbation", "margin_drift", [float(x) for x in cfg.scale_factors]),
        ):
            for scale in scales:
                for baseline in ("global", "local"):
                    primary_row = index[("primary_deepseek", experiment, baseline, metric, scale)]
                    secondary_row = index[(secondary_backbone, experiment, baseline, metric, scale)]
                    p_imp = primary_row["mean_improvement"]
                    s_imp = secondary_row["mean_improvement"]
                    concordance_rows.append({
                        "dataset": dataset,
                        "experiment": experiment,
                        "baseline": baseline,
                        "metric": metric,
                        "scale": scale,
                        "primary_improvement": p_imp,
                        "secondary_improvement": s_imp,
                        "same_direction": bool(
                            p_imp is not None
                            and s_imp is not None
                            and np.sign(p_imp) == np.sign(s_imp)
                        ),
                        "secondary_ci_excludes_zero_positive": bool(
                            secondary_row.get("ci_low") is not None
                            and secondary_row["ci_low"] > 0
                        ),
                    })

    output = {
        "version": EXPERIMENTS6_VERSION,
        "secondary_label": cfg.secondary_label,
        "secondary_model": cfg.secondary_model,
        "protocol": {
            "shared_structure": (
                "Frozen from the original DeepSeek experiments. The second backbone does NOT redo decomposition/connections."
            ),
            "changed_variable": "representation geometry-generation LLM backbone only",
            "partnet_subset": (
                f"{cfg.partnet_per_category} primary-complete objects per PartNet category, deterministic seed {cfg.subset_seed}"
            ),
            "openworld_subset": "all Experiments-5 primary complete cases",
            "comparison_cohort": (
                "For primary-vs-secondary effect comparison, both are recomputed/reported on the secondary three-method complete cohort."
            ),
            "primary_static_metrics": list(EXPERIMENTS6_STATIC_METRICS),
            "primary_perturbation_metric": "margin_drift",
            "statistics": (
                "paired object differences; category-stratified bootstrap; numerical-zero-consistent Wilcoxon; "
                "Holm correction separately within each dataset/backbone static and perturbation family"
            ),
            "scope_limitation": (
                "This tests representation-generation backbone robustness, not decomposition-backbone robustness."
            ),
        },
        "summary": summary_rows,
        "paired_statistics": paired_rows,
        "category_direction": category_rows,
        "generation_status": generation_rows,
        "backbone_concordance": concordance_rows,
    }

    out = cfg.output_path("tables")
    _write_csv(summary_rows, out / "summary.csv")
    _write_csv(paired_rows, out / "paired_statistics.csv")
    _write_csv(category_rows, out / "category_direction.csv")
    _write_csv(generation_rows, out / "generation_status.csv")
    _write_csv(concordance_rows, out / "backbone_concordance.csv")
    save_json(output, cfg.output_path("results.json"))
    return output


def experiments6_summary_view(results: dict, dataset: str, experiment: str = "static"):
    import pandas as pd
    rows = [
        r
        for r in results["summary"]
        if r["dataset"] == dataset
        and (
            (experiment == "static" and r.get("scale") is None)
            or (experiment == "scale_perturbation" and r.get("scale") is not None)
        )
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if experiment == "static":
        return frame.pivot_table(
            index=["backbone", "method"],
            columns="metric",
            values="mean",
        ).round(6)
    return frame.pivot_table(
        index=["backbone", "scale", "method"],
        columns="metric",
        values="mean",
    ).round(6)


def export_experiments6_review(cfg: Experiments6Config) -> Path:
    from zipfile import ZipFile, ZIP_DEFLATED

    required = [
        "experiments6_manifest.json",
        "runtime_metadata.json",
        "results.json",
        "tables/summary.csv",
        "tables/paired_statistics.csv",
        "tables/category_direction.csv",
        "tables/generation_status.csv",
        "tables/backbone_concordance.csv",
        "partnet/evaluation_all.json",
        "partnet/perturbation_all.json",
        "openworld/evaluation_all.json",
        "openworld/perturbation_all.json",
    ]
    path = cfg.output_path("experiments6_review.zip")
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name in required:
            source = cfg.output_path(name)
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.write(source, arcname=name)
    return path

# =============================================================================
# Experiments 7: Explicit Text-to-Structure Fidelity
# Evaluates whether an LLM recovers only the parts, multiplicities, root and
# direct attachment relations explicitly stated in synthetic natural language.
# No 3D geometry is generated in this experiment.
# =============================================================================

EXPERIMENTS7_VERSION = "explicit-text-structure-v1"
EXPERIMENTS7_PROMPT_VERSION = "explicit-fidelity-prompt-v1"
EXPERIMENTS7_PRIMARY_METRICS = (
    "part_instance_f1",
    "type_count_accuracy",
    "connection_f1",
    "root_correct",
    "exact_inventory",
    "exact_connections",
    "exact_structure",
)


@dataclass
class Experiments7Config:
    project_dir: str = "."
    output_dir: str = "experiments7_results"
    benchmark_path: str = "explicit_structure_benchmark_90.json"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_attempts: int = 3
    workers: int = 4
    timeout: float = 180.0
    transport_retries: int = 2
    bootstrap_samples: int = 10000
    statistics_seed: int = 42
    show_every: int = 15
    allow_api: bool = True

    def project_path(self, name: str | Path) -> Path:
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = Path(self.project_dir).expanduser().resolve() / path
        return path.resolve()

    def output_path(self, name: str | Path = "") -> Path:
        return self.project_path(self.output_dir) / name


@dataclass
class Experiments7ModelConfig:
    label: str
    model: str
    base_url: str
    api_key_env: str
    extra_body: dict = field(default_factory=dict)

    def scientific_config(self, cfg: Experiments7Config) -> dict:
        return {
            "label": self.label,
            "model": self.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "max_attempts": cfg.max_attempts,
            "extra_body": copy.deepcopy(self.extra_body),
        }


def _e7_canonical_name(name: str) -> str:
    value = str(name or "").strip().lower().replace("-", " ").replace("_", " ")
    return " ".join(value.split())


def build_experiments7_decomposition_messages(row: dict) -> list:
    description = str(row["description"]).strip()
    system = f"""
You extract an explicit physical part tree from a natural-language object description.

Return exactly one JSON object and no prose.

Output schema:
{{
  "object_category": "short category name",
  "root_part_id": "part_0",
  "parts": [
    {{
      "part_id": "part_0",
      "name": "singular part term",
      "instance_index": 0
    }}
  ],
  "connections": [
    {{
      "parent_part_id": "part_0",
      "child_part_id": "part_1"
    }}
  ]
}}

Critical fidelity rules:
- Use ONLY physical parts explicitly stated in the description.
- Do NOT add plausible but unstated parts from world knowledge.
- Preserve every explicitly stated part type.
- Preserve every explicit numerical multiplicity exactly.
- If a type has N instances, create N separate part records with the same singular name and instance_index 0..N-1.
- Preserve the part terminology from the description as closely as possible; do not replace it with a broader or narrower synonym.
- The description explicitly specifies a complete directed attachment tree.
- A statement such as "the child is attached to the parent" means parent -> child.
- A statement such as "the parent supports the child" also means parent -> child.
- Use ONLY direct attachment relations explicitly stated in the description.
- Do NOT infer extra connections.
- Choose as root the unique explicitly described part that has no parent attachment.
- Every non-root part must have exactly one parent.
- Return exactly N-1 connections for N part instances.
- The graph must be connected and acyclic.
- Assign part_0, part_1, ... in first-mention order; repeated instances are consecutive.
- Do not output coordinates, dimensions, rotations, anchors, geometry, explanations or markdown.

The benchmark is designed so that repeated same-name instances are structurally exchangeable. Do not invent labels such as "leg 1" or "leg 2" in the name field.
""".strip()
    user = "Object description:\n" + description
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def validate_experiments7_decomposition_schema(spec: dict) -> dict:
    check = validate_shared_part_spec(spec)
    errors = list(check.get("errors", []))
    if not isinstance(spec, dict):
        return {"valid": False, "errors": ["invalid_spec"]}
    category = spec.get("object_category")
    if not isinstance(category, str) or not category.strip():
        errors.append("invalid_object_category")
    parts = spec.get("parts", [])
    if len(parts) > 30:
        errors.append("too_many_parts")
    return {
        "valid": len(errors) == 0,
        "errors": sorted(set(errors)),
        "part_count": len(parts) if isinstance(parts, list) else 0,
        "connection_count": check.get("connection_count", 0),
    }


def load_experiments7_benchmark(cfg: Experiments7Config, model_configs: list[Experiments7ModelConfig]) -> dict:
    path = cfg.project_path(cfg.benchmark_path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiments 7 benchmark not found: {path}")
    benchmark = load_json(path)
    if not isinstance(benchmark, dict) or not isinstance(benchmark.get("objects"), list):
        raise ValueError("Experiments 7 benchmark must contain an objects list")
    objects = benchmark["objects"]
    ids = [str(row.get("object_id")) for row in objects]
    if len(objects) != 90 or len(ids) != len(set(ids)):
        raise ValueError("Expected exactly 90 unique ExplicitStructure benchmark prompts")
    base_groups = defaultdict(list)
    for row in objects:
        for key in (
            "object_id", "base_structure_id", "paraphrase_id", "paraphrase_style",
            "category", "complexity", "description", "gt_spec",
        ):
            if key not in row:
                raise ValueError(f"Benchmark row missing {key}: {row.get('object_id')}")
        validation = validate_shared_part_spec(row["gt_spec"])
        if not validation["valid"]:
            raise ValueError(f"Invalid GT spec for {row['object_id']}: {validation['errors']}")
        base_groups[str(row["base_structure_id"])].append(row)
    if len(base_groups) != 30 or any(len(rows) != 3 for rows in base_groups.values()):
        raise ValueError("Expected 30 base structures with exactly 3 paraphrases each")

    manifest = {
        "version": EXPERIMENTS7_VERSION,
        "prompt_version": EXPERIMENTS7_PROMPT_VERSION,
        "benchmark_path_name": path.name,
        "benchmark_hash": content_hash(benchmark),
        "prompt_count": len(objects),
        "base_structure_count": len(base_groups),
        "models": [mc.scientific_config(cfg) for mc in model_configs],
    }
    manifest_path = cfg.output_path("experiments7_manifest.json")
    if manifest_path.is_file():
        old = load_json(manifest_path)
        if old != manifest:
            raise ValueError(
                "Experiments 7 benchmark/scientific config changed. Use a new output_dir rather than overwriting the frozen experiment."
            )
    else:
        save_json(manifest, manifest_path)
    return {"benchmark": benchmark, "objects": objects, "manifest": manifest}


def _e7_request_cfg(cfg: Experiments7Config, model_cfg: Experiments7ModelConfig) -> PartNetConfig:
    request_cfg = PartNetConfig()
    request_cfg.model = model_cfg.model
    request_cfg.temperature = cfg.temperature
    request_cfg.max_tokens = cfg.max_tokens
    request_cfg.max_attempts = cfg.max_attempts
    request_cfg.workers = cfg.workers
    request_cfg.timeout = cfg.timeout
    request_cfg.transport_retries = cfg.transport_retries
    request_cfg.base_url = model_cfg.base_url
    request_cfg.extra_body = copy.deepcopy(model_cfg.extra_body)
    request_cfg.allow_api = cfg.allow_api
    return request_cfg


def run_experiments7_decomposition(
    benchmark_data: dict,
    cfg: Experiments7Config,
    model_cfg: Experiments7ModelConfig,
    client=None,
) -> dict:
    rows = benchmark_data["objects"]
    request_cfg = _e7_request_cfg(cfg, model_cfg)
    out_dir = cfg.output_path(Path("models") / model_cfg.label / "objects")
    out_dir.mkdir(parents=True, exist_ok=True)

    if client is None and cfg.allow_api:
        client = create_openai_compatible_client(
            api_key_env=model_cfg.api_key_env,
            base_url=model_cfg.base_url,
            timeout=cfg.timeout,
        )

    scientific = model_cfg.scientific_config(cfg)

    def one(row: dict) -> tuple[str, dict]:
        oid = str(row["object_id"])
        messages = build_experiments7_decomposition_messages(row)
        fingerprint = content_hash({
            "version": EXPERIMENTS7_VERSION,
            "prompt_version": EXPERIMENTS7_PROMPT_VERSION,
            "scientific_model_config": scientific,
            "object_id": oid,
            "description": row["description"],
        })
        path = out_dir / f"{oid}.json"
        cached = _read_cache(path, fingerprint)
        if cached is not None and cached.get("status") == "valid":
            return oid, cached
        if not cfg.allow_api:
            if cached is None:
                return oid, {"fingerprint": fingerprint, "status": "missing_cache", "attempts": []}
            return oid, cached
        record = _generate_valid_json(
            client,
            request_cfg,
            path,
            messages,
            validate_experiments7_decomposition_schema,
            fingerprint,
        )
        return oid, record

    results = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(cfg.workers))) as executor:
        futures = {executor.submit(one, row): row for row in rows}
        for future in as_completed(futures):
            oid, record = future.result()
            results[oid] = record
            completed += 1
            if cfg.show_every and (completed % cfg.show_every == 0 or completed == len(rows)):
                valid = sum(r.get("status") == "valid" for r in results.values())
                print(f"{model_cfg.label}: {completed}/{len(rows)} processed, valid={valid}")

    ordered = []
    for row in rows:
        oid = str(row["object_id"])
        rec = results[oid]
        attempts = rec.get("attempts", [])
        ordered.append({
            "object_id": oid,
            "base_structure_id": row["base_structure_id"],
            "category": row["category"],
            "complexity": row["complexity"],
            "paraphrase_id": row["paraphrase_id"],
            "paraphrase_style": row["paraphrase_style"],
            "status": rec.get("status"),
            "attempt_count": int(rec.get("attempt_count", len(attempts))),
            "first_attempt_valid": bool(attempts and attempts[0].get("validation", {}).get("valid")),
            "returned_model": next((a.get("returned_model") for a in reversed(attempts) if a.get("returned_model")), None),
            "result": copy.deepcopy(rec.get("result")) if rec.get("status") == "valid" else None,
            "final_errors": copy.deepcopy(attempts[-1].get("validation", {}).get("errors", [])) if attempts else [],
        })
    aggregate = {
        "version": EXPERIMENTS7_VERSION,
        "model": scientific,
        "objects": ordered,
        "valid_count": sum(r["status"] == "valid" for r in ordered),
        "prompt_count": len(ordered),
    }
    save_json(aggregate, cfg.output_path(Path("models") / model_cfg.label / "decomposition_all.json"))
    return aggregate


def _e7_semantic_view(spec: dict) -> dict:
    parts = spec.get("parts", []) if isinstance(spec, dict) else []
    part_names = {}
    counts = Counter()
    for part in parts:
        pid = str(part.get("part_id"))
        name = _e7_canonical_name(part.get("name"))
        part_names[pid] = name
        if name:
            counts[name] += 1
    edge_counts = Counter()
    for edge in spec.get("connections", []) if isinstance(spec, dict) else []:
        parent = part_names.get(str(edge.get("parent_part_id")), "")
        child = part_names.get(str(edge.get("child_part_id")), "")
        if parent and child:
            edge_counts[(parent, child)] += 1
    root_name = part_names.get(str(spec.get("root_part_id")), "") if isinstance(spec, dict) else ""
    return {"part_counts": counts, "edge_counts": edge_counts, "root_name": root_name}


def _e7_multiset_prf(gt: Counter, pred: Counter) -> tuple[int, int, int, float, float, float]:
    keys = set(gt) | set(pred)
    tp = sum(min(gt[k], pred[k]) for k in keys)
    gt_n = sum(gt.values())
    pred_n = sum(pred.values())
    precision = 1.0 if pred_n == 0 and gt_n == 0 else (tp / pred_n if pred_n else 0.0)
    recall = 1.0 if gt_n == 0 else tp / gt_n
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return tp, pred_n, gt_n, float(precision), float(recall), float(f1)


def evaluate_experiments7_prediction(row: dict, generation_row: dict) -> dict:
    gt = _e7_semantic_view(row["gt_spec"])
    schema_valid = generation_row.get("status") == "valid" and isinstance(generation_row.get("result"), dict)
    pred_spec = generation_row.get("result") if schema_valid else {"parts": [], "connections": []}
    pred = _e7_semantic_view(pred_spec)

    _, pred_parts, gt_parts, part_precision, part_recall, part_f1 = _e7_multiset_prf(gt["part_counts"], pred["part_counts"])
    _, pred_edges, gt_edges, edge_precision, edge_recall, edge_f1 = _e7_multiset_prf(gt["edge_counts"], pred["edge_counts"])

    union_types = set(gt["part_counts"]) | set(pred["part_counts"])
    exact_type_counts = sum(gt["part_counts"].get(k, 0) == pred["part_counts"].get(k, 0) for k in union_types)
    type_count_accuracy = exact_type_counts / len(union_types) if union_types else 1.0

    missing_instances = sum(max(gt["part_counts"].get(k, 0) - pred["part_counts"].get(k, 0), 0) for k in union_types)
    unsupported_instances = sum(max(pred["part_counts"].get(k, 0) - gt["part_counts"].get(k, 0), 0) for k in union_types)
    missing_edges = sum(max(gt["edge_counts"].get(k, 0) - pred["edge_counts"].get(k, 0), 0) for k in set(gt["edge_counts"]) | set(pred["edge_counts"]))
    unsupported_edges = sum(max(pred["edge_counts"].get(k, 0) - gt["edge_counts"].get(k, 0), 0) for k in set(gt["edge_counts"]) | set(pred["edge_counts"]))

    exact_inventory = int(schema_valid and gt["part_counts"] == pred["part_counts"])
    exact_connections = int(schema_valid and gt["edge_counts"] == pred["edge_counts"])
    root_correct = int(schema_valid and gt["root_name"] == pred["root_name"])
    exact_structure = int(exact_inventory and exact_connections and root_correct)

    return {
        "object_id": row["object_id"],
        "base_structure_id": row["base_structure_id"],
        "category": row["category"],
        "complexity": row["complexity"],
        "paraphrase_id": row["paraphrase_id"],
        "paraphrase_style": row["paraphrase_style"],
        "schema_valid": int(schema_valid),
        "attempt_count": int(generation_row.get("attempt_count", 0)),
        "part_instance_precision": part_precision,
        "part_instance_recall": part_recall,
        "part_instance_f1": part_f1,
        "type_count_accuracy": float(type_count_accuracy),
        "connection_precision": edge_precision,
        "connection_recall": edge_recall,
        "connection_f1": edge_f1,
        "root_correct": root_correct,
        "exact_inventory": exact_inventory,
        "exact_connections": exact_connections,
        "exact_structure": exact_structure,
        "missing_part_instances": int(missing_instances),
        "unsupported_part_instances": int(unsupported_instances),
        "missing_connections": int(missing_edges),
        "unsupported_connections": int(unsupported_edges),
        "gt_part_instances": int(gt_parts),
        "pred_part_instances": int(pred_parts),
        "gt_connections": int(gt_edges),
        "pred_connections": int(pred_edges),
        "gt_root_name": gt["root_name"],
        "pred_root_name": pred["root_name"],
    }


def run_experiments7_evaluation(
    benchmark_data: dict,
    generations_by_model: dict[str, dict],
    cfg: Experiments7Config,
) -> dict:
    benchmark_index = {str(r["object_id"]): r for r in benchmark_data["objects"]}
    all_models = {}
    for label, aggregate in generations_by_model.items():
        gen_index = {str(r["object_id"]): r for r in aggregate["objects"]}
        rows = []
        for oid, row in benchmark_index.items():
            if oid not in gen_index:
                raise ValueError(f"{label}: missing generation row {oid}")
            rows.append(evaluate_experiments7_prediction(row, gen_index[oid]))
        all_models[label] = rows
        save_json(
            {"version": EXPERIMENTS7_VERSION, "model_label": label, "objects": rows},
            cfg.output_path(Path("models") / label / "evaluation.json"),
        )
    result = {"version": EXPERIMENTS7_VERSION, "models": all_models}
    save_json(result, cfg.output_path("evaluation_all.json"))
    return result


def _e7_cluster_bootstrap_ci(rows: list[dict], metric: str, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    by_base = defaultdict(list)
    for row in rows:
        by_base[str(row["base_structure_id"])].append(float(row[metric]))
    base_ids = sorted(by_base)
    base_means = np.array([np.mean(by_base[b]) for b in base_ids], dtype=float)
    mean = float(np.mean(base_means))
    if len(base_means) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    samples = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, len(base_means), len(base_means))
        samples[i] = float(np.mean(base_means[idx]))
    return mean, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize_experiments7_results(
    benchmark_data: dict,
    evaluation: dict,
    cfg: Experiments7Config,
) -> dict:
    summaries = []
    category = []
    complexity = []
    style = []
    base_consistency = []

    summary_metrics = (
        "schema_valid", "part_instance_precision", "part_instance_recall", "part_instance_f1",
        "type_count_accuracy", "connection_precision", "connection_recall", "connection_f1",
        "root_correct", "exact_inventory", "exact_connections", "exact_structure",
        "missing_part_instances", "unsupported_part_instances", "missing_connections", "unsupported_connections",
    )

    for model_index, (label, rows) in enumerate(evaluation["models"].items()):
        for metric in summary_metrics:
            mean, lo, hi = _e7_cluster_bootstrap_ci(
                rows, metric, cfg.bootstrap_samples, cfg.statistics_seed + 1000 * model_index + len(summaries)
            )
            summaries.append({"model": label, "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi, "n_prompts": len(rows), "n_base_structures": 30})

        for group_name, output in (("category", category), ("complexity", complexity), ("paraphrase_style", style)):
            groups = defaultdict(list)
            for row in rows:
                groups[str(row[group_name])].append(row)
            for group_value, group_rows in sorted(groups.items()):
                for metric in EXPERIMENTS7_PRIMARY_METRICS:
                    output.append({
                        "model": label,
                        group_name: group_value,
                        "metric": metric,
                        "mean": float(np.mean([float(r[metric]) for r in group_rows])),
                        "n": len(group_rows),
                    })

        groups = defaultdict(list)
        for row in rows:
            groups[str(row["base_structure_id"])].append(row)
        for base_id, group_rows in sorted(groups.items()):
            if len(group_rows) != 3:
                raise ValueError(f"Expected 3 paraphrases for {base_id}")
            base_consistency.append({
                "model": label,
                "base_structure_id": base_id,
                "category": group_rows[0]["category"],
                "complexity": group_rows[0]["complexity"],
                "exact_structure_count": int(sum(r["exact_structure"] for r in group_rows)),
                "all_three_exact": int(all(r["exact_structure"] for r in group_rows)),
                "all_three_inventory_exact": int(all(r["exact_inventory"] for r in group_rows)),
                "all_three_connections_exact": int(all(r["exact_connections"] for r in group_rows)),
                "mean_part_f1": float(np.mean([r["part_instance_f1"] for r in group_rows])),
                "mean_connection_f1": float(np.mean([r["connection_f1"] for r in group_rows])),
            })

    labels = list(evaluation["models"].keys())
    cross_model = {}
    paired = []
    if len(labels) == 2:
        a_label, b_label = labels
        a = {r["object_id"]: r for r in evaluation["models"][a_label]}
        b = {r["object_id"]: r for r in evaluation["models"][b_label]}
        ids = sorted(set(a) & set(b))
        both_exact = sum(a[i]["exact_structure"] and b[i]["exact_structure"] for i in ids)
        a_only = sum(a[i]["exact_structure"] and not b[i]["exact_structure"] for i in ids)
        b_only = sum(b[i]["exact_structure"] and not a[i]["exact_structure"] for i in ids)
        neither = len(ids) - both_exact - a_only - b_only
        cross_model = {
            "model_a": a_label,
            "model_b": b_label,
            "n": len(ids),
            "both_exact": int(both_exact),
            "a_only_exact": int(a_only),
            "b_only_exact": int(b_only),
            "neither_exact": int(neither),
            "exact_agreement_rate": float((both_exact + neither) / len(ids)),
        }
        from scipy.stats import binomtest
        discordant = a_only + b_only
        cross_model["mcnemar_exact_p"] = 1.0 if discordant == 0 else float(binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue)

        # Cluster by base structure for model-difference CIs; this is descriptive, not a superiority claim.
        for metric in ("part_instance_f1", "type_count_accuracy", "connection_f1", "exact_structure"):
            by_base = defaultdict(list)
            for oid in ids:
                base = a[oid]["base_structure_id"]
                by_base[base].append(float(a[oid][metric]) - float(b[oid][metric]))
            base_ids = sorted(by_base)
            diffs = np.array([np.mean(by_base[x]) for x in base_ids], dtype=float)
            rng = np.random.default_rng(cfg.statistics_seed + 9000 + len(paired))
            boot = np.empty(cfg.bootstrap_samples, dtype=float)
            for j in range(cfg.bootstrap_samples):
                idx = rng.integers(0, len(diffs), len(diffs))
                boot[j] = np.mean(diffs[idx])
            paired.append({
                "metric": metric,
                "model_a": a_label,
                "model_b": b_label,
                "mean_a_minus_b": float(np.mean(diffs)),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "n_base_structures": len(diffs),
            })

    result = {
        "version": EXPERIMENTS7_VERSION,
        "benchmark": benchmark_data["benchmark"].get("benchmark", "ExplicitStructure-90"),
        "summary": summaries,
        "category_summary": category,
        "complexity_summary": complexity,
        "paraphrase_style_summary": style,
        "base_structure_consistency": base_consistency,
        "cross_model_exact": cross_model,
        "cross_model_metric_differences": paired,
        "protocol": {
            "main_scope": "Fidelity to explicitly stated part types, multiplicities, root and direct attachment relations.",
            "matching": "Exact canonical part-term matching; repeated same-name instances are evaluated permutation-invariantly through semantic multiset counts.",
            "retry_policy": "Retries receive only JSON/schema errors. Ground-truth semantic mismatches are never fed back to the model.",
            "statistics_unit": "Confidence intervals resample 30 base structures as clusters because each structure has three paraphrases.",
            "not_measured": "Open-ended plausibility, unstated world knowledge, mesh quality and 3D geometry are outside this benchmark.",
        },
    }
    save_json(result, cfg.output_path("results.json"))
    return result


def experiments7_summary_view(results: dict):
    import pandas as pd
    frame = pd.DataFrame(results["summary"])
    if frame.empty:
        return frame
    return frame.pivot(index="model", columns="metric", values="mean").round(4)


def export_experiments7_review(cfg: Experiments7Config, model_configs: list[Experiments7ModelConfig]) -> Path:
    from zipfile import ZipFile, ZIP_DEFLATED
    required = [
        cfg.output_path("experiments7_manifest.json"),
        cfg.output_path("evaluation_all.json"),
        cfg.output_path("results.json"),
    ]
    for mc in model_configs:
        required.append(cfg.output_path(Path("models") / mc.label / "decomposition_all.json"))
        required.append(cfg.output_path(Path("models") / mc.label / "evaluation.json"))
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Experiments 7 outputs: {missing}")
    path = cfg.output_path("experiments7_review.zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for p in required:
            archive.write(p, arcname=str(p.relative_to(cfg.output_path())))
    return path



