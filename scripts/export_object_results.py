#!/usr/bin/env python3
"""Export cached object-level measurements without model calls or source mutation.

Unpack the published cached_runs archives so their experiment directories are
under --data-root (the repository root by default). Additional offline-control
reference JSON files are loaded separately from --controls-dir. CSV rows retain
the object as the statistical unit. Edge interventions and seeds are averaged
within objects in the original records; seed-level exports are also provided.
Only the Python standard library is needed. Existing output files are replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SAT = "shared_edge_obb_sat_deviation"
TOL = 1e-9


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def flatten(rows, **extra):
    """Keep identifiers and scalar metadata, expand object-level metrics only."""
    out = []
    for row in rows:
        rec = {k: v for k, v in row.items()
               if not isinstance(v, (dict, list)) and k != "fingerprint"}
        rec.update(row.get("metrics", {}))
        rec.update(extra)
        out.append(rec)
    return out


def average(rows, metric, **where):
    values = [r[metric] for r in rows
              if all(r.get(k) == v for k, v in where.items())
              and isinstance(r.get(metric), (int, float))
              and math.isfinite(r[metric])]
    if not values:
        raise ValueError(f"No finite values for {metric}: {where}")
    return statistics.fmean(values), len(values)


def paired(rows, metric, baseline, **where):
    selected = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    lookup = {(str(r["object_id"]), r["method"]): r[metric] for r in selected}
    ids = sorted({oid for oid, method in lookup if method == "anchor"})
    diffs = [lookup[(oid, baseline)] - lookup[(oid, "anchor")] for oid in ids]
    clean = [0.0 if abs(x) <= TOL else x for x in diffs]
    return {"n": len(clean), "wins": sum(x > 0 for x in clean),
            "ties": sum(x == 0 for x in clean), "losses": sum(x < 0 for x in clean),
            "median_baseline_minus_anchor": statistics.median(clean)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO)
    parser.add_argument("--output-dir", type=Path, default=REPO / "data/object_results")
    parser.add_argument("--controls-dir", type=Path, default=REPO / "data/additional_controls")
    args = parser.parse_args()
    sources, exports, checks = {}, {}, []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def load(relative):
        path = args.data_root / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read_json(path)

    def control(filename):
        path = args.controls_dir / filename
        sources["data/additional_controls/" + filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read_json(path)

    def export(filename, rows, unique_key, expected_rows):
        if len(rows) != expected_rows:
            raise ValueError(f"{filename}: expected {expected_rows} rows, got {len(rows)}")
        keys = [tuple(str(r.get(k, "")) for k in unique_key) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{filename}: duplicate key {unique_key}")
        preferred = ["dataset", "backbone", "object_id", "category", "method", "regime", "scale", "seed"]
        all_fields = list(dict.fromkeys(k for r in rows for k in r))
        fields = [k for k in preferred if k in all_fields] + [k for k in all_fields if k not in preferred]
        path = args.output_dir / filename
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                                 if isinstance(v, (dict, list)) else v for k, v in row.items()})
        exports[filename] = {"rows": len(rows), "unique_objects": len({str(r["object_id"]) for r in rows}),
                             "unique_key": unique_key, "fields": fields,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def check_mean(table, rows, metric, expected, digits=5, n=None, **where):
        actual, count = average(rows, metric, **where)
        passed = round(actual, digits) == round(expected, digits) and (n is None or count == n)
        checks.append({"table": table, "metric": metric, "selection": where, "n": count,
                       "mean": actual, "paper_rounded_value": expected, "digits": digits, "pass": passed})

    pn_eval = load("partnet_results/experiments2/evaluation_all.json")
    ow_eval = load("openworld_results/evaluation_all.json")
    pn, ow = flatten(pn_eval["objects"]), flatten(ow_eval["objects"])
    export("partnet_static.csv", pn, ["object_id", "method"], 2631)
    export("openworld_static.csv", ow, ["object_id", "method"], 357)
    random_rows = {}
    child_rows = {}
    for dataset, prefix, n in [("partnet", "partnet_results/experiments2", 877),
                               ("openworld", "openworld_results", 119)]:
        random_data = load(prefix + "/random_anchor_all.json")
        random_rows[dataset] = flatten(random_data["objects"])
        export(dataset + "_random_anchors.csv", random_rows[dataset], ["object_id", "method"], n)
        seed_rows = []
        for row in random_data["objects"]:
            for seed in row["seed_results"]:
                seed_rows.append({"object_id": row["object_id"], "category": row["category"],
                                  "method": "random_anchor", "seed": seed["seed"], **seed["metrics"]})
        export(dataset + "_random_anchor_seeds.csv", seed_rows, ["object_id", "seed"], n * 5)
        child_rows[dataset] = flatten(load(prefix + "/perturbation_all.json")["objects"])
        export(dataset + "_child_perturbations.csv", child_rows[dataset], ["object_id", "method", "scale"], n * 9)
    same = flatten(load("partnet_results/experiments2/same_geometry/results.json")["object_metrics"])
    export("partnet_child_perturbations_matched.csv", same, ["object_id", "method", "scale"], 7893)
    parents = flatten(load("partnet_results/experiments3/geometry_results.json")["records"])
    export("parent_edits.csv", parents, ["object_id", "regime", "method"], 5262)
    tasks = load("partnet_results/experiments3/edit_tasks.json")["tasks"]
    tasks_out = []
    for task in tasks:
        row = {k: v for k, v in task.items() if k != "expected_plan"}
        row.update({"intended_" + k: v for k, v in task["expected_plan"].items()})
        tasks_out.append(row)
    export("parent_edit_tasks.csv", tasks_out, ["object_id"], 877)
    parsed = load("partnet_results/experiments3/language_parse_summary.json")["rows"]
    export("parent_edit_parser.csv", parsed, ["object_id"], 877)

    manifest = load("experiments6_results/qwen3_7_plus/experiments6_manifest.json")
    bb_static, bb_child = [], []
    bb_cohorts = {}
    for dataset, primary, primary_child in [("partnet", pn, child_rows["partnet"]),
                                            ("openworld", ow, child_rows["openworld"])]:
        prefix = "experiments6_results/qwen3_7_plus/" + dataset
        secondary = load(prefix + "/evaluation_all.json")
        planned = {str(r["object_id"]) for r in manifest[dataset + "_objects"]}
        common = planned & {str(x) for x in secondary["complete_ids"]}
        primary_methods = Counter(r["object_id"] for r in primary if str(r["object_id"]) in common)
        if len(common) != (121 if dataset == "partnet" else 117) or any(v != 3 for v in primary_methods.values()):
            raise ValueError(f"Invalid common cohort for {dataset}")
        bb_cohorts[dataset] = {"selected_n": len(planned), "common_n": len(common),
                               "common_ids": sorted(common), "excluded_ids": sorted(planned - common),
                               "secondary_all_available_records": len(secondary["objects"])}
        secondary_child = flatten(load(prefix + "/perturbation_all.json")["objects"])
        for name, static, perturbed in [("primary_deepseek", primary, primary_child),
                                        ("qwen3_7_plus", flatten(secondary["objects"]), secondary_child)]:
            bb_static.extend({**r, "dataset": dataset, "backbone": name} for r in static if str(r["object_id"]) in common)
            bb_child.extend({**r, "dataset": dataset, "backbone": name} for r in perturbed if str(r["object_id"]) in common)
    export("backbone_static_common_cohorts.csv", bb_static, ["dataset", "backbone", "object_id", "method"], 1428)
    export("backbone_child_perturbations_common_cohorts.csv", bb_child,
           ["dataset", "backbone", "object_id", "method", "scale"], 4284)
    explicit = []
    for model, records in load("experiments7_results/evaluation_all.json")["models"].items():
        explicit.extend({**r, "backbone": model} for r in records)
    export("explicit_structure_recovery.csv", explicit, ["backbone", "object_id"], 180)

    size_data = control("size_aware_local_control.json")
    random_data = control("matched_anchor_reparameterization.json")
    size_rows, refit_rows, refit_seeds = [], [], []
    for row in size_data["objects"]:
        for method, score in row["scores"].items():
            size_rows.append({"object_id": row["object_id"], "category": row["category"], "method": method,
                              "regime": "same_initial_geometry", "axis": row["axis"], "factor": row["factor"],
                              "target_is_root": row["target_is_root"], "outgoing_edge_count": row["outgoing_edges"],
                              "outgoing_margin_drift": score, "outgoing_before_abs_margin": row["before_mean_abs_margin"],
                              "outgoing_after_abs_margin": row["after_mean_abs_margin"][method]})
    variant_fields = {"anchor": ("original_anchor", "original_after_abs_margin"),
                      "global": ("matched_global", "matched_global_after_abs_margin"),
                      "local": ("matched_local", "matched_local_after_abs_margin"),
                      "random_refit_anchor": ("random_refit_anchor", "random_refit_after_abs_margin")}
    for row in random_data["objects"]:
        common = {k: row[k] for k in ["object_id", "category", "axis", "factor", "target_part_id", "outgoing_before_abs_margin"]}
        common["regime"] = "same_initial_geometry"
        for method, (drift, post) in variant_fields.items():
            refit_rows.append({**common, "method": method, "outgoing_margin_drift": row[drift],
                               "outgoing_after_abs_margin": row[post]})
        for seed, value in row["random_seed_scores"].items():
            refit_seeds.append({**common, "method": "random_refit_anchor", "seed": int(seed),
                                "outgoing_margin_drift": value,
                                "outgoing_after_abs_margin": row["random_refit_seed_after_abs_margins"][str(seed)]})
    export("parent_control_size_aware_local.csv", size_rows, ["object_id", "method"], 3508)
    export("parent_control_random_refit.csv", refit_rows, ["object_id", "method"], 3508)
    export("parent_control_random_refit_seeds.csv", refit_seeds, ["object_id", "seed"], 4385)

    # Each check below recomputes a displayed cell from object-level records.
    metrics = ["relative_translation_error", "world_aabb_size_mae", "gt_edge_bbox_margin_mae", SAT, "obb_overlap_pair_fraction"]
    table1 = {"anchor": [.35309, .17320, .10682, .02784, .13938],
              "global": [.33333, .17567, .12724, .10135, .23813],
              "local": [.33363, .17159, .12058, .08464, .24683]}
    for method, values in table1.items():
        for metric, expected in zip(metrics, values):
            check_mean(1, pn, metric, expected, n=876 if metric == "gt_edge_bbox_margin_mae" else 877, method=method)
    for dataset, rows, sat, overlap, n in [("partnet", random_rows["partnet"], .03574, .40078, 877),
                                          ("openworld", random_rows["openworld"], .03204, .37103, 119)]:
        check_mean(2, rows, SAT, sat, n=n)
        check_mean(2, rows, "obb_overlap_pair_fraction", overlap, n=n)
    child_values = [(child_rows["partnet"], 877, [[.00288,.02457,.02336],[.00266,.02111,.02117],[.00642,.04664,.04730]]),
                    (same, 877, [[.00288,.02198,.02198],[.00266,.01882,.01882],[.00642,.04222,.04222]]),
                    (child_rows["openworld"],119, [[.00185,.02260,.01887],[.00151,.02010,.01744],[.00367,.04550,.03973]])]
    for rows, n, conditions in child_values:
        for scale, values in zip([.8,1.2,1.5], conditions):
            for method, expected in zip(["anchor","global","local"], values):
                check_mean(3, rows, "margin_drift", expected, n=n, scale=scale, method=method)
    for regime, values in [("independent_generation",[.000853,.011619,.010867]),
                           ("same_initial_geometry",[.000853,.010345,.010345])]:
        for method, expected in zip(["anchor","global","local"],values):
            check_mean(4, parents, "outgoing_margin_drift", expected, digits=6, n=877, regime=regime, method=method)
    for rows, method, drift, post, outcomes in [(size_rows,"anchor",.000853,.026007,None),
                                             (size_rows,"local",.010345,.033657,(347,529,1)),
                                             (size_rows,"size_aware_local",.007422,.031564,(364,510,3)),
                                             (refit_rows,"random_refit_anchor",.011222,.033424,(418,444,15))]:
        check_mean(5,rows,"outgoing_margin_drift",drift,digits=6,n=877,method=method)
        check_mean(5,rows,"outgoing_after_abs_margin",post,digits=6,n=877,method=method)
        check_mean(5,rows,"outgoing_before_abs_margin",.026582,digits=6,n=877,method=method)
        if outcomes:
            result=paired(rows,"outgoing_margin_drift",method)
            checks.append({"table":5,"baseline":method,"paired_outcomes":result,
                           "pass":tuple(result[k] for k in ["wins","ties","losses"])==outcomes})
    for method, values in {"anchor":[.01791,.13209,.10539],"global":[.05293,.18726,.12982],"local":[.04935,.21596,.12868]}.items():
        for metric, value in zip([SAT,"obb_overlap_pair_fraction","part_volume_sum"], values):
            check_mean(6,ow,metric,value,n=119,method=method)
    bb_stats=load("experiments6_results/qwen3_7_plus/results.json")["paired_statistics"]
    for row in bb_stats:
        if row["experiment"]=="static" and row["metric"]==SAT:
            model="primary_deepseek" if row["backbone"]=="primary_deepseek" else "qwen3_7_plus"
            for method, expected in [("anchor",row["anchor_mean"]),(row["baseline"],row["baseline_mean"])]:
                check_mean(7,bb_static,SAT,expected,digits=12,n=row["n"],dataset=row["dataset"],backbone=model,method=method)
    for model, values in {"deepseek":[1,1,1,1,1],"qwen3_7_plus":[1,.9950617283950618,.9944444444444445,1,89/90]}.items():
        for metric,value in zip(["schema_valid","part_instance_f1","connection_f1","root_correct","exact_structure"],values):
            check_mean(8,explicit,metric,value,digits=12,n=90,backbone=model)
    outcomes = [(pn,SAT,{},"global",(613,174,90)),(pn,SAT,{},"local",(532,221,124)),
                (ow,SAT,{},"global",(86,21,12)),(ow,SAT,{},"local",(91,18,10)),
                (parents,"outgoing_margin_drift",{"regime":"independent_generation"},"global",(392,442,43)),
                (parents,"outgoing_margin_drift",{"regime":"independent_generation"},"local",(396,444,37)),
                (parents,"outgoing_margin_drift",{"regime":"same_initial_geometry"},"global",(347,529,1)),
                (parents,"outgoing_margin_drift",{"regime":"same_initial_geometry"},"local",(347,529,1))]
    for rows,metric,where,baseline,expected in outcomes:
        result=paired(rows,metric,baseline,**where)
        checks.append({"table":9,"metric":metric,"baseline":baseline,"selection":where,"paired_outcomes":result,
                       "pass":tuple(result[k] for k in ["wins","ties","losses"])==expected})
    report={"all_checks_passed":all(r["pass"] for r in checks),"check_count":len(checks),
            "source_sha256":sources,"exports":exports,"backbone_cohorts":bb_cohorts,"checks":checks,
            "notes":["CSV files contain object-level records, not aggregate summary rows.",
                     "Retain original archives: common-cohort CSVs intentionally omit nonpaired Qwen outputs and do not replace failure records.",
                     "PartNet GT-edge margin MAE has 876 finite values per method; all other main columns have 877.",
                     "Bootstrap intervals and adjusted p-values are preserved in source aggregate JSON; this exporter does not rerun inference.",
                     "Paper table numbering differs from legacy table01--table07 filenames: added controls are Table 5 and outcomes are Table 9.",
                     "Published files preserve cached outcomes; no new LLM generations are performed."]}
    (args.output_dir/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"csv_files":len(exports),"object_level_rows":sum(x["rows"] for x in exports.values()),
                      "verification_checks":len(checks),"all_checks_passed":report["all_checks_passed"]}))
    if not report["all_checks_passed"]:
        raise SystemExit("Paper verification failed; inspect verification.json before release.")


if __name__=="__main__":
    main()
