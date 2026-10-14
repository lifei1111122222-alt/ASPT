import argparse, copy, hashlib, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(
    description='Reproduce the matched-geometry anchor reparameterization control from frozen caches; no API calls.'
)
parser.add_argument(
    '--data-root', type=Path, default=REPO_ROOT,
    help='Directory containing partnet_results/ (default: repository root).'
)
parser.add_argument(
    '--output-dir', type=Path, default=REPO_ROOT / 'reproduced',
    help='Directory for reproduced reports (default: repository/reproduced).'
)
args = parser.parse_args()
ROOT = args.data_root.expanduser().resolve()
OUTPUT_DIR = args.output_dir.expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUTPUT_DIR / 'anchor_reparameterization_control.json'
TOL = 1e-9
SEEDS = [0, 1, 2, 3, 4]

def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))

def qmat(q):
    q = np.asarray(q, dtype=float)
    w,x,y,z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def decode(raw, spec):
    pars = {p["part_id"]:p for p in raw["parts"]}
    parents = {e["child_part_id"]:e["parent_part_id"] for e in spec["connections"]}
    world = {}
    def solve(pid):
        if pid in world: return world[pid]
        p = pars[pid]
        s = np.asarray(p["bbox_size"], float)
        if pid == spec["root_part_id"]:
            c = np.asarray(p["center"], float)
            r = qmat(p["rotation"])
        else:
            par = solve(parents[pid]); r = par["r"] @ qmat(p["rotation"])
            c = (par["c"] + par["r"] @ (par["s"]*np.asarray(p["parent_anchor"])+np.asarray(p["offset"]))
                 - r @ (s*np.asarray(p["child_anchor"])))
        world[pid] = {"c":c, "r":r, "s":s}
        return world[pid]
    for p in spec["parts"]: solve(p["part_id"])
    return world

def sat(a,b):
    ra,rb=a["r"],b["r"]
    axes=np.vstack((ra.T,rb.T,np.cross(ra.T[:,None,:],rb.T[None,:,:]).reshape(-1,3)))
    norms=np.linalg.norm(axes,axis=1)
    axes=axes[norms>1e-10]/norms[norms>1e-10,None]
    radii_a=np.abs(axes@ra)@(a["s"]/2)
    radii_b=np.abs(axes@rb)@(b["s"]/2)
    return float(np.max(np.abs(axes@(b["c"]-a["c"]))-radii_a-radii_b))

def max_layout_error(a,b):
    return max(float(np.max(np.abs(a[pid][k]-b[pid][k]))) for pid in a for k in ("c","r","s"))

def edit_raw(raw,plan):
    x=copy.deepcopy(raw)
    axis={"x":0,"y":1,"z":2}[plan["axis"]]
    for p in x["parts"]:
        if p["part_id"]==plan["target_part_id"]:
            p["bbox_size"][axis]*=float(plan["scale_factor"])
    return x

def outgoing_drift(before,after,spec,plan):
    p=plan["target_part_id"]
    edges=[e for e in spec["connections"] if e["parent_part_id"]==p]
    assert edges
    return float(np.mean([abs(sat(after[p],after[e["child_part_id"]])-sat(before[p],before[e["child_part_id"]])) for e in edges]))

def outgoing_abs_margin(layout,spec,plan):
    p=plan["target_part_id"]
    edges=[e for e in spec["connections"] if e["parent_part_id"]==p]
    return float(np.mean([abs(sat(layout[p],layout[e["child_part_id"]])) for e in edges]))

def outgoing_closed_form(base,raw,spec,plan):
    p=plan["target_part_id"]; axis={"x":0,"y":1,"z":2}[plan["axis"]]
    par=base[p]; ds=np.zeros(3);ds[axis]=par["s"][axis]*(float(plan["scale_factor"])-1)
    par_after={"c":par["c"],"r":par["r"],"s":par["s"]+ds}
    pars={x["part_id"]:x for x in raw["parts"]};vals=[]
    for e in spec["connections"]:
        if e["parent_part_id"]!=p:continue
        cid=e["child_part_id"]; child=base[cid]
        newchild={"c":child["c"]+par["r"]@(ds*np.asarray(pars[cid]["parent_anchor"])),"r":child["r"],"s":child["s"]}
        vals.append(abs(sat(par_after,newchild)-sat(par,child)))
    return float(np.mean(vals))

def random_refit(raw,spec,base,seed):
    x=copy.deepcopy(raw);rng=np.random.default_rng(seed)
    # Reset for every object/seed; preserve the original raw parameter-list order.
    for p in x["parts"]:
        if p["part_id"]==spec["root_part_id"]:continue
        for field in ("parent_anchor","child_anchor"):
            a=rng.uniform(-.5,.5,3)
            a[int(rng.integers(0,3))]=.5 if rng.random()>=.5 else -.5
            p[field]=a.tolist()
    pars={p["part_id"]:p for p in x["parts"]}
    for e in spec["connections"]:
        par=base[e["parent_part_id"]]; child=base[e["child_part_id"]];p=pars[e["child_part_id"]]
        ap=np.asarray(p["parent_anchor"]); ac=np.asarray(p["child_anchor"]);rl=qmat(p["rotation"])
        p["offset"]=(par["r"].T@(child["c"]-par["c"])-par["s"]*ap+rl@(child["s"]*ac)).tolist()
    return x

def paired(rows,left,right):
    # Positive right-left improvement favors left. Same numerical-zero rule and
    # category ordering as methods.py paired_object_statistics.
    rows=sorted(rows,key=lambda r:r["object_id"])
    av=np.array([r[left] for r in rows]);bv=np.array([r[right] for r in rows]);d=bv-av
    d[np.abs(d)<=TOL]=0
    strata=defaultdict(list)
    for i,r in enumerate(rows):strata[r["category"]].append(i)
    rng=np.random.default_rng(42);boot=np.zeros(10000)
    for idx in strata.values():
        vals=d[idx]
        for start in range(0,10000,256):
            stop=min(start+256,10000);draw=rng.integers(0,len(vals),size=(stop-start,len(vals)))
            boot[start:stop]+=vals[draw].sum(axis=1)/len(rows)
    ci=np.quantile(boot,[.025,.975])
    return {"n":len(rows),"left":left,"right":right,"left_mean":float(av.mean()),"right_mean":float(bv.mean()),
            "mean_right_minus_left":float(d.mean()),"median_right_minus_left":float(np.median(d)),
            "ci_low":float(ci[0]),"ci_high":float(ci[1]),"wins_left":int((d>0).sum()),
            "ties":int((d==0).sum()),"losses_left":int((d<0).sum()),"tie_tolerance":TOL,
            "bootstrap_samples":10000,"statistics_seed":42,"p_values_computed":False}

start=time.time()
evaluation=load("partnet_results/experiments2/evaluation_all.json")
base_rows={str(r["object_id"]):r for r in evaluation["objects"] if r["method"]=="anchor"}
shared=load("partnet_results/experiments2/shared_specs.json")
print("shared_container",list(shared),flush=True)
specs={str(r["object_id"]):r["spec"] for r in shared["objects"]}
tasks=load("partnet_results/experiments3/edit_tasks.json")["tasks"]
old_geom=load("partnet_results/experiments3/geometry_results.json")["records"]
old={(str(r["object_id"]),r["method"]):r for r in old_geom if r["regime"]=="same_initial_geometry"}
old_stats=load("partnet_results/experiments3/final_results.json")
assert len(tasks)==877
assert {str(t["object_id"]) for t in tasks}==set(map(str,evaluation["complete_ids"]))
checks={"original_initial_max_error":0.,"random_initial_max_error":0.,"original_metric_max_error":0.,
        "original_closed_form_max_error":0.,"random_closed_form_max_error":0.,"random_initial_sat_max_error":0.,"original_after_abs_max_error":0.,"shared_before_abs_max_error":0.}
rows=[]
for i,t in enumerate(tasks):
    oid=str(t["object_id"]);spec=specs[oid];plan=t["expected_plan"]
    cached=load(f"partnet_results/experiments2/same_geometry/objects/{oid}.json")
    assert cached["normalization"]=={"center":[0.,0.,0.],"scale":1.}
    raw=cached["initial_representations"]["anchor"]
    base={p["part_id"]:{"c":np.asarray(p["center"],float),"r":np.asarray(p["world_rotation"],float),"s":np.asarray(p["bbox_size"],float)} for p in base_rows[oid]["layout_parts"]}
    before=decode(raw,spec);checks["original_initial_max_error"]=max(checks["original_initial_max_error"],max_layout_error(base,before))
    after=decode(edit_raw(raw,plan),spec)
    score=outgoing_drift(base,after,spec,plan)
    score_cf=outgoing_closed_form(base,raw,spec,plan)
    saved=old[(oid,"anchor")]["metrics"]["outgoing_margin_drift"]
    checks["original_metric_max_error"]=max(checks["original_metric_max_error"],abs(score-saved))
    checks["original_closed_form_max_error"]=max(checks["original_closed_form_max_error"],abs(score-score_cf))
    original_after_abs=outgoing_abs_margin(after,spec,plan)
    common_before_abs=outgoing_abs_margin(base,spec,plan)
    checks["original_after_abs_max_error"]=max(checks["original_after_abs_max_error"],abs(original_after_abs-old[(oid,"anchor")]["metrics"]["outgoing_after_abs_margin"]))
    for method in ["anchor","global","local"]:
        checks["shared_before_abs_max_error"]=max(checks["shared_before_abs_max_error"],abs(common_before_abs-old[(oid,method)]["metrics"]["outgoing_before_abs_margin"]))
    vals=[];after_abs_vals=[]
    for seed in SEEDS:
        rr=random_refit(raw,spec,base,seed);rb=decode(rr,spec)
        checks["random_initial_max_error"]=max(checks["random_initial_max_error"],max_layout_error(base,rb))
        for e in spec["connections"]:
            p,c=e["parent_part_id"],e["child_part_id"]
            checks["random_initial_sat_max_error"]=max(checks["random_initial_sat_max_error"],abs(sat(base[p],base[c])-sat(rb[p],rb[c])))
        ra=decode(edit_raw(rr,plan),spec)
        drift=outgoing_drift(base,ra,spec,plan);cf=outgoing_closed_form(base,rr,spec,plan)
        checks["random_closed_form_max_error"]=max(checks["random_closed_form_max_error"],abs(drift-cf))
        vals.append(drift)
        after_abs_vals.append(outgoing_abs_margin(ra,spec,plan))
    row={"object_id":oid,"category":t["category"],"original_anchor":score,"original_anchor_saved":saved,
         "random_refit_anchor":float(np.mean(vals)),"random_seed_scores":dict(zip(map(str,SEEDS),vals)),
         "matched_global":old[(oid,"global")]["metrics"]["outgoing_margin_drift"],
         "matched_local":old[(oid,"local")]["metrics"]["outgoing_margin_drift"],
         "axis":plan["axis"],"factor":float(plan["scale_factor"]),"target_part_id":plan["target_part_id"],
         "outgoing_before_abs_margin":common_before_abs,"original_after_abs_margin":original_after_abs,
         "random_refit_after_abs_margin":float(np.mean(after_abs_vals)),
         "random_refit_seed_after_abs_margins":dict(zip(map(str,SEEDS),after_abs_vals)),
         "matched_global_after_abs_margin":old[(oid,"global")]["metrics"]["outgoing_after_abs_margin"],
         "matched_local_after_abs_margin":old[(oid,"local")]["metrics"]["outgoing_after_abs_margin"]}
    rows.append(row)
    if (i+1)%100==0:print("completed",i+1,flush=True)
assert all(v<=TOL for v in checks.values()),checks
rep=paired(rows,"original_anchor","matched_global")
oldstat=next(r for r in old_stats["paired_statistics"] if r["regime"]=="same_initial_geometry" and r["subset"]=="intended_plan_full" and r["baseline"]=="global" and r["metric"]=="outgoing_margin_drift")
for k in ["ci_low","ci_high"]:assert abs(rep[k]-oldstat[k])<1e-12,(k,rep[k],oldstat[k])
assert (rep["wins_left"],rep["ties"],rep["losses_left"])==(oldstat["wins"],oldstat["ties"],oldstat["losses"])
comparisons=[paired(rows,"original_anchor","random_refit_anchor"),paired(rows,"random_refit_anchor","matched_global"),rep]
category=[]
for cat in sorted(set(r["category"] for r in rows)):
    sub=[r for r in rows if r["category"]==cat]
    category.append({"category":cat,"n":len(sub),**{k:float(np.mean([r[k] for r in sub])) for k in ["original_anchor","random_refit_anchor","matched_global"]}})
result={"protocol":{"analysis":"New post-hoc matched-initial-geometry anchor replacement control; offline only",
  "cohort":"Original 877 PartNet complete cases; one frozen intended parent edit each",
  "intervention":"Randomize both edge anchors, refit offsets to recover all original world OBBs, then apply frozen single-axis parent resize",
  "seed_policy":"For each object and seed, reset numpy.default_rng(seed); traverse original raw parts order; skip root; sample parent_anchor then child_anchor using uniform(-.5,.5,3), integers(0,3) face axis, random() >= .5 face sign",
  "seeds":SEEDS,"aggregation":"Outgoing-edge absolute margin drifts averaged within object and seed, then five seeds averaged within object; equal object means",
  "normalization":"Original Anchor normalized frame fixed; no post-edit normalization",
  "statistical_unit":"object, not object-seed","confidence_interval":"10,000 category-stratified paired percentile bootstrap replicates; seed 42; no p values",
  "scope":"Conditional on original geometry and orientations; offsets unconstrained after refitting; child-anchor randomness cancels from local outgoing-edge response for this task"},
 "software":{"numpy":np.__version__},"checks":checks,"original_mean_reproduction":{"recomputed":float(np.mean([r['original_anchor'] for r in rows])),"saved":float(np.mean([r['original_anchor_saved'] for r in rows]))},
 "seed_aggregate_means":{str(s):float(np.mean([r['random_seed_scores'][str(s)] for r in rows])) for s in SEEDS},
 "absolute_margin_summary":{"same_target_outgoing_before_mean":float(np.mean([r["outgoing_before_abs_margin"] for r in rows])),
   "original_after_mean":float(np.mean([r["original_after_abs_margin"] for r in rows])),
   "random_refit_after_mean":float(np.mean([r["random_refit_after_abs_margin"] for r in rows])),
   "matched_global_after_mean":float(np.mean([r["matched_global_after_abs_margin"] for r in rows])),
   "matched_local_after_mean":float(np.mean([r["matched_local_after_abs_margin"] for r in rows]))},
 "comparisons":comparisons,"category_summary":category,"objects":rows,"elapsed_seconds":time.time()-start}
OUT.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({"output":str(OUT),"checks":checks,"comparisons":comparisons,"seed_aggregate_means":result["seed_aggregate_means"],"absolute_margin_summary":result["absolute_margin_summary"],"elapsed_seconds":result["elapsed_seconds"]},indent=2))
