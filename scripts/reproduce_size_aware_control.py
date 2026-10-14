from pathlib import Path
import argparse, ast, copy, hashlib, itertools, json, warnings, __future__
from collections import Counter, defaultdict
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(
    description='Reproduce the size-aware local control from frozen caches; no API calls.'
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
OUT = args.output_dir.expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
source = REPO_ROOT / 'methods.py'
source_text = source.read_text(encoding='utf-8-sig')
needed = {'quaternion_to_matrix', 'validate_partnet_representation', 'partnet_box_corners', 'transform_partnet_representation', 'obb_sat_margin', '_edit3_apply_raw'}
module = ast.parse(source_text)
selected = ast.Module(body=[node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in needed], type_ignores=[])
assert len(selected.body) == len(needed)
_BOX_SIGNS = np.asarray(list(itertools.product((-0.5, 0.5), repeat=3)), dtype=float)
exec(compile(selected, str(source), 'exec', flags=__future__.annotations.compiler_flag), globals())

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

paths = {
    'tasks': ROOT / 'partnet_results/experiments3/edit_tasks.json',
    'saved_geometry': ROOT / 'partnet_results/experiments3/geometry_results.json',
    'shared_specs': ROOT / 'partnet_results/experiments2/shared_specs.json',
    'same_geometry_directory': ROOT / 'partnet_results/experiments2/same_geometry/objects',
}
tasks = read(paths['tasks'])['tasks']
specs = {str(r['object_id']): r['spec'] for r in read(paths['shared_specs'])['objects'] if r['status'] == 'valid'}
saved = {(str(r['object_id']), r['method']): r['metrics']['outgoing_margin_drift'] for r in read(paths['saved_geometry'])['records'] if r['regime'] == 'same_initial_geometry'}
saved_full = {(str(r['object_id']), r['method']): r['metrics'] for r in read(paths['saved_geometry'])['records'] if r['regime'] == 'same_initial_geometry'}
assert len(tasks) == 877 and len({str(t['object_id']) for t in tasks}) == 877

identity = {'center': [0., 0., 0.], 'scale': 1.}

def index(layout):
    return {p['part_id']: p for p in layout['parts']}

def maximum_error(a, b):
    return max(float(np.max(np.abs(np.asarray(a[pid][field]) - np.asarray(b[pid][field])))) for pid in a for field in ('center', 'bbox_size', 'world_rotation'))

def score(before, after, outgoing):
    return float(np.mean([abs(obb_sat_margin(after[p], after[c]) - obb_sat_margin(before[p], before[c])) for p, c in outgoing]))

def absolute_margin(layout, outgoing):
    return float(np.mean([abs(obb_sat_margin(layout[p], layout[c])) for p, c in outgoing]))

def encode_size_aware(before, spec, local_raw):
    encoded = copy.deepcopy(local_raw)
    by_id = {p['part_id']: p for p in encoded['parts']}
    coefficients = {}
    for edge in spec['connections']:
        p, c = edge['parent_part_id'], edge['child_part_id']
        parent, child = before[p], before[c]
        t = np.asarray(parent['world_rotation']).T @ (np.asarray(child['center']) - np.asarray(parent['center']))
        coefficients[c] = (t / np.asarray(parent['bbox_size'])).tolist()
    return encoded, coefficients

def size_aware_decode(encoded, coefficients, spec, plan=None):
    edited = _edit3_apply_raw(encoded, plan) if plan is not None else copy.deepcopy(encoded)
    by_id = {p['part_id']: p for p in edited['parts']}
    for edge in spec['connections']:
        p, c = edge['parent_part_id'], edge['child_part_id']
        by_id[c]['relative_position'] = (np.asarray(by_id[p]['bbox_size']) * np.asarray(coefficients[c])).tolist()
    return index(transform_partnet_representation(edited, spec, 'local', identity))

records = []
cache_digest = hashlib.sha256()
for k, task in enumerate(tasks):
    oid = str(task['object_id'])
    spec, plan = specs[oid], task['expected_plan']
    path = paths['same_geometry_directory'] / (oid + '.json')
    cache_bytes = path.read_bytes()
    cache_digest.update(oid.encode() + b'\0' + hashlib.sha256(cache_bytes).digest())
    same = json.loads(cache_bytes.decode('utf-8-sig'))
    assert same['normalization'] == identity
    raw = same['initial_representations']
    before = index(transform_partnet_representation(raw['global'], spec, 'global', identity))
    outgoing = [(e['parent_part_id'], e['child_part_id']) for e in spec['connections'] if e['parent_part_id'] == plan['target_part_id']]
    assert outgoing
    scores, errors, reproduction, after_abs = {}, {}, {}, {}
    before_abs = absolute_margin(before, outgoing)
    for name in ('anchor', 'local', 'global'):
        initial = index(transform_partnet_representation(raw[name], spec, name, identity))
        errors[name] = maximum_error(initial, before)
        after = index(transform_partnet_representation(_edit3_apply_raw(raw[name], plan), spec, name, identity))
        scores[name] = score(before, after, outgoing)
        after_abs[name] = absolute_margin(after, outgoing)
        reproduction[name] = abs(scores[name] - saved[(oid, name)])
    encoded, coefficients = encode_size_aware(before, spec, raw['local'])
    initial = size_aware_decode(encoded, coefficients, spec)
    errors['size_aware_local'] = maximum_error(initial, before)
    size_aware_after = size_aware_decode(encoded, coefficients, spec, plan)
    scores['size_aware_local'] = score(before, size_aware_after, outgoing)
    after_abs['size_aware_local'] = absolute_margin(size_aware_after, outgoing)
    # Independent pairwise cross-check: SAT is unchanged by common translation.
    # Keep the target center fixed and shift each direct child by Rp * (delta_sp * u).
    direct = copy.deepcopy(before)
    pid = plan['target_part_id']
    axis = {'x': 0, 'y': 1, 'z': 2}[plan['axis']]
    delta_size = np.zeros(3)
    delta_size[axis] = np.asarray(before[pid]['bbox_size'])[axis] * (plan['scale_factor']-1.)
    direct[pid]['bbox_size'] = (np.asarray(before[pid]['bbox_size']) + delta_size).tolist()
    for p, c in outgoing:
        direct[c]['center'] = (np.asarray(before[c]['center']) + np.asarray(before[p]['world_rotation']) @ (delta_size * np.asarray(coefficients[c]))).tolist()
    direct_error = max(abs(scores['size_aware_local'] - score(before, direct, outgoing)), abs(after_abs['size_aware_local'] - absolute_margin(direct, outgoing)))
    absolute_metric_error = max(max(abs(before_abs - saved_full[(oid, name)]['outgoing_before_abs_margin']), abs(after_abs[name] - saved_full[(oid, name)]['outgoing_after_abs_margin'])) for name in ('anchor','local','global'))
    assert absolute_metric_error <= 1e-12, (oid, absolute_metric_error)
    assert direct_error <= 1e-12, (oid, direct_error)
    assert max(errors.values()) <= 1e-9, (oid, errors)
    assert max(reproduction.values()) <= 1e-9, (oid, reproduction)
    records.append({'object_id': oid, 'category': task['category'], 'target_is_root': task['target_is_root'], 'axis': plan['axis'], 'factor': plan['scale_factor'], 'outgoing_edges': len(outgoing), 'scores': scores, 'before_mean_abs_margin': before_abs, 'after_mean_abs_margin': after_abs, 'initialization_max_error': errors, 'original_score_abs_error': reproduction, 'independent_pairwise_formula_error': direct_error, 'original_absolute_metric_error': absolute_metric_error})
    if (k+1) % 200 == 0:
        print(f'Validated {k+1}/{len(tasks)} objects', flush=True)

def comparison(rows, baseline, nboot=10000, seed=42):
    ordered = sorted(rows, key=lambda r: r['object_id'])
    av = np.asarray([r['scores']['anchor'] for r in ordered])
    bv = np.asarray([r['scores'][baseline] for r in ordered])
    difference = bv-av
    difference[np.abs(difference) <= 1e-9] = 0.
    strata = defaultdict(list)
    for i, row in enumerate(ordered):
        strata[row['category']].append(i)
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(nboot)
    for indices in strata.values():
        values = difference[indices]
        for start in range(0, nboot, 256):
            stop = min(start+256, nboot)
            draw = rng.integers(0, len(values), size=(stop-start, len(values)))
            bootstrap[start:stop] += values[draw].sum(axis=1)/len(ordered)
    interval = np.quantile(bootstrap, [.025, .975])
    return {'n': len(ordered), 'anchor_mean': float(av.mean()), 'baseline_mean': float(bv.mean()), 'baseline_minus_anchor_mean': float(difference.mean()), 'raw_mean_difference': float((bv-av).mean()), 'ci95_low': float(interval[0]), 'ci95_high': float(interval[1]), 'anchor_wins': int((difference>0).sum()), 'ties': int((difference==0).sum()), 'anchor_losses': int((difference<0).sum()), 'relative_reduction_percent': float(100*(bv.mean()-av.mean())/bv.mean())}

categories = sorted({r['category'] for r in records})
result = {
    'protocol': {'cohort': 'Original 877 complete PartNet cases; exact AnchorTree-initialized shared geometry', 'edit': 'Frozen intended parent local-axis resize, factors 0.8 or 1.25, same tasks and no new generation', 'baseline': 'u_pc = R_p.T (c_c-c_p) / s_p; c_c = c_p + R_p (s_p * u_pc); fixed u, relative rotation, untargeted sizes, root pose; recursive tree decode', 'metric': 'Mean absolute outgoing-edge SAT margin change within object, then mean across objects; original normalized frame fixed', 'statistics': 'Paired object differences baseline minus AnchorTree; numerical differences <=1e-9 treated as zero; category-stratified percentile bootstrap, 10000 replicates, seed 42; win/tie/loss is AnchorTree perspective', 'timing': 'Post-hoc offline control; not a prespecified original experiment', 'no_new_generation': True},
    'sources': {k: str(v) for k,v in paths.items()},
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, paths['tasks'], paths['saved_geometry'], paths['shared_specs']]},
    'matched_cache_digest_in_task_order': cache_digest.hexdigest(),
    'n': len(records),
    'validation': {'maximum_original_absolute_metric_error': max(r['original_absolute_metric_error'] for r in records), 'maximum_independent_pairwise_formula_error': max(r['independent_pairwise_formula_error'] for r in records), 'maximum_initialization_error': max(max(r['initialization_max_error'].values()) for r in records), 'maximum_original_metric_error': max(max(r['original_score_abs_error'].values()) for r in records), 'original_anchor_saved_mean': float(np.mean([saved[(r['object_id'],'anchor')] for r in records])), 'original_fixed_local_saved_mean': float(np.mean([saved[(r['object_id'],'local')] for r in records]))},
    'means': {name: float(np.mean([r['scores'][name] for r in records])) for name in ('anchor','local','global','size_aware_local')},
    'absolute_margin_means': {'before_all_methods': float(np.mean([r['before_mean_abs_margin'] for r in records])), 'after': {name: float(np.mean([r['after_mean_abs_margin'][name] for r in records])) for name in ('anchor','local','global','size_aware_local')}},
    'comparisons': {name: comparison(records, name) for name in ('local','size_aware_local')},
    'by_category': [{'category': cat, **comparison([r for r in records if r['category']==cat], 'size_aware_local')} for cat in categories],
    'by_target_type': [{'target_is_root': root, **comparison([r for r in records if r['target_is_root']==root], 'size_aware_local')} for root in (True,False)],
    'objects': records,
}
report = OUT/'size_aware_local_control.json'
report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('objects','sources','source_sha256')}, ensure_ascii=False, indent=2))
print('REPORT_PATH='+str(report))
