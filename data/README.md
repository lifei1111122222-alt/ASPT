# Frozen inputs, requests and object-level results

## Archives and manuscript mapping

Each ZIP extracts paths relative to the repository root. `scripts/unpack_cached_runs.py` verifies archive and member SHA-256 digests against `manifest.json` and restores these paths.

| Archive | Restored paths | Manuscript evidence |
| --- | --- | --- |
| `cached_runs/partnet_primary.zip` | `partnet_results/experiments2/` | Tables 1, 2, 3 and 9: primary generation, fixed-offset anchor replacement, child-size perturbation, matched initial geometry and object-level outcomes |
| `cached_runs/parent_edit.zip` | `partnet_results/experiments3/` | Tables 4 and 9: intended edit tasks, parser outcomes and geometry results under both initialization regimes |
| `cached_runs/openworld_primary.zip` | `openworld_results/` | Tables 2, 3, 6 and 9: primary OpenWorld evaluation and perturbations |
| `cached_runs/backbone.zip` | `experiments6_results/qwen3_7_plus/` | Table 7 and backbone figure: secondary generation, shared-cohort comparisons and child-size perturbations |
| `cached_runs/explicit_structure.zip` | `experiments7_results/` | Table 8: 90 prompts for each backbone, grouped by 30 base structures |
| `additional_controls/*.json` | Already available without extraction | Table 5: object-level size-aware local and random-anchor/refitted-offset analyses |

The PartNet statistical source used by the manuscript is `tables/results_tie_corrected.json` and its companion `paired_statistics_tie_corrected.csv`. The older uncorrected statistical summary is deliberately not distributed as the authoritative result. The object measurements themselves are unchanged.

## Principal schemas

- Root benchmark files use `objects` arrays. Preserve `object_id`, `category`, descriptions, ordered inventories and explicit-structure `base_structure_id`/`paraphrase_id` fields. `gt_spec` belongs to evaluation only.
- `shared_specs.json` retains generated/frozen semantic trees and failed cases. PartNet imported history is marked reconstructed, as explained in the root README.
- Generation and parser caches contain `status`, `attempt_count`, `attempts`, per-attempt validation and response text, request configuration, recorded messages where available, and accepted `result` objects. Provider aliases and token-use metadata are retained.
- `evaluation_all.json` contains `objects` rows and `complete_ids` (where applicable). A row identifies one object and method, with metrics and generated OBB layouts. The object is the statistical unit; parts or edges are not independent samples.
- `perturbation_all.json` retains object/method records and per-scale conditions, including edge-level intervention measurements. CSV exports keep one row per object, method and scale after the original within-object aggregation.
- `random_anchor_all.json` retains per-object/per-seed measurements; five seeds are averaged within an object for reported comparisons.
- `same_geometry/objects/` contains the 877 common initial parameterizations and reconstruction checks. These files enable the two additional controls without original meshes.
- `edit_tasks.json` has the English instructions and frozen `expected_plan`. Expected plans are scoring/execution references, not extra inputs supplied to the language parser. `geometry_results.json` has 5,262 object/regime/method rows (877 objects × two initialization regimes × three representations).
- `prompts/recorded_requests.jsonl.gz` has one JSON object per saved request attempt: `archive`, `cache_file`, `attempt`, `request_config`, `returned_model`, and `messages`. Messages are the recorded role/content lists, including retry context when saved. It is a convenience export of the archive contents, not a reconstruction of missing historical requests.
- `additional_controls/size_aware_local_control.json` and `matched_anchor_reparameterization.json` each contain 877 object records, aggregate means, numerical checks and statistical summaries. The latter also records the five individual seed scores.

## Cohort and metric interpretation

Primary PartNet: 877 complete objects; reference-edge margin MAE: 876. Primary OpenWorld: 119. Backbone PartNet: 121. Backbone OpenWorld: 117 common complete objects; the secondary all-available file contains 355 rows, whereas a three-method paired cohort contains 351. Explicit recovery: 90 prompts per model; confidence calculations must group paraphrases by their 30 base structures.

SAT deviation is an average absolute separating-axis margin on the shared tree. Overlap fraction uses all unordered pairs and does not establish mesh collision. Edit drift is the absolute change from an initial margin and does not independently establish a correct contact. Object-level CSVs retain method, dataset/category, scale, initialization regime and target information where relevant. Distances use the frozen normalized object frame; interventions are not renormalized.

Wins/ties/losses use a tolerance of 1e-9 on baseline-minus-AnchorTree error. Paired statistical summaries retain the original category-stratified bootstrap, seed settings and Holm correction families. The release performs no new model sampling or selective exclusion of failed outputs.

`object_results/verification.json` records the counts, means and object-level outcomes checked against the manuscript. It verifies reproduction from these frozen records, not external validity or future model reproducibility.
