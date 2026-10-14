# AnchorTree / ASPT

Research code and frozen evaluation materials for **AnchorTree: Explicit Attachment Parameterization for Language-Guided Structured 3D Object Generation**. The evaluated representations are oriented bounding boxes (OBBs), not reconstructed meshes. Attachment scores and resizing scores are box-level proxies.

The 2026-10-14 data release adds the complete benchmark inputs, recorded model requests, generated representations, object-level evaluation records, and the additional offline parent-edit controls. It contains no newly generated model outputs.

## Contents

| Material | Location | Coverage |
| --- | --- | --- |
| PartNet sample manifest | `partnet_benchmark_manifest.json` | 880 objects; 80 in each of 11 categories |
| PartNet descriptions and ordered part inventories | `partnet_descriptions_880.json` | All 880 descriptions; preserve `source.parts` order |
| Derived reference adjacency | `partnet_physical_adjacency_880.json` | 880 objects; identifiers, semantic names, distances and adjacency summaries; no meshes |
| Frozen PartNet structure import | `partnet_shared_part_specs_880_final.json` | 880 records, including the three unsuccessful structures |
| OpenWorld inputs | `openworld_objects_120.json` | All 120 descriptions in six groups |
| Explicit structure inputs and scoring references | `explicit_structure_benchmark_90.json` | 30 base structures, three paraphrases each; `gt_spec` is an evaluation reference, not a model input |
| Actual recorded requests | `data/prompts/recorded_requests.jsonl.gz` | 4,931 saved request attempts, including retries; see the provenance limitation below |
| Frozen runs and evaluation records | `data/cached_runs/` | Five ZIP archives, preserving original relative paths |
| Convenient object-level CSVs | `data/object_results/` | Exports from the frozen records, plus numerical verification |
| Additional controls | `data/additional_controls/` | Original object-level outputs for size-aware local translations and geometry-preserving anchor randomization |
| File integrity and provenance | `data/manifest.json` | SHA-256 digests, archive membership, row counts and sanitization notes |
| Code | `methods.py`, `experiments1.ipynb` through `experiments7.ipynb`, `summary.ipynb` | Original implementation and notebooks |

See [data/README.md](data/README.md) for schemas, cohorts and the mapping to the manuscript tables.

## Reproduce the published results from frozen records

Use Python 3.11 or newer (validated with Python 3.12.14 and NumPy 2.3.5). The integrity check and object-level CSV export use only the Python standard library. The two additional control analyses also require NumPy.

```bash
python scripts/unpack_cached_runs.py
python scripts/export_object_results.py
python -m pip install -r requirements-reproduce.txt
python scripts/reproduce_size_aware_control.py
python scripts/reproduce_anchor_reparameterization_control.py
```

These commands do not call a model service or require original PartNet meshes. The first command verifies every archived file before restoring the experiment directories. It refuses to overwrite a different existing local file. The export creates 18 CSV files and `data/object_results/verification.json`; the control analyses write their recomputed reports to `reproduced/`. Published reference results in `data/additional_controls/` are not overwritten. The release was independently rerun from the extracted archives: all 36,241 compared numeric fields in the two control reports matched their reference values exactly (`data/reproduction_check.json`). See the scripts' `--help` for separate input/output directories.

To check the downloaded release without extracting it:

```bash
python scripts/unpack_cached_runs.py --verify-only
```

The original notebooks are retained as experiment records. They include machine-specific configuration and are not the offline reproduction entry point. Re-running generation or original mesh preprocessing requires separate model-service configuration and access to the original dataset. The DeepSeek client reads an explicit `api_key` argument or the `DEEPSEEK_API_KEY` environment variable. Never put API credentials in tracked files. A provider alias and temperature of zero do not guarantee the same output in a future service call.

## Scope and provenance

- PartNet has 880 selected inputs and 877 complete geometry cases. Chair 37772, Chair 42511 and Table 34471 fail shared-structure generation. Lamp 17559 has no reference adjacency edges and is excluded only from the corresponding reference-edge metric (876 cases).
- OpenWorld has 120 inputs and 119 complete cases; OW107 fails shared-structure generation.
- The backbone comparison uses 121 PartNet objects and 117 common OpenWorld objects. The latter excludes Qwen AnchorTree failures OW012 and OW116. All available records and failure statuses are retained; paired summaries use the common cohort.
- Parent editing uses 877 frozen intended plans. The parser is correct on 855; geometric comparisons execute the intended plans for all 877 cases. Child-size perturbations and parent-size edits are different interventions.
- Explicit structure recovery contains 90 prompts per backbone, grouped into 30 base structures. Exact agreement is based on semantic-name multisets and the root name, not full instance-level graph isomorphism.
- The additional controls are post-hoc analyses of existing cached layouts. They preserve initial geometry and make no model calls. Five anchor-randomization seeds are averaged within each object.

**Historical logging limitation.** The imported PartNet shared-structure cache preserves 880 records and 901 attempts, but the original raw responses and request messages for that stage were not retained. These records explicitly have `raw_response_available=false` and `raw_text_reconstructed=true`; reconstructed strings are not presented as original model responses. The complete benchmark descriptions and frozen structure results are supplied. The request JSONL contains only requests actually saved in the other stage caches, with archive, source-file and attempt identifiers.

In the released JSON inputs, run caches and control results, absolute workstation paths were replaced with repository-relative references or `<LOCAL_PATH>/basename`. Numerical results, semantic inputs, random seeds, model identifiers, failure statuses and recorded messages are preserved. Source-byte hashes and exported-byte hashes are distinct: original run fingerprints describe the original input files and will not necessarily match the sanitized JSON serialization. The release manifest verifies the distributed bytes.

## Data sources and use

This repository distributes study-specific prompts, benchmark identifiers, generated part representations, cached outputs and derived evaluation records. Raw PartNet/ShapeNet meshes, textures, point clouds, original annotation archives and the local reference-geometry cache are not included. PartNet-derived descriptions and adjacency records retain dataset identifiers for provenance; they are not represented as wholly independent datasets.

Obtain original PartNet data from the [official PartNet archive](https://huggingface.co/datasets/ShapeNet/PartNet-archive), subject to the provider's access requirements and terms of use. Consult the [PartNet project](https://partnet.cs.stanford.edu/) and cite PartNet and ShapeNet when using the underlying dataset. This release does not grant new rights to third-party data or add a new software/data license.

- Mo et al. *PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding*. CVPR 2019.
- Chang et al. *ShapeNet: An Information-Rich 3D Model Repository*. arXiv:1512.03012, 2015.
