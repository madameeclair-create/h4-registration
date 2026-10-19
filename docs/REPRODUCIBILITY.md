# Reproduction protocol and release validation

## Scope

The release adapter runs the measured MAC-based backend from existing descriptor correspondences. It preserves the scientific preprocessing, geometric scoring, action-distance selection and rank-fusion mechanics. Retired development launchers and absolute experiment paths were removed. The manuscript's experiments were already completed; preparing this release does not constitute a new full-benchmark experiment.

The test-pair manifests fix 6808 indoor dataset/descriptor cases and 1110 KITTI cases. `--per-group 1` selects the first entry of each dataset/descriptor group, before observing any result. Execution failures and cases with no usable hypotheses remain in RR's denominator. RE and TE means use successful cases only. Strict success thresholds are documented in METHOD.md.

## Main experiment

```bash
python run.py --check
python run.py
```

The full method is `diverse_scale_rank`. Other variants in `all_controls.csv` provide intermediate diagnostics on shared hypotheses. They are not independent reruns of all baseline papers. The separate MAC Top1 control is used to verify refinement equivalence; do not present it as every detail of the published MAC baseline.

The source manifests contain SHA-256 hashes of point clouds, correspondences and ground-truth matrices. Hashing the ground-truth file checks input identity; its numerical values are not passed into generation, DS, refinement or scoring. The native generator receives an identity placeholder and all-zero labels. Actual ground truth is read after all selections.

## Strict component ablations and hypothesis coverage

```bash
python experiments/analyze.py outputs/YOUR_COMPLETED_MAIN_RUN
```

The input run retains the union of conventional Top16 and diversity-selected Top16. This union was refined once during the main run. The analysis reuses those exact poses and the same geometry caches:

- `without_diversity`: conventional shortlist, with the rest of scoring unchanged.
- `forward_only`: source-to-target support, preserving normals and density weights.
- `without_normal`: remove only the normal gate; retain density weights.
- `without_density_balance`: remove only density weights; retain the normal gate.
- `without_multiscale`: fixed central scale, 0.10 m indoors / 0.60 m on KITTI.
- `budget_top1`, `budget_top4`, `budget_top8`: prefixes of the same diversity shortlist; `full_h4` is Top16.

Full-method and single-scale candidate IDs must agree with the main run. Coverage uses ground truth offline to ask whether at least one available hypothesis passes both RE and TE criteria. It is not the fraction of correct hypotheses and not actual registration recall. Because refinement changes poses, the final-selection upper bound is the coverage of the refined shortlist being scored.

For the manuscript's indoor ablation and coverage tables, analyze a completed `--scope indoor` run. KITTI analysis is additionally supported but is not presented as a new manuscript table.

## Five-stage input-size timing

```bash
python scripts/build_timing.py --jobs 2
python experiments/timing.py --output outputs/timing_full
```

Default sizes: **250, 500, 1000, 2500, 5000**. The protocol uses all eight 3DMatch scenes and both descriptors. Within each descriptor/scene, pairs are sorted lexicographically by `pair_id`; the first pair containing at least 5000 correspondence rows is selected. Sampling uses `np.random.default_rng(24080 + int(original_case_id)).permutation(M)` and nested first-N subsets, without replacement. Manifest case IDs preserve FPFH rows 0–3403 followed by FCGF rows 3404–6807.

Each condition is repeated three times after one warm-up at the first condition per descriptor: **80 conditions and 240 timed runs**. Execution is serial and the native computational thread limit is one. Results average repetitions within each pair, then average pair means across scenes. Repetitions must yield identical candidate-cache and selected-pose hashes.

The source under `cpp/timing/` adds timestamp hooks to the frontend and refinement tool without changing the selection parameters:

1. **Point cloud preprocessing:** fresh geometry loading, voxel downsampling, normal estimation, sampling and density weights.
2. **Graph construction:** compatibility graph, graph-related auxiliaries and igraph conversion up to the clique-search call.
3. **Maximal-clique search:** igraph maximal-clique enumeration call.
4. **Clique and hypothesis selection:** native node-guided selection and cleanup plus action-space diversity filtering.
5. **Pose estimation and evaluation:** native pose stage plus correspondence rescoring, cache/file conversion, weighted refinement, whole-cloud scores and rank fusion.

`H4_total_ms` sums the **unrounded five stage times**. It excludes feature extraction, initial matching, pre-graph native input reading, cross-system startup/scheduling, diagnostic serialization after the native timer, and evaluation against real ground truth. `external_wall_seconds` is retained separately. MAC totals in the timing output contain four instrumented native phases recorded within these same runs, not a separate end-to-end MAC timing experiment.

For a short execution check only:

```bash
python experiments/timing.py --output outputs/timing_smoke --sizes 250 --scene-limit 1 --repeats 2
```

Reduced sizes, scene counts or repetitions are explicitly marked as a smoke test in `protocol.json`. Their timings must not be substituted for the paper's full timing table. Runtime is hardware- and environment-dependent.

## Validation limits

The release was compiled from source on the existing Windows/Ubuntu-WSL environment and exercised on a six-pair main-run smoke test, a cached-analysis smoke test, and a reduced timing run. The synthetic protocol checks do not require benchmark downloads. The release package excludes data, build outputs, local configuration and run caches. See `validation.json` for completed checks and their exact scope.

Native Linux execution is supported by the platform adapter but was not independently exercised on a second Linux-only installation. The package does not claim a full clean-machine or full-benchmark rerun. Source provenance distinguishes the original measured code from portability changes.
