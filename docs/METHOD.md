# Method to code map

The pipeline is **MAC hypothesis generation → DS shortlist → weighted refinement → whole-cloud geometry → multiscale ranking → evaluation**.

## Inherited frontend

`cpp/producer.cpp` runs the MAC registration adapter with generator seed 123, captures pose/clique provenance, and establishes a deterministic pose ordering. `cpp/PCR_provenance.cpp` records hypotheses. `cpp/registration_initialized.cpp` supplies the initialized adapter used in the experiments. The vendored MAC routines are in `third_party/mac/Linux/`.

These files preserve the upstream graph construction, maximal-clique search, node-guided clique selection, and initial SVD mechanics. Adapter instrumentation and initialization do not make those mechanics new contributions. Provenance is documented in `source_provenance.json`.

## Section 3.3 Action-space diversity

Find **本文创新代码起点** in `h4_registration/core_indoor.py` or `core_kitti.py`. Given source sample mean μ and covariance Σ, each pose is embedded as `[vec(R sqrt(Σ)), R μ + t]`. The Euclidean distance of embeddings equals the RMS displacement difference on the same source samples. Candidates are visited by descending correspondence support and retained only if their distance from every retained hypothesis reaches the threshold. The budget is at most 16.

## Section 3.4 Refinement

`cpp/weighted_refine_indoor.cpp` and `weighted_refine_kitti.cpp` implement residual-weighted centroids and cross-covariance, weighted SVD, rank-degeneracy checks, and monotonic acceptance using the original truncated correspondence-support sum. Correspondence associations stay fixed. At most 20 iterations are attempted. This is a supporting refinement procedure with established estimation foundations.

## Section 3.5 Whole-cloud geometry

`preprocess()` uses voxel downsampling, 20-neighbor normal estimation, coordinate-lexicographic ordering, and at most 4096 equally spaced sample indices. Normal-density weights are inverse square roots of neighborhood counts in unoriented normal space, normalized to mean one.

`geometry()` computes forward and backward nearest-neighbor support. Valid normals must satisfy an absolute-dot-product gate. Density weights belong to the query side. Samples outside the distance threshold or failing normal checks contribute zero but stay in the averaging denominator. The two directions are combined by harmonic mean.

## Section 3.6 Rank fusion

Find **新增模块 MS** in the two core modules. At each of five length scales, descending geometric scores receive average ranks for ties. Final selection minimizes worst rank, then mean rank, then maximizes refined correspondence support, then minimizes the original candidate ID.

## Parameters

| Parameter | Indoor | KITTI |
|---|---:|---:|
| Maximum retained hypotheses | 16 | 16 |
| Action distance threshold (m) | 0.10 | 0.60 |
| Refinement distance threshold (m) | 0.10 | 0.60 |
| Voxel size (m) | 0.05 | 0.30 |
| Geometry radii (m) | 0.05, 0.075, 0.10, 0.15, 0.20 | 0.30, 0.45, 0.60, 0.90, 1.20 |
| Normal consistency angle | 30° | 30° |
| Normal-density angle | 20° | 20° |
| Normal-estimation neighbors | 20 | 20 |
| Maximum samples per cloud | 4096 | 4096 |
| Maximum refinement iterations | 20 | 20 |
| MAC generator seed | 123 | 123 |
| Success criteria | RE ≤ 15° and TE ≤ 30 cm | RE ≤ 5° and TE ≤ 60 cm |

KITTI length scaling was established through benchmark development diagnostics; it is not an unseen-domain generalization claim. The source is annotated in Chinese at the innovation boundaries and major mathematical steps.
