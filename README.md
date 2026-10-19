# Point Cloud Registration via Action-Space Diversity and Multiscale Geometric Evaluation

Research code for **Point Cloud Registration via Action-Space Diversity of Pose Hypotheses and Multiscale Geometric Evaluation**.

**Authors:** Minghu Wu, Yiming Zhang, Xiang Tu, and Juan Wang.

[Numerical tables and Figure 2 source data](paper_data/supplementary_data_1/) · [中文说明](README_zh.md) · [Data preparation](docs/DATA.md) · [Method-to-code map](docs/METHOD.md) · [Reproduction protocol](docs/REPRODUCIBILITY.md)

This implementation keeps the maximal-clique hypothesis-generation frontend of [MAC](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques). It adds action-space diversity selection, weighted hypothesis refinement, bidirectional whole-cloud geometric evaluation, normal-density balancing, and multiscale rank fusion. It does not train a feature extractor.

## Quick start

The measured environment uses **Python 3.11 on Windows with Ubuntu 24.04 in WSL** for the C++ tools. The launcher also supports native Linux, but native Linux end-to-end numerical equivalence has not been established. Install PCL, Eigen, igraph, CMake, a C++17 compiler, and OpenMP in Linux/WSL first.

```bash
# Run inside Ubuntu 24.04 / WSL; sudo may ask for your system password.
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libpcl-dev libigraph-dev libeigen3-dev
```

Use a Python 3.11 environment on the host that will run the Python entry point:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/build.py --jobs 2
```

Copy `config.example.json` to `config.local.json` and set the two data roots. `config.local.json` is ignored by Git. See [DATA.md](docs/DATA.md) for required files and download sources. The package does **not** include benchmark point clouds or extracted correspondences.

```bash
python run.py --plan                 # Print all six groups; no data or binaries required
python run.py --check                # Check dependencies and input file presence
python run.py --per-group 1          # Fixed first pair in each group: six pairs total
python run.py                        # Full benchmark: 7918 dataset/descriptor cases
python run.py --scope indoor         # 3DMatch + 3DLoMatch, both descriptors
python run.py --scope kitti          # KITTI, both descriptors
```

In VS Code, open this repository, select the installed Python environment, open `run.py`, and click **Run Python File** to run the full benchmark. Choose **Six-pair demo** in Run and Debug for a presentation. Environment and data setup are required before either option.

## What is new

| Component | Source |
|---|---|
| Inherited MAC graph, maximal cliques, node-guided clique selection, initial SVD | `third_party/mac/Linux/`, `cpp/PCR_provenance.cpp`, `cpp/registration_initialized.cpp` |
| Action-space diversity selection (DS), Section 3.3 | `h4_registration/core_indoor.py`, `core_kitti.py`, marker **本文创新代码起点** |
| Residual-weighted SVD refinement, Section 3.4 | `cpp/weighted_refine_indoor.cpp`, `weighted_refine_kitti.cpp` |
| Whole-cloud geometry, NC and density balancing, Section 3.5 | `preprocess()` and `geometry()` in the two core modules |
| Multiscale rank fusion (MS), Section 3.6 | The **新增模块 MS** block in the two core modules |

Weighted SVD is a supporting estimation step with prior methodological foundations. The code and manuscript distinguish it from the proposed selection mechanisms. Ground truth is read **after** hypothesis selection, solely for RE/TE/RR evaluation. The native MAC generator receives identity-transform and zero-label placeholders.

## Experiments and outputs

Each run creates a new directory under `outputs/` containing `results_raw.csv`, `summary.csv`, `all_controls.csv`, native process logs, candidate pools, cached geometry, refined poses, shortlists, and final 4×4 transformations. Existing completed runs are never overwritten. All failures remain in the recall denominator.

```bash
python run.py --resume outputs/YOUR_RUN_DIRECTORY
python experiments/analyze.py outputs/YOUR_COMPLETED_RUN_DIRECTORY
```

Resume with the same scope, `--per-group`, configuration and source files. A completed block is checked by hash. An incomplete block is retained for investigation; start a new run after diagnosing it.

`experiments/analyze.py` computes strict component ablations, budgets 1/4/8/16, and offline hypothesis coverage from the saved union of conventional and diversity-selected shortlists. It does **not** generate or refine hypotheses again. The eight intermediate controls emitted by the main runner are diagnostic controls, not the strict leave-one-component-out ablation table.

The timing experiment and its precise inclusion/exclusion rules are documented in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Manuscript reference results

| Dataset | Descriptor | Pairs | RR (%) | RE (°) | TE (cm) |
|---|---|---:|---:|---:|---:|
| 3DMatch | FPFH | 1623 | 87.00 | 2.03 | 6.53 |
| 3DMatch | FCGF | 1623 | 94.27 | 1.99 | 6.42 |
| 3DLoMatch | FPFH | 1781 | 45.42 | 3.65 | 9.81 |
| 3DLoMatch | FCGF | 1781 | 62.94 | 3.60 | 10.38 |
| KITTI | FPFH | 555 | 99.46 | 0.36 | 7.75 |
| KITTI | FCGF | 555 | 99.28 | 0.34 | 7.55 |

These are the manuscript's existing full-benchmark measurements, not a new full-benchmark run of the release adapter. Unrounded values and success counts are in `reference_results/main_summary.csv`. A six-pair smoke test validates execution, not these full-benchmark recall values. Rebuilt native libraries or different point-cloud preprocessing versions can change numerical details.

## Checks

```bash
python -m unittest discover -s tests -v
```

Synthetic tests cover the action-distance identity, bidirectional geometry and normal-sign invariance, cache ordering, rank-fusion tie breaking, manifest counts, and failure-denominator behavior. GitHub Actions runs these checks without downloading benchmark data.

## Citation and attribution

The manuscript is not represented as an accepted or published journal article. Use `CITATION.cff` for its author/title information and cite MAC when using its frontend:

```bibtex
@inproceedings{zhang2023mac,
  title={3D Registration With Maximal Cliques},
  author={Zhang, Xiyu and Yang, Jiaqi and Zhang, Shikun and Zhang, Yanning},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17745--17754},
  year={2023}
}
```

MAC's MIT license and original attribution are retained in `third_party/mac/LICENSE`. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the licensing scope. Benchmark data remain subject to their providers' terms.
