# Third-party notices

## MAC

Source: https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques

Local upstream checkout used for the experiment adaptation: `b0d19635c4c9d69399397495d02422eda94248c2`.

Copyright (c) 2023 zhangxy0517. License: MIT; the full notice is retained at `third_party/mac/LICENSE`.

The subset under `third_party/mac/Linux/` contains the frontend routines required by the build. `cpp/PCR_provenance.cpp` and `cpp/registration_initialized.cpp` are adapted from MAC with provenance/initialization changes. `cpp/producer.cpp` is the experiment adapter. Weighted-refinement and control tools also call MAC utility routines. See `docs/source_provenance.json` for source-file hashes and adaptation scope. Do not attribute MAC's inherited graph/clique/SVD mechanics to the current authors.

## Installed dependencies

PCL, igraph, Eigen, Boost, OpenMP, NumPy, SciPy and Open3D are installed dependencies, not relicensed by this repository. Their respective notices and terms apply. In particular, review igraph's license when redistributing a compiled program linked to it. This release distributes source and build instructions, not prebuilt linked executables.

## Data

No point clouds, correspondence files, descriptor weights or third-party dataset archives are bundled. Manifests identify files in the existing benchmark protocol. Obtain benchmark inputs from their providers and follow their terms.
