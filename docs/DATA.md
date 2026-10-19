# Benchmark input preparation

This repository starts from point clouds and **existing FPFH/FCGF correspondence files**, not from feature extraction or network training. No benchmark data are redistributed here.

## Sources

- [MAC upstream repository and its dataset download section](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques#datasets) provides processed benchmark inputs. Follow the current upstream download instructions and the data providers' terms. The upstream page links a dataset archive and lists its access code.
- [3DMatch project](https://3dmatch.cs.princeton.edu/) describes the indoor scans and benchmark.
- [PREDATOR](https://github.com/prs-eth/OverlapPredator) defines the 3DLoMatch low-overlap evaluation pairs.
- [KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) describes the outdoor source data. Raw KITTI downloads alone do not supply the exact processed descriptor correspondences used here.

The manifest hashes identify the exact processed files used in the manuscript. Download availability is controlled by the upstream providers. Do not silently substitute different point sampling, descriptor extraction, or matching when claiming to reproduce the same-input protocol.

## Directory layout

```text
INDOOR_DATA_ROOT/
  7-scenes-redkitchen/
    cloud_bin_0.ply
    cloud_bin_0+cloud_bin_10@corr.txt
    cloud_bin_0+cloud_bin_10@GTmat.txt
    cloud_bin_0+cloud_bin_10@corr_fcgf.txt
    cloud_bin_0+cloud_bin_10@GTmat_fcgf.txt
    ...
  7-scenes-redkitchen_3dlomatch/
    ...
  <seven other scene folders and their _3dlomatch counterparts>/

KITTI_DATA_ROOT/
  correspondence_fpfh/
    0/src_kpts.pcd
    0/tgt_kpts.pcd
    0/fpfh@corr.txt
    0/fpfh@gtmat.txt
    ...
  correspondence_fcgf/
    0/src_kpts.pcd
    0/tgt_kpts.pcd
    0/fcgf@corr.txt
    0/fcgf@gtmat.txt
    ...
```

Set `indoor_data_root` and `kitti_data_root` in `config.local.json` to these roots. The public manifests use `${INDOOR_DATA_ROOT}` and `${KITTI_DATA_ROOT}` placeholders, never developer-machine paths. Environment variables `INDOOR_DATA_ROOT` and `KITTI_DATA_ROOT` override the configuration.

On Windows, Open3D's PCD reader may reject non-ASCII paths. The main runner temporarily copies the unchanged PCD bytes to an ASCII path before preprocessing and cleans that temporary file afterward. Point ordering and all scientific preprocessing remain unchanged. Using ASCII-only dataset roots avoids this extra file copy.

Each correspondence file has six whitespace-separated coordinates per row: `sx sy sz tx ty tz`. Coordinates are in meters. A ground-truth file contains a 4×4 rigid transform from the source frame to the target frame. File names, pair directions, descriptor branches, and byte hashes must match `manifests/*.csv`.

The complete protocol has 1623 3DMatch pairs, 1781 3DLoMatch pairs, and 555 KITTI pairs, each evaluated with two descriptors. Each manifest lists dataset, scene, pair ID, paths, and SHA-256 hashes. The preflight checks file presence; actual processing checks file hashes.

New user-created data should be added through a separately named manifest and documented protocol. Changing existing hashed inputs changes the reproduction setting.
