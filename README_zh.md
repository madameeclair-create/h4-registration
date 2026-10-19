# 基于作用空间多样性与多尺度几何评价的点云配准

作者：武明虎、张一铭、涂祥、王娟。对应英文论文题目见 [README](README.md)。本仓库不宣称论文已经被期刊接收。

## 给老师展示时打开哪里

用 VS Code 打开本目录，选择安装了 `requirements.txt` 的 Python 3.11 环境。主入口是 **`run.py`**。环境、数据和 C++ 编译完成后，点击右上角运行按钮默认运行三个数据集、两种描述子的完整主实验，共7918例。演示请在“运行和调试”中选“Six-pair demo”，每组固定取第一例，共6例，不按结果挑选。

## 初次配置

1. 在 Ubuntu 24.04 或 Windows 的 Ubuntu 24.04 WSL 中安装 PCL、igraph、Eigen、CMake 和编译器，具体命令见英文 README。
2. 在主机 Python 环境执行 `python -m pip install -r requirements.txt`。
3. 执行 `python scripts/build.py --jobs 2`，从源码生成5个工具。本仓库不上传本机预编译二进制。
4. 复制 `config.example.json` 为 `config.local.json`，填写室内数据与 KITTI 数据目录。数据准备见 [DATA.md](docs/DATA.md)。
5. 先运行环境检查和6例演示，通过后再运行完整主实验。

```text
python run.py --plan
python run.py --check
python run.py --per-group 1
python run.py
```

实测环境为 Windows Python + Ubuntu 24.04 WSL。代码同时提供原生 Linux 调用分支，但没有把这一分支描述为已完成全部数值复现。更换依赖版本可能影响极大团、浮点计算及点云预处理。

## 我们的创新从哪里开始

- `h4_registration/core_indoor.py`：3DMatch / 3DLoMatch 后端。
- `h4_registration/core_kitti.py`：KITTI 后端，长度参数统一乘6。
- 搜索 **“本文创新代码起点”**：进入作用空间多样性选择 DS。
- `preprocess()`：采样、法向估计、无向法向密度平衡 DB。
- `geometry()`：整云双向近邻支持 BG、法向一致性 NC、密度平衡。
- 搜索 **“新增模块 MS”**：跨5个尺度的最差名次、平均名次融合及最终选择。
- `cpp/weighted_refine_*.cpp`：固定初始对应上的加权精化，是支持步骤，不把 SVD 本身称为创新。
- `third_party/mac/` 与两份适配的 MAC C++ 文件：继承的图、极大团、节点引导选择和初始 SVD，保留原作者许可证。

选择不读取真值，真实变换只在最终 RE/TE/RR 评价和离线覆盖率诊断中读取。

## 消融、覆盖率与输出

主实验保存原始候选、常规与多样性短名单、共同精化的候选并集、几何缓存、结果与日志。结果在 `outputs/`，不会覆盖原始实验。

```text
python experiments/analyze.py outputs/你的完整运行目录
```

此脚本复用已保存候选完成严格消融、Top1/4/8/16预算与覆盖率诊断，不重新生成和精化。主入口输出的8个中间控制版本不等于论文逐项去除的严格消融。演示样本的 RR 不代表论文完整数据集的 RR。

所有路径均由本地配置与清单解析，不依赖作者电脑的盘符。输入清单保留7918例及文件哈希；数据、私人配置、编译目录、运行缓存和论文 Word 不上传到仓库。

## 论文结果数据

[论文表格数值与图2源数据](paper_data/supplementary_data_1/)包含10个CSV及来源说明，不包含原始点云或逐对位姿档案。
