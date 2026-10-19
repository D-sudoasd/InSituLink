<p align="center">
  <img src="assets/readme/hero.png" width="100%" alt="StrainLink — Map diffraction records to a reference stress–strain curve / 将衍射记录映射到参考应力–应变曲线. Conceptual illustration / 概念插图。">
</p>

# StrainLink

**将逐帧衍射记录中的应力或应变，与完整参考应力–应变曲线对应起来。**

A Tk desktop tool for mapping frame-level stress or strain against a laboratory reference curve. It supports unit checks, overlay review, aligned tables, and plot export.

[安装与启动](#install--quick-start) · [数据映射步骤](#usage) · [输入条件](#features) · [图像应变工具 StrainTrace](https://github.com/D-sudoasd/StrainTrace)

[![MIT](https://img.shields.io/badge/License-MIT-455A64)](LICENSE)

<picture>
  <source media="(max-width: 600px)" srcset="assets/readme/diagrams/workflow-readme-md-1-mobile.svg">
  <img src="assets/readme/diagrams/workflow-readme-md-1.svg" width="100%" alt="StrainLink — workflow schematic / 流程示意图">
</picture>

<sub>[Editable diagram source / 可编辑图源](assets/readme/diagrams/workflow-readme-md-1.mmd)</sub>

**每行代表一次谱线、帧或采集。** 对应结果是参考曲线的插值估计，不是从衍射峰重新测量应力或应变。使用前必须核对应变的小数/百分数及应力单位；反向映射还需确认所选参考区段的对应关系。

## 原理示意 / Principle schematic

<p align="center">
  <img src="assets/readme/principle.png" width="100%" alt="Reference-curve interpolation maps frame strain to estimated stress — conceptual schematic / 概念示意图">
</p>

*逐帧已知应变在参考应力–应变曲线上对应插值应力；输出是参考曲线估计，使用前核对单位与参考区段。概念示意，非实验数据。*

*Known frame strain maps to interpolated stress on a reference stress–strain curve; estimates depend on units and the selected reference segment. Conceptual schematic, not experimental data.*

[查看完整示意图 / View full-size schematic](assets/readme/principle.png)

## Features

| Station has… | Mapper does… |
|--------------|--------------|
| Strain only | Interpolate → stress |
| Stress only | Inverse-interpolate → strain |
| Both | Units · alignment check · export |

- **PCHIP** shape-preserving interpolation when SciPy is available; otherwise linear fallback
- Optional Savitzky–Golay / smoothing guides (SciPy signal)
- Unit warnings (fraction vs percent strain; stress in MPa)
- High-contrast plotting and publication export presets
- Excel / CSV friendly table I/O via pandas + openpyxl

## Install / Quick start

Windows (double-click):

```text
start_stress_strain_mapper.bat
```

From source:

```powershell
pip install pandas numpy matplotlib scipy openpyxl
python sxrd_stress_strain_mapper_gui_v3.py
```

Requires Python 3 with **tkinter** (standard on most Windows / macOS Python installs).

## Usage

1. Load a **complete** lab reference σ–ε curve.
2. Load the beamline / station table (one row per frame).
3. Choose which column is missing (stress or strain) and confirm units.
4. Map → review overlay → export aligned table and plots.

## Scientific boundary — what it is NOT

- **Not** a constitutive material model, crystal plasticity solver, or FEM post-processor
- **Not** a DIC / virtual-extensometer strain extractor (see [StrainTrace](https://github.com/D-sudoasd/StrainTrace), formerly ezDIC, for image-based strain)
- Output quality **tracks the reference curve**; garbage in → garbage out
- Always verify **fraction vs percent** strain before publishing figures or tables

## Frame time alignment / 帧–应力–应变时间对齐

The Tk tool completes a missing stress or strain **on a reference curve**. The separate alignment entry places each frame on the **tensile record in time**. Those are different operations. Time alignment is not a wall-clock lock and not a sub-second lock.

帧时间对齐不是参考曲线插值，也不是墙钟锁定。每帧的查询时刻是曝光中点加一个 Δt。Δt 来自晶格应变起跳和力离开预载；分不清就保持 0，并在 decision 里写明原因。帧间隔是有限 `start_time` 的差，不用名义 `frame_time`、读出时间、`帧号 × 曝光`、填充帧数或 NeXus GUI 与探测器的时差。

```powershell
python -m strainlink align --demo heps_id07_eiger
python -m strainlink align --demo extracted_frame_clock
```

Profiles: `heps_id07_eiger` (finite Eiger starts + count time, no image read) and `extracted_frame_clock` (a frame table from another station, no HEPS HDF5). Details and the flags an agent must read: [docs/TIME_ALIGNMENT.md](docs/TIME_ALIGNMENT.md).

## Tests

```powershell
python -m unittest tests/test_stress_strain_mapper.py tests/test_time_align.py -v
```

## License

MIT — see [LICENSE](LICENSE).
