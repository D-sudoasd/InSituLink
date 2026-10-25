# InSituLink 请求与交接格式

请求使用 UTF-8 JSON，必须包含 `schema_version: 1`、`frames` 和非空的 `sources`。相对路径以请求文件所在目录为基准；输出使用独立目录，不允许包含实验输入。

## 单参数请求

```json
{
  "schema_version": 1,
  "experiment": "thermal",
  "frames": {"type": "table", "path": "frames.csv", "time": {"unit": "s"}},
  "sources": [{
    "id": "furnace", "path": "temperature.csv", "time_column": "time",
    "time": {"unit": "s"},
    "channels": [{"name": "temperature", "column": "T", "unit": "degC", "kind": "continuous", "required": true}],
    "clock": {"method": "offset", "offset_s": 2.4, "confirmed": true,
              "evidence": "Replace with this experiment's independent clock evidence"}
  }]
}
```

数值 `2.4` 仅演示字段，必须换成本次实验依据。`experiment` 可取 `tensile`、`thermal`、`generic`。

逐帧 CSV 默认列示例：

```csv
frame_id,frame_index,t_start_s,exposure_s,source_file,dataset
sample_A_001,0,0.0,0.5,raw/master.h5,/entry/data/data
sample_A_002,1,0.9,0.6,raw/master.h5,/entry/data/data
```

`frame_index` 是该文件/数据集内从 0 开始的索引。单个 TIFF/CBF 文件通常为 0，不是整次扫描的累积编号。`frame_number` 为该索引加 1，`sequence_index` 表示完整采集序列中的位置。不同文件可具有相同索引；同一原始文件/数据集/索引不能重复。没有原始图像时可省略 `source_file`。

通过 `frames.columns` 中的 `id`、`frame_index`、`start`、`exposure`、`source_file`、`dataset` 映射其他表头。JSON 表使用记录数组；可用 `data` 记录数组代替 `path`。外部记录也支持 CSV/XLSX/JSON/内联数组，XLSX 可指定 `sheet`。

## 时间与 HDF5

`time.unit` 支持秒、毫秒、微秒、纳秒、分钟、小时；`time.start` 是 ISO8601 起点，`time.scaling_factor` 表示硬件时钟刻度缩放。ISO 时间没有时区时需要 `time.timezone`，如 `+08:00`；不采用机器时区补全。

默认 `exposure_s` 始终按秒解释。自定义曝光列使用 `exposure_unit`，未指定时采用 `time.unit`。绝对时间转换为 UTC 秒，纯经过时间保留相对起点；相同格式不证明设备同步。

HDF5 帧配置：

```json
{
  "type": "hdf5", "path": "raw/master.h5",
  "data_dataset": "/entry/data/data",
  "start_dataset": "/entry/instrument/detector/start_time",
  "exposure_dataset": "/entry/instrument/detector/count_time"
}
```

路径需与本次 Eiger/NeXus 文件一致。读取时间字段的 `units`、`start`、`scaling_factor` 属性，显式请求配置优先。这里只读取元数据和图像数据集形状，不读取像素。二维数据集表示一帧，三维数据集使用第一维表示物理帧；时间戳不能制造不存在的帧。尾部填充及数量差异保留诊断；内部缺失时间保持原索引。

实际时间缺失时，仅在确认采集连续且周期恒定后加入：

```json
{"constant_period": {"confirmed": true, "start_s": 0, "exposure_s": 0.5, "interval_s": 0.9}}
```

`interval_s` 是相邻曝光开始时刻之差。重建使用原始文件内索引，并标明来源；多文件序列应提供整次采集的实际时间。

## 外部通道与时间偏移

每个 `source` 代表一台共用计时的设备，多个独立设备使用多个 `sources`。设备和通道标识使用字母、数字、下划线或连字符。

- 通道必须声明单位，无量纲用 `1`。不从量值猜测单位，不自动将载荷换算为应力。
- `kind: continuous` 线性插值；`kind: step` 右连续前值保持。
- `max_gap_s` 默认是该记录中位采样间隔的三倍，必须为正；缺口和缺失值不被跨越。
- `required` 默认为 `true`，完整曝光覆盖参与 `valid_analysis`；可选通道保留独立有效性。
- `phase_tolerance` 的单位是通道单位/秒；通道可用 `experiment` 覆盖主实验类型，例如热机械耦合记录中的温度和载荷。

时间关系约定为 `外部时间 = 探测器时间 + Δt`。四种方法：

```json
{"method": "offset", "offset_s": -3.5, "confirmed": true, "evidence": "Independent marker observation"}
```

```json
{"method": "shared", "confirmed": true, "evidence": "Verified common clock and common origin"}
```

```json
{"method": "anchors", "pairs": [
  {"frame_time_s": 10, "source_time_s": 12.4},
  {"frame_time_s": 80, "source_time_s": 82.4}
], "tolerance_s": 0.5}
```

```json
{"method": "event", "channel": "force", "baseline_samples": 8, "min_run": 3, "k_sigma": 4}
```

`offset/shared` 要求确认和非空依据。锚点必须是同一物理事件两侧的观测时间，默认按实际中位帧间隔检查一致性。事件检测使用基线和持续变化，可指定 `direction: rise|fall`，结合峰位或晶格应变检查候选。通用实验需要 `confirmed_same_event: true`；变温波形估计始终保留为候选，确认独立依据后改用 offset/anchors 再恢复。

## 按需峰分析

只有设备使用事件对齐，且没有现成 `eps_hkl` 帧列时才读取 `features`。已有一维谱配置：

```json
{
  "type": "spectra", "axis": "d_A", "columns": [0, 1], "skiprows": 1,
  "files": [{"frame_index": 0, "path": "spectra/frame_001.csv"},
            {"frame_index": 1, "path": "spectra/frame_002.csv"}],
  "windows": [{"id": "peak_1", "lower": 1.95, "upper": 2.05}],
  "baseline_frames": [0, 1], "snr_min": 5
}
```

files 应覆盖输入帧序列，索引重复时附加对应 `source_file`。坐标可为 `d_A`、`q_A^-1`、`two_theta_deg`，后者要求 `wavelength_A`。CSV 默认逗号，TXT 默认空白，可配置 `delimiter`、`skiprows`。省略 windows 时从首、中、末谱选择最多三个共同可靠峰。默认基线为前八帧，需确认这些帧代表参考状态。

原始二维帧配置：

```json
{
  "type": "images", "axis": "q_A^-1", "geometry": "calibration.poni",
  "mask": "detector_mask.edf", "correct_solid_angle": true,
  "polarization_factor": null, "azimuth_range": [-10, 10], "npt": 1500,
  "windows": [{"id": "peak_1", "lower": 2.8, "upper": 3.2}]
}
```

范围仅示范字段，不是实验推荐值。不使用附加掩膜时仍需写 `mask: null`；波长从 PONI 读取，显式 wavelength_A 必须一致。不指定 azimuth_range 时使用全环，不推断 LD 方向。一次只计算所需一维谱，保留掩膜及无支持区间。

## 选择、交付与恢复

`analysis.frame_indices` 指定文件内从 0 开始的索引；跨文件索引重复时用 `analysis.frame_uids` 精确选择。默认选择全部帧，基线与重复参数值均保留。

`frames.csv` 保留原始帧身份和时间。参数使用 `设备.通道` 前缀，附加 `.mid`、`.mean`、`.min`、`.max`、`.coverage`、`.valid`、`.flags`、`.unit`、`.phase`。中点值可独立存在；曝光均值和极值仅在完整曝光受支持时发布。

交付包含 `frame_manifest.json`、`decisions.json`、`peaks.csv`、`handoff.json`、`status.json` 和自包含的 `report.html`。交接 schema 为 `insitulink.handoff.v1`，帧 schema 为 `insitulink.frames.v1`，运行 schema 为 `insitulink.run.v1`。

`completed` 表示本次对齐交付完成，部分帧仍可因覆盖不足无效；`needs_input` 表示依据或输入待补充；`failed` 表示运行或交付失败。`ready_for_integration` 还要求至少有一帧满足所有必需通道，并且这些分析帧均引用存在的原始数据文件。仅有时间表的合成演示可以完成对齐，但不标为可进行真实积分；原因见 `raw_source_issues`。消费者仍需检查运行状态、文件完整性和原始帧/标定预检。

`inspect` 验证输出 SHA-256，不检查源文件；`resume` 重新检查输入与配置后复用有效阶段。只做元数据对齐时不读取图像像素；按需图像预分析的缓存还校验图像、几何与掩膜内容。`truth.json` 不参与估计或恢复。
