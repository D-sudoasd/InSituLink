# InSituLink 请求与交接格式

请求使用 UTF-8 JSON，必须包含 `schema_version: 1`、`frames` 和非空的 `sources`。相对路径以请求文件所在目录为基准；输出使用独立目录，不允许包含实验输入。

## 从文件生成请求

```powershell
py -3 -m insitulink prepare --frames D:/experiment/frames.csv --source mechanics=D:/experiment/load.csv --source furnace=D:/experiment/temperature.csv --output D:/experiment/setup
```

`prepare` 接受 CSV、XLSX、JSON 帧表或 HDF5；每台外部设备使用一个 `--source ID=PATH`。输出目录必须不存在。生成的 `experiment.json` 保留原输入路径，`import_preview.json` 展示列和元数据，`actions.json` 列出实际待补字段，`profile.json` 保存可复用配置；身份信息充分时另生成帧清单。

`--profile PATH` 可以复用列映射和设备配置。profile 使用 `schema_version: 1`，`frames` 保存帧配置，`sources` 是以设备 ID 为键的配置对象，`metadata` 保存实验/采集登记标识。例如：

```json
{
  "schema_version": 1,
  "frames": {
    "columns": {"start": "elapsed", "exposure": "exposure"},
    "time": {"unit": "s"},
    "exposure_unit": "s"
  },
  "sources": {
    "mechanics": {
      "time_column": "elapsed",
      "time": {"unit": "s"},
      "channels": [{"name": "force", "column": "F", "unit": "N", "kind": "continuous"}],
      "clock": {"method": "shared", "confirmed": false, "evidence": ""}
    }
  }
}
```

示例的单位和表头必须与实际记录相符。缺少同步依据时保持未确认，不以两个时间列都从零开始证明共用时钟。命令也接受 `--experiment`、`--frame-time-unit`、`--frame-time-start`、可重复的 `--source-time-unit ID=UNIT` 和 `--source-time-start ID=ISO8601`；更完整的字段保存在 profile 中。

帧表中的相对 `source_file` 默认仍以请求目录为基准。可选的 `frames.source_root` 仅改变这些原始帧路径的基准，自身相对路径仍按请求目录解析；它不改变 `frames.path` 或外部记录路径。`prepare` 用它保持原帧表内的路径含义，不复制表格来替代输入文件。谱文件及其身份路径仍按请求目录规则解析。

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

`frame_index` 是该文件/数据集内从 0 开始的索引。引用原始文件的表格行必须显式提供这个索引，不能用整表行号代替。单个 TIFF/CBF 文件通常为 0，不是整次扫描的累积编号。`frame_number` 为该索引加 1，`sequence_index` 表示完整采集序列中的位置。不同文件可具有相同索引；同一原始文件/数据集/索引不能重复。没有原始图像时可省略 `source_file`，此时缺省索引仅表示元数据记录顺序。

通过 `frames.columns` 中的 `id`、`frame_index`、`start`、`exposure`、`source_file`、`dataset` 映射其他表头。JSON 表使用记录数组；可用 `data` 记录数组代替 `path`。外部记录也支持 CSV/XLSX/JSON/内联数组，XLSX 可指定 `sheet`。CSV 与 XLSX 表头均要求非空且唯一，重复列名会直接要求修正，不静默选择后一列。

## 时间与 HDF5

`time.unit` 支持秒、毫秒、微秒、纳秒、分钟、小时；`time.start` 是 ISO8601 起点，`time.scaling_factor` 表示硬件时钟刻度缩放。ISO 时间没有时区时需要 `time.timezone`，如 `+08:00`；不采用机器时区补全。

默认 `exposure_s` 始终按秒解释。自定义曝光列使用 `exposure_unit`，未指定时采用 `time.unit`。绝对时间转换为 UTC 秒，纯经过时间保留相对起点；相同格式不证明设备同步。

计算同时保留精确十进制起点 `time_origin_s` 和 `t_start_rel_s/t_end_rel_s/t_mid_rel_s`，外部记录对应 `t_rel_s`。原有绝对数值字段继续提供，但高分辨率时间应使用“起点＋相对量”，不要对大数值 UTC 浮点端点做差。原始整数纳秒或 ISO 小数在减去起点前保留精度；缺少共同起点的相对数字与 ISO 混合列会被拒绝。起点规范化只改变计算坐标，不提供设备同步证据。

HDF5 帧配置：

```json
{
  "type": "hdf5", "path": "raw/master.h5",
  "data_dataset": "/entry/data/data",
  "start_dataset": "/entry/instrument/detector/start_time",
  "exposure_dataset": "/entry/instrument/detector/count_time"
}
```

路径需与本次 Eiger/NeXus 文件一致。读取时间字段的 `units`、`start`、`scaling_factor` 属性，显式请求配置优先。HDF5 曝光单位按显式 `exposure_unit`、曝光数据集 `units`、默认秒依次选择；元数据记录实际单位和来源。这里只读取元数据和图像数据集形状，不读取像素。二维数据集表示一帧，三维数据集使用第一维表示物理帧；时间戳不能制造不存在的帧。尾部填充及数量差异保留诊断；内部缺失时间保持原索引。

HDF5 身份检查解析别名及外链，并对支持的 VDS 保留逻辑索引到实际源帧的映射。未映射位置、重复物理帧和不能确定逐帧关系的映射保留具体诊断，不把虚拟长度直接当作已验证物理帧数。所有 HDF5 请求均在写输出前检查间接存储路径，避免元数据处理覆盖外部原始存储。

实际时间缺失时，仅在确认采集连续且周期恒定后加入：

```json
{"constant_period": {"confirmed": true, "start_s": 0, "exposure_s": 0.5, "interval_s": 0.9}}
```

`interval_s` 是相邻曝光开始时刻之差。重建使用原始文件内索引，并标明来源；多文件序列应提供整次采集的实际时间。HDF5 显式 `frame_count` 在数据集的物理数量范围内决定保留帧数，已确认周期时可重建其中缺失的尾部时间。没有显式数量、且有限时间记录不能确定真实帧与尾部填充边界时，返回 `needs_input` 要求确认数量，而不是宣称已经完整处理。

## 外部通道与时间偏移

每个 `source` 代表一台共用计时的设备，多个独立设备使用多个 `sources`。设备和通道标识使用字母、数字、下划线或连字符。

- 通道必须声明单位，无量纲用 `1`。不从量值猜测单位，不自动将载荷换算为应力。
- `kind: continuous` 线性插值；`kind: step` 右连续前值保持。
- `max_gap_s` 默认是该记录中位采样间隔的三倍，必须为正；缺口和缺失值不被跨越。曝光采样与阶段标注使用相同的间隔限制。
- 事件检测也遵守所选通道的 `max_gap_s`；跨越禁止缺口的起点不能确认为同步事件。
- `required` 默认为 `true`，完整曝光覆盖参与 `valid_analysis`；可选通道保留独立有效性。
- `phase_tolerance` 的单位是通道单位/秒；通道可用 `experiment` 覆盖主实验类型，例如热机械耦合记录中的温度和载荷。

时间关系约定为 `外部时间 = 探测器时间 + Δt`。四种方法：

决策保留原数值 `dt_s/candidate_dt_s`，并附 `dt_s_decimal/candidate_dt_s_decimal` 保存精确十进制表达。`local_dt_s/local_candidate_dt_s` 对应两设备各自相对起点下的计算坐标。实际采样使用局部偏移，不能把大数值公共浮点偏移再换算回来而丢失小数部分。

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

`offset/shared` 要求确认和非空依据。锚点必须是同一物理事件两侧的观测时间，默认按实际中位帧间隔检查一致性。事件检测完整使用声明的 `baseline_samples`，不会因前两点相等自动缩短基线；MAD 为零而基线仍有波动时使用标准差后备。基线不足或已明显变化时要求补充依据。可指定 `direction: rise|fall`，结合峰位或晶格应变检查候选。事件类型采用所选通道的 `experiment`，未声明时使用顶层类型。通用事件需要 `confirmed_same_event: true`；变温波形估计始终保留为候选，该标志不取消热响应限制。确认独立依据后改用 offset/anchors 再恢复。

事件还检查阈值变化及噪声/局部斜率造成的起点时间敏感性。不同信号越过各自阈值不一定表示同一个物理时刻；敏感性超过采样起点区间时保留候选，不能只用采样间隔给出过小不确定度。锚点的适用范围与残差随结果记录；本版不自动拟合漂移或跨设备重启拼接时间轴。

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

files 应覆盖输入帧序列，索引重复时附加对应 `source_file`。谱文件和 `baseline_frames` 身份中的 `source_file` 与帧表使用相同的请求目录解析规则，相对路径与等价绝对路径都可使用，但仍必须唯一对应。坐标可为 `d_A`、`q_A^-1`、`two_theta_deg`，后者要求 `wavelength_A`。CSV 默认逗号，TXT 默认空白，可配置 `delimiter`、`skiprows`。省略 windows 时从首、中、末谱选择最多三个共同可靠峰。默认基线为前八帧，需确认这些帧代表参考状态。

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

身份范围、持久采集/资产标识、HDF5 解析身份与旧 UID 兼容选择见 [逐帧身份与追溯](FRAME_IDENTITY.md)。修改时钟参数不会改变声明的物理身份；文件迁移时保留资产标识并更新其路径。

`frames.csv` 保留原始帧身份和时间。参数使用 `设备.通道` 前缀，附加 `.mid`、`.mean`、`.min`、`.max`、`.coverage`、`.valid`、`.flags`、`.unit`、`.phase`。中点值可独立存在；曝光均值和极值仅在完整曝光受支持时发布。

新增来源记录给出中点支点、插值权重和曝光所用记录区间，并保留输入行号及方法。均值与极值为该插值模型的推导值；`coverage=1` 不代表采样分辨率足以捕获所有变化。`.phase` 指曝光中点阶段，曝光内跨阶段单独记录。

交付包含 `frame_manifest.json`、逐帧追溯文件 `frame_records.json`、`decisions.json`、`peaks.csv`、`handoff.json`、`status.json` 和自包含的 `report.html`。交接 schema 为 `insitulink.handoff.v1`，帧 schema 为 `insitulink.frames.v1`，运行 schema 为 `insitulink.run.v1`；现有字段保持兼容，新增信息以附加字段提供。

`completed` 表示本次对齐交付完成，部分帧仍可因覆盖不足无效；`needs_input` 表示依据或输入待补充；`failed` 表示运行或交付失败。`ready_for_integration` 还要求至少有一帧满足所有必需通道，并且这些分析帧均引用存在的原始数据文件。仅有时间表的合成演示可以完成对齐，但不标为可进行真实积分；原因见 `raw_source_issues`。消费者仍需检查运行状态、文件完整性和原始帧/标定预检。

原始身份验证结果与文件存在性分别呈现；明确无效的物理寻址不能标就绪，未能验证的格式仍需消费者预检。报告可以搜索、按有效性/设备/时间筛选并展开帧详情；无效原因同时包含帧级和必需通道级 flags。缺口断线，阶梯量按右连续规则显示。

`inspect` 验证输出 SHA-256，不检查源文件；`resume` 重新检查输入与配置后复用有效阶段。只做元数据对齐时不读取图像像素。按需图像预分析追踪选中数据集的 master、嵌套外链、VDS 映射及外部存储，将实际数据文件、几何和掩膜内容加入缓存与运行期间变更检查。VDS 或外部存储存在多个搜索位置时，校验全部存在的候选及其递归依赖，并记录搜索条件，避免把某个文件系统候选误认为实际读取文件。依赖不能可靠枚举时禁止复用；重新执行仍无法验证时要求补充输入，不发布可交接结果。实际文件路径也受到输出目录保护。`truth.json` 不参与估计或恢复。
