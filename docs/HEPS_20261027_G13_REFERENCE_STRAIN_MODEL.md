# HEPS G13：参考应力–应变模型估计

2026-11-04。用户明确批准在原G13应变通道失效时，用G12/G21加载曲线与G14峰后形状，根据原有逐帧应力估计总应变。此模型不使用横梁位移、机器柔度或G13失效第三列拟合。

[完整报告与数据](<H:/1_Large_Scientific_Facilities_Source_Data/20261027_HEPS_ID07_GEN-26-01118_insitu_tensile_SXRD/05_alignment/G13_reference_stress_model_20261104/README.md>) · [主曲线](<H:/1_Large_Scientific_Facilities_Source_Data/20261027_HEPS_ID07_GEN-26-01118_insitu_tensile_SXRD/05_alignment/G13_reference_stress_model_20261104/G13_estimated_stress_strain.png>) · [逐帧CSV](<H:/1_Large_Scientific_Facilities_Source_Data/20261027_HEPS_ID07_GEN-26-01118_insitu_tensile_SXRD/05_alignment/G13_reference_stress_model_20261104/G13_frame_stress_strain_estimates.csv>)。

## 已采用的方法

各样品以自身前10点中位数为预载相对基线，参考第三列相对值按百分数解释，终止行排除。G21主应力使用现场面积1.9872 mm²；G13应力保持TXT原值。

G12/G21加载曲线按10 MPa应力增量取双中位数节点，保留精确原点和峰点，再PCHIP插值。令a_j=S13/Sj，峰前eps13,j=a_j*f_j(delta_sigma13/a_j)，主线为两参考等权平均。这是保留弹性斜率的强度范围同比缩放假设，不能当成参考原载荷范围内的实测插值。

G14同应力加载插值独立对照。峰后以相对峰载应力下降比例映射G14下降分支，将其非弹性模型量的相对增长按G13峰载非弹性模型量缩放，再加G13当前应力的弹性等效量。E主假设为三参考50–150 MPa表观E中位数88.558148 GPa。峰后为强假设；峰值只保证数值连续，不保证斜率连续。

## 结果与读取

- G13峰应力保持1006.5872 MPa；G12/G21分别预测峰载应变1.205986%/1.362189%，等权主估计1.284087%。
- 最后非终止行207.584s的条件估计1.454843%，不是标准断后伸长率。
- 保留191帧；wall/nominal应力和估计覆盖179/188帧，峰后10/0帧，原查询时间、支点、权重、身份与应力逐字保持。
- 原始2080行全部保留；最初5行低于相对基线、最后1行为疑似软件终止行，新总应变留空并注明原因。
- 敏感性为两参考/等权平均 × 5/10/20 MPa节点 × 三弹性窗口，共27组。21组可作峰后估计，6组因峰载非弹性模型量不为正而峰后留空。末端模型范围约1.279–1.794%，不是置信区间或完整不确定度。

逐帧读取wall_estimated_total_strain_pct / nominal_estimated_total_strain_pct，必须同时读branch、estimate_status、模型范围。原strain_pct_candidate_selected仍为空；新估计保存在reference_model_estimates，原parameter_samples不改。

“总估计减假定弹性项”在部分加载段为负，保留符号诊断，不称作实测塑性。G14近峰同应力箱可能含宽应变区间；峰后形状受分箱影响。原G13通道没有恢复，设备同步和仪器标定没有新确认。不得将本估计当作唯一重建、精确屈服/均匀延伸或实测断裂应变。

## 后续复用

原始TXT及TC4硬链接旁注已链接此模型包；旧五样品包中的G13空值继续保留。重算使用模型包内build_g13_reference_model.py，并指定新的输出目录。模型参数、参考节点及源行、敏感性、帧记录与handoff均随包交付，自定义研究格式不代表DIDASFlow自动接入。
