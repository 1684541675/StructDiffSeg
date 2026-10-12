# ACDC Class Embedding + SDF 最小实验

实现分支：`research/acdc-geo-label`。以 `8abe97e` 的已验证 DiffUNet 基线为起点。
未启动完整 ACDC 训练，未更改 main 分支、数据划分、训练入口、其他模型或基线权重。

## 修改文件与关键差异

- `models/DiffUNet3201/diffunet/nnunet2d_denoise.py`：新增 `lambda_geo` 与统一 `encode_mask()`，替换普通训练 GT、MC-Diff GT、DDIM argmax 的三处编码。
- `models/DiffUNet3201/diffunet/sdf.py`：独立的 SciPy 欧氏距离变换工具。
- `tests/test_geo_label.py`：单元测试及实际网络轻量集成测试。
- `tests/benchmark_geo_label.py`：交替顺序、CUDA 同步、保留各次原始耗时的合成数据测量。
- 本目录：中文说明、测试结果与三个耗时 JSON 记录。

四通道依次为 Background / RV / MYO / LV。每通道 one-vs-rest：

```text
SDF = clip(EDT(outside) - EDT(inside), -16, 16) / 16
C = 2 * sigmoid(Embedding(Y)) - 1
z0 = 0.001 * ((1 - lambda_geo) * C - lambda_geo * SDF)
```

内部负、外部正，缺失类别恒 +1，整幅类别恒 -1。距离单位为像素，使用像素中心距离；边界两侧相邻像素为 ±1/16。
默认 `lambda_geo=0.25`；工厂支持 `DIFFUNET_LAMBDA_GEO` 环境变量，显式工厂参数优先。
模型创建时会打印实际 `lambda_geo`、编码模式、SDF 开关、参数来源、R、bit_scale 和 DDIM 步数。
未设置环境变量时打印 `source=default (DIFFUNET_LAMBDA_GEO unset)`；设置为 0 时打印 `Class Embedding only` 与 `SDF=disabled (skipped)`。
`lambda_geo=0` 直接返回原 Class Embedding 编码，逐元素一致，并完全跳过 SDF。
SDF 在模型中从已完成空间增强的输入标签计算；MC-Diff 在重复 batch 前编码。
推理仅从当步预测的 argmax 生成 SDF，GT 不进入编码。

网络参数、输出四类别 logits、直接类别预测、CE/Dice、噪声日程、DDIM 更新公式、三步采样和双分支输出组合均保持原状。
SDF 不参与求导；Embedding 梯度保留。没有新增注意力、解码器、几何损失或采样优化。

## 测试结果

2026-10-12 本地环境：Python 3.12.3，PyTorch 2.6.0+cu124，SciPy 1.16.2，NVIDIA GeForce MX350。
项目虚拟环境：`D:/code/Python/MedSeg/.venv-structdiffseg/Scripts/python.exe`。

```powershell
& D:\code\Python\MedSeg\.venv-structdiffseg\Scripts\python.exe -m unittest discover -s tests -p test_geo_label.py -v
```

最终 12 项全部通过，无跳过，执行测试约 2.919 秒（不含 Python 启动及依赖导入）：

1. 四类 SDF 形状、范围、方向、16 像素截断及边界距离。
2. 缺失类别、全背景、整幅任一单类别、单像素类别，均无 NaN/Inf。
3. 非法 SDF 输入校验。
4. `lambda_geo=0` 与旧编码逐元素相等，并确认不调用 SDF。
5. 融合公式和 Embedding 梯度。
6. CPU 训练前向及 CE/Dice 反向，确认普通/MC 两次统一编码调用。
7. GPU 训练前向及反向。
8. DDIM 恰好三次编码，逐步输入与 decoder argmax 一致；无 GT 时可推理，传入不可读取的 GT 哨兵仍得到相同输出。
9. CPU/GPU、标签跨设备、编码 FP16/BF16/FP32/FP64 的 dtype 与设备转换。
10. CUDA AMP 前反向及推理。
11. 工厂实验参数、三步默认值、bit_scale 及 state_dict 严格加载。
12. 原尺寸六阶段完整 DiffUNet 双分支（CPU、64×64、B=1）真实前反向及 `[1,4,64,64]` 三步推理。

单元测试大多缩小阶段数和通道数以保持轻量；第 12 项保留生产网络结构，仅缩小合成图像。
新增属性不增加 state_dict 参数键，兼容原权重加载；加载原权重后仍必须设置正确编码权重。

## 耗时测量

每种权重预热两次，7 次计时，交替运行 `0` / `0.25`，报告各组中位数。
CUDA 起止均同步；CPU 设置 2 线程。训练耗时包含普通前向、MC-Diff、原 CE/Dice 反向；推理包含三步 DDIM。
仅测去噪器，**不含图像编码分支、优化器更新和数据加载**，不等同于完整 ACDC 训练步或整卷推理耗时。

原六阶段去噪器，GPU，256×256，B=1，FP32（`timing_production_256_cuda.json`）：

| 操作 | lambda=0 | lambda=0.25 | 额外耗时 | 增幅 |
|---|---:|---:|---:|---:|
| 单次标签编码 | 0.43 ms | 33.73 ms | 33.29 ms | 编码本身较小，比例不宜比较 |
| 训练前向+反向 | 232.16 ms | 307.15 ms | 74.99 ms | 32.3% |
| 三步 DDIM | 396.95 ms | 600.83 ms | 203.88 ms | 51.4% |

原六阶段去噪器，CPU，64×64，B=1（`timing_production_64_cpu.json`）：训练 294.07 → 310.40 ms（+16.33 ms）；推理 166.19 → 183.48 ms（+17.29 ms）。
缩小测试网络，256×256，B=1（`timing_reduced_256.json`）：CPU 训练 +76.64 ms、推理 +153.53 ms；GPU 训练 +60.76 ms、推理 +105.73 ms。

完整原始样本保存在 JSON 中。生产 GPU 推理存在明显波动（原编码约 111–399 ms，融合约 302–629 ms），上述增幅只描述这次本地测量，不能直接外推到服务器或 batch=16。
预测 mask 的类别覆盖和几何复杂度也会影响 EDT 计算开销。

复现：

```powershell
$py = 'D:\code\Python\MedSeg\.venv-structdiffseg\Scripts\python.exe'
& $py tests/benchmark_geo_label.py --production --size 256 --rounds 7 --device cuda --output experiment_records/geo_label/timing_production_256_cuda.json
& $py tests/benchmark_geo_label.py --production --size 64 --rounds 7 --device cpu --output experiment_records/geo_label/timing_production_64_cpu.json
& $py tests/benchmark_geo_label.py --size 256 --rounds 7 --output experiment_records/geo_label/timing_reduced_256.json
```

## 已知风险与正式训练建议

- SciPy EDT 在 CPU 上执行，CUDA 标签需要同步回传与拷贝；当前训练两次 GT 编码、三步推理三次预测编码，确有额外开销。本实验保留这些调用，未做缓存或采样优化。
- `lambda_geo` 是实验属性，不写入 state_dict。训练、恢复和推理必须设置相同值；原基线推理显式设为 0，几何实验设为 0.25。
- R=16 是增强/缩放后像素距离，不是物理毫米；SDF 保留本实验约定的背景通道。
- 测试只验证编码及计算路径正确性，不能证明分割质量改善。正式实验应使用相同初始化种子、数据划分、批大小、轮数、损失和采样设置，与 lambda=0 对照。
- 本地 MX350 的 B=1 合成测试不代表能容纳正式 B=16；正式训练使用原基线已验证的训练硬件与配置。
- 默认系统 Python 的 PyTorch 有 DLL 导入问题；上述项目虚拟环境可用。本次没有修改依赖。

建议在仓库根目录执行以下 PowerShell 命令。**本次没有执行此训练命令**：

```powershell
$py = 'D:\code\Python\MedSeg\.venv-structdiffseg\Scripts\python.exe'
$env:DIFFUNET_LAMBDA_GEO = '0.25'
$env:LOCAL_DEBUG = '0'
& $py -c "import os; from datetime import datetime; from train_acdc import main; from configs.config_setting_acdc import setting_config as c; c.lambda_geo=float(os.environ['DIFFUNET_LAMBDA_GEO']); c.sdf_radius=16; c.work_dir='results/acdc-diffunet-geo025_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'/'; main(c)"
```

这只在调用时设置独立输出目录及日志属性，训练入口文件无需修改；保持当前配置的 seed=42、B=16、100 epochs、AMP=False、原优化器和损失。
独立目录避免自动恢复或覆盖原基线 checkpoint。
如需同分支对照，设置 `DIFFUNET_LAMBDA_GEO='0'` 并将目录前缀改为 `acdc-diffunet-geo000_`；正式推理也应对应设定环境变量。
禁用本地已有的 audit/debug 环境变量，确保从新目录开始完整训练。
