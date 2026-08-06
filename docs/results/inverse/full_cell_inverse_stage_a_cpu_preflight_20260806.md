<!--
  Evidence report for the full-cell inverse Stage-A CPU contract preflight.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-inverse-stage-a-cpu-preflight"></a>

# Full-cell inverse Stage-A CPU 合同预检

| Metadata | Value |
|---|---|
| Status | Draft evidence report；CPU contract passed，N40 execution pending authorization |
| Date | 2026-08-06 |
| Primary layer | L4 implementation |
| Implementation commits | `2d35006`、`d2b229a` |
| Evidence run | `runs/20260806/full_cell_inverse_stage_a_cpu_v1_202608061701` |
| Environment | WSL Ubuntu + conda `meep_env`；resolved device=`cpu` |
| Validation boundary | 单 batch Python/训练合同；不是 GPU、正式训练或 MPB/Meep 证据 |
| Owner | 研究生维护，待人工审阅 |
| Reading time | 6 minutes |

## 一、结论

Full-cell inverse 已具备独立的 Stage-A 数据与单 batch 训练 preflight 接口。真实 binary/graded 10k HDF5 上的 **5 个变体 × 3 个方法 = 15 cases** 全部完成 forward、encoded/raster/unit-circle loss、backward、AdamW step、原子 checkpoint 写入及精确恢复。

本次结果只证明 `full_cell_analytic_params` 的数据—网络—soft decoder—梯度—checkpoint 合同可执行。运行明确解析到 CPU，未使用 N40、未开始 epoch 级正式训练，也未调用 MPB/Meep。N40 正式单卡配置已冻结，但仍需单独计费授权。

## 二、实现与配置

实现入口：

- [`src/phc2d/inverse/full_cell_training.py`](../../../src/phc2d/inverse/full_cell_training.py)：按变体加载、route-train-only `Wh` normalization、三种无界 analytic 输出网络、15-case preflight 与标准 run 制品；
- [`configs/smoke/full_cell_inverse_stage_a_cpu_v1.yaml`](../../../configs/smoke/full_cell_inverse_stage_a_cpu_v1.yaml)：本次 CPU 合同配置；
- [`configs/research/full_cell_inverse_stage_a_n40_preflight_v1.yaml`](../../../configs/research/full_cell_inverse_stage_a_n40_preflight_v1.yaml)：待授权 N40 配置，强制 `cuda:0`、FP32、batch 64、`64x64` soft raster、禁用 AMP/TF32；
- [`jobs/paracloud_gpu_full_cell_inverse_stage_a_preflight.sh`](../../../jobs/paracloud_gpu_full_cell_inverse_stage_a_preflight.sh)：N40 allocation 内的冻结执行入口；要求显式标准 run 目录；
- [`tests/test_phc2d_full_cell_inverse_training.py`](../../../tests/test_phc2d_full_cell_inverse_training.py)：数据泄漏、15-case artifact、CLI 和正式配置 fail-closed 测试。

CLI：

```bash
phc2d inverse full-cell-preflight --config <config.yaml> [--run-dir <path>]
```

`representation` 必须显式冻结为：

```yaml
condition_target: Wh
output_repr: full_cell_analytic_params
condition_normalization: route_train_per_k_band
```

旧 `rho_wedge/rho_grid` inverse runner 未被伪装或静默复用。

## 三、数据合同

| 路线 | Dataset SHA-256 | 变体 train/val/test |
|---|---|---|
| binary | `1f131f2bda7ccb1f1736809bdf6bafd534096cb275e22ce3e56fc5d910530c2f` | polar-1 `2000/250/250`；polar-2 `2000/250/250`；spectral `4000/500/500` |
| graded | `d16d838e098215c05dc299bf7c4bbc8c6bc00f6ef35b9772546aa4e7c2b8b728` | RBF/Fourier-1 `4000/500/500`；RBF/Fourier-2 `4000/500/500` |

两条路线分别从自己的完整 train split 计算逐 `(k,band)` mean/std；validation/test 不参与 normalization 或 codec fit。保存的 `condition_normalization_binary.npz`、`condition_normalization_graded.npz` 和 `codec.resolved.json` 是后续 N40 checkpoint 的必需 lineage。

## 四、15-case 结果

| 方法 | cases | total loss 范围 | gradient norm 范围 | checkpoint exact reload |
|---|---:|---:|---:|---:|
| `direct_inverse_mlp` | 5 | 0.1282–0.5221 | 0.3146–0.5949 | 5/5 |
| `single_mode_generator` | 5 | 0.1180–0.5262 | 0.04126–0.3086 | 5/5 |
| `mixture_conditional_generator` | 5 | 0.1219–0.5243 | 0.04527–0.2388 | 5/5 |
| 合计 | **15** | finite | finite、non-zero | **15/15** |

Run 汇总：

| 字段 | 结果 |
|---|---|
| `status` | `cpu_contract_preflight_pass` |
| `case_count` | 15 |
| `batch_size` | 4 |
| `raster_shape` | `[32,32]` |
| `all_gradients_finite_nonzero` | `true` |
| `all_checkpoint_reloads_exact` | `true` |
| `wall_time_total_s` | 25.1086 |
| `formal_training_started` | `false` |
| `mpb_meep_used` | `false` |
| recorded git commit | `d2b229a95bb1e7f45e8162fd319027f700c306c2` |

## 五、验证命令

真实 10k CPU preflight：

```powershell
wsl -d Ubuntu -e bash -lc "source /home/fangxiao/miniforge3/etc/profile.d/conda.sh && conda activate meep_env && cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' && PYTHONPATH=src python -m phc2d.cli inverse full-cell-preflight --config configs/smoke/full_cell_inverse_stage_a_cpu_v1.yaml --run-dir runs/20260806/full_cell_inverse_stage_a_cpu_v1_202608061701"
```

WSL 回归集合：

```powershell
wsl -d Ubuntu -e bash -lc "source /home/fangxiao/miniforge3/etc/profile.d/conda.sh && conda activate meep_env && cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' && PYTHONPATH=src python -m pytest -q tests/test_phc2d_full_cell_inverse_codec.py tests/test_phc2d_full_cell_inverse_training.py tests/test_phc2d_full_cell.py tests/test_phc2d_inverse_candidate_protocol.py tests/test_phc2d_cli_and_profiles.py"
```

回归结果为 `62 passed`。Raw run 保留在 evidence-owning workstation 的 ignored `runs/`；核心状态通过本报告进入 Git，但本报告不替代 raw checkpoint、history 和 config snapshot。

## 六、下一 gate

1. 人工授权后，在 N40 单张 RTX4090 上运行冻结的 research preflight 配置；
2. 核对 15 cases、batch 64、显存峰值、数据读取和 checkpoint exact reload；
3. 单卡通过后，提交 18 个独立短任务，实际核对 `AllocTRES` 与共享 HDF5 吞吐；
4. 只有前述 GPU gate 通过，才实现/执行带 validation、best checkpoint 与 patience 的 Stage-A 正式 epoch runner；
5. Stage B、candidate generation、8-target MPB smoke 仍是后续独立 gate。

## 七、人工理解记录

```text
解决了什么问题：把 full-cell analytic codec 接入可执行的三方法单 batch 训练合同，并验证 checkpoint 可恢复。
为什么独立实现：旧 inverse runner 假设 rho 输出和 Sigmoid 末层，直接复用会破坏周期参数及无界 standardized analytic tensor。
改变了哪一层：L4 数据适配、模型接口、CLI、配置与测试；没有形成 L3 模型效果或 replay 结论。
用什么验证：真实双路线 10k HDF5、15-case WSL CPU preflight、标准 run artifacts 和 62 项相关回归。
剩余风险：尚无 N40 显存/吞吐证据，尚无 validation epoch runner、Stage B、候选池或 full-cell MPB replay。
```

[↑ 回顶部](#full-cell-inverse-stage-a-cpu-preflight)
