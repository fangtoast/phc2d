<!--
  Curated figure provenance for the 50k full-cell forward report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 50k 全胞正向训练精选图件索引

| Metadata | Value |
|---|---|
| Status | Draft provenance index |
| Audience | 项目负责人、研究人员与报告维护者 |
| Owner | 研究生维护，导师审阅 |
| Source training campaign | `runs/20260807/full_cell_forward_50k_17wave_campaign_202608071737/` |
| Source evaluation campaign | `runs/20260808/full_cell_forward_50k_evaluation_campaign_20260808162200/` |
| Curated on | 2026-08-08 |
| Last reviewed | Pending human review |
| Reading time | 3 minutes |

## 本地精选图件

| 文件 | 来源 | SHA-256 |
|---|---|---|
| `training_curves_binary_native_n40000_e20.png` | `aggregations/binary_native_n40000_e20/figures/training_curves_by_model.png` | `791d9fe18241c9172298318b634f35ec9eafc0a40c6b955b6debaef443988943` |
| `current_40k_cross_evaluation.png` | evaluation `figures/current_40k_cross_evaluation.png` | `1ca4d1fbe94f6aca8bf749e59b4602a22ff1ffede6444f9c5df00c86126ff2ab` |
| `zero_shot_resolution_transfer.png` | evaluation `figures/zero_shot_resolution_transfer.png` | `cdf5f59987e1e0b85a623cb4e302705350ff5f20471216126acaaf7205f35008` |

## 17 波完整训练曲线

完整 PNG/PDF 保留在 N40 run 目录。下表记录 PNG 的相对路径、字节数与 SHA-256；本索引不把远端 raw gallery 全量复制进 Git。

| Wave | Bytes | SHA-256 |
|---|---:|---|
| `binary_native_n40000_e20` | 529,614 | `791d9fe18241c9172298318b634f35ec9eafc0a40c6b955b6debaef443988943` |
| `binary_native_n40000_e100` | 622,330 | `fe3cc5df1e47e6c01292819ea588e72564cfae1caffd38dba1d1f305467f291a` |
| `binary_native_n40000_e200` | 641,031 | `7c2567c11d0f90ba602af06eb4a5129443c9e82d1ad7e79ed6033191d147fba8` |
| `graded_continuous_n40000_e20` | 454,766 | `1295ad0a65833b1052892cece55af7d1b288290cf7b074434d0b092b9fc01585` |
| `graded_continuous_n40000_e100` | 542,989 | `139f1a4d369afdc544660c355df80a431201df0ffc1a2fcab48d8d91cd4cc773` |
| `graded_continuous_n40000_e200` | 570,364 | `84ff2780afeb0e5b3f91797dc64aea55e8d03181f7e01a854ace05245b118d04` |
| `binary_native_n50000_e20` | 520,811 | `2d8ac79f3226e31eb89dbc3dc75de86a69a0a47a7cc366e901d07e6b36b4c3dc` |
| `binary_native_n50000_e100` | 601,312 | `f905f8da57861c91333a34f464543bf5410751fd6a3710037c347794971574de` |
| `binary_native_n50000_e200` | 637,488 | `47f07398f3313f75fbf673c758dbf744307f683f791732be5dcca8fb44d8b2ea` |
| `graded_continuous_n50000_e20` | 514,463 | `d331fa201e34f6ce487e9304730ebb8dbbbd921856827b65934822fe84b5778c` |
| `graded_continuous_n50000_e100` | 609,710 | `b8dbf85f863c0f0b75204e6ad5e4d22935d74c01e29fc2565bbab747d02d6664` |
| `graded_continuous_n50000_e200` | 631,123 | `75ee371daa50943c5869a79d0ec007baa8f7aa49d222a56ea258a5832e5cac6b` |
| `graded_t03_n50000_e100` | 705,044 | `8527c0fd9fead9bf6fdb1b8b2e7ad5790122f8b76ea7f5021dd48255a25cd9a1` |
| `graded_t04_n50000_e100` | 679,810 | `4c48e2fdb71e67416e719b9a19b0513dab25e903ca4542ecf033f00b39d414f8` |
| `graded_t05_n50000_e100` | 603,493 | `1a66aeaf2b3177aa2480427f165cad9c10e253719e9c8b28ede491b1d8d175b7` |
| `graded_t06_n50000_e100` | 637,060 | `029f8f8c2d3cc8c7cf6b33513b2b16504f057d7b8a7ed624e8e370bdb856e612` |
| `graded_t07_n50000_e100` | 608,814 | `da11241e76d9e285859ca86aa96b0af1de1dbdbc231b4f1a4029ef163a60385a` |

每个文件的完整相对路径为：

```text
runs/20260807/full_cell_forward_50k_17wave_campaign_202608071737/
  aggregations/{wave}/figures/training_curves_by_model.png
```
