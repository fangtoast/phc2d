---
title: "M9 binary 10k 导入与跨站点代码同步（2026-08-06）"
summary: "日期化实验与验证证据；正文保持原始记录。"
page_type: evidence
status: draft
audience: "研究用户、复现执行者与项目维护者"
owner: "研究生维护"
last_reviewed: 2026-08-13
reading_time: 4 minutes
primary_layer: L3
evidence_date: 2026-08-06
evidence_scope: "日期化证据或历史记录；正文未因站点重构而改写。"
---

<!--
  Paracloud M9 binary dataset import and code synchronization evidence.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# M9 binary 10k 导入与跨站点代码同步（2026-08-06）

| Metadata | Value |
|---|---|
| Status | data import and static deployment passed；未提交 MPB 作业 |
| Primary layer | L4 data/deployment provenance |
| N40/M9 deployed commit | `b93cc2d` |
| M9 root | `/publicfs10/fs10-m9/home/m9s005801/phc2d` |
| Evidence boundary | 证明文件传输、哈希、源码编译与脚本语法；不证明 packed Meep/MPB 在当前计算节点运行，也不是 replay 结果 |

## 结论

原先缺失的 binary full-cell 10k HDF5 已重新导入 M9。上传先写入隐藏临时文件，远端大小和 SHA-256 均匹配后才原子改名；未覆盖既有文件。M9 现在同时拥有 binary/graded 两份冻结数据：

| 数据 | bytes | SHA-256 | M9 状态 |
|---|---:|---|---|
| `data/full_cell_binary_bz9_10000_xingyi_v1.h5` | 250,275,716 | `1f131f2bda7ccb1f1736809bdf6bafd534096cb275e22ce3e56fc5d910530c2f` | imported / verified |
| `data/full_cell_graded_bz9_10000_paracloud_m9_v1.h5` | 1,210,491,259 | `d16d838e098215c05dc299bf7c4bbc8c6bc00f6ef35b9772546aa4e7c2b8b728` | existing / reverified |

两个哈希与本机及 N40 对应文件一致，因此没有复制后数据漂移。HDF5 的 schema/content acceptance 仍沿用各自已记录的本机 WSL 验收；本次不把字节一致扩大为新的 MPB 物理结论。

## 代码同步

同一份 Git archive 同步到 N40 与 M9：

- archive：`phc2d_b93cc2d.tar.gz`；
- SHA-256：`ac291ddd46ca909e097d6bc8cc9dc23d85ea67a78e35879f3734f818a403df34`；
- deployment identity：`b93cc2d`；
- M9 部署前备份：`/publicfs10/fs10-m9/home/m9s005801/deployments/phc2d_before_b93cc2d_20260806175639.tar.gz`。

第一次 staging compile 使用 M9 默认 `python3=3.6.8`，因不支持 `from __future__ import annotations` 而失败。部署脚本在 backup/copy/commit 更新之前退出，所以旧部署未被修改。随后明确使用平台已有 `python3.11=3.11.13` 完成全 `src/` compile，并通过 M9 三个 shell 入口的 `bash -n`，才执行备份、覆盖与 `DEPLOYED_GIT_COMMIT` 原子更新。

登录节点 Python 3.11 不含 `yaml/numpy/h5py`；这不替代 packed `meep_env`。正式 M9 作业仍须在 allocation 内展开哈希固定的环境归档，由 `paracloud_m9_common.sh` 验证 Meep `1.33.0-beta` 与 MPB `1.12.0`。

## 当前门禁

1. 静态数据与代码交接已 ready；N40/M9 使用同一 deployed commit 和冻结数据哈希；
2. 当前用户队列快照无 N40 或 M9 本任务作业；
3. M9 `amd_m9_768` 瞬时快照有 idle 节点，但容量会变化，不能据此自动提交；
4. full-cell inverse 候选尚不存在，正向 surrogate 也未训练，所以当前没有合法的 inverse replay manifest；
5. 只有候选 lineage/geometry hash 冻结并取得单独授权后，才能执行每路线 8-target inverse smoke。

## 人工理解记录

```text
解决了什么问题：补齐原来缺失的 M9 binary 10k 数据，并把 N40/M9 代码部署统一到同一 commit。
为什么采用这个方案：临时文件、远端 hash、原子改名和部署前备份能避免半文件与不可恢复覆盖；同 commit 便于 checkpoint/candidate/replay provenance 对齐。
改变了项目哪一层/哪一模块：L4 数据部署和跨站点交接；没有运行求解器或改变科学结论。
使用什么证据验证：两份 M9 文件的 bytes/SHA-256、Git archive hash、Python 3.11 compile、M9 shell syntax 和部署 commit。
最大风险、未知或下一 gate：packed Meep/MPB 尚未在本轮计算节点作业中重验；没有 forward/inverse checkpoint 或合法 candidate manifest，不能启动 replay。
```
