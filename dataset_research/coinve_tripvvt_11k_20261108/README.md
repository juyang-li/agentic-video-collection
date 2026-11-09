# CoinVE＋TripVVT 11K：数据入口与执行经验

本地准备已完成：训练 10,000、验证 500、测试 500，其中 CoinVE 8,800 对、TripVVT 2,200 对。最终作业 `196203` 在 `testqueue` 正常结束。按最终要求，不追加独立验收、正式打包或上传；之前单独要求的 50 对滚动审核包已交付。

## 使用入口

- `manifest.jsonl`：正式配对、完整 prompt、参考图、split、来源和原媒体 SHA256；`manifest.csv` 是表格副本。
- `crops.csv`：两侧起点及共同结束时间。所有媒体路径相对清单所在目录解析。
- `dataset_reader.py`：`iter_samples(manifest, split=..., dataset=...)` 返回含绝对媒体路径的记录；`sample_times(row, fps=..., frames=...)` 返回两侧共用时间戳，超出共同区间会报错。
- `HANDOFF_FOR_COLLEAGUE.txt`：给师兄的完整字段、模型输入条件和接入示例。不要移动或清理被清单引用的历史原媒体。

```bash
python dataset_reader.py manifest.jsonl --split train --dataset tripvvt --fps 12 --frames 49 --limit 1
```

## 最重要的经验

1. **集群任务只能提交 `testqueue`。** 脚本显式指定分区并检查 `SLURM_JOB_PARTITION`；需要排队就休眠。该队列要求至少申请一张 GPU，但本次下载、探测、评分只用 CPU。
2. **先固定版本、元数据哈希和种子，再处理素材。** 本次 seed 为 `20261108`。网络失败重试原候选，不随意换样；固定版本明确返回 `404 / EntryNotFound` 才记为配对不全技术失败。
3. **长期下载要刷新签名地址。** CoinVE 地址实测约一小时有效；采用 45 分钟主动刷新和 401/403 刷新重试。tar 缓存已验证头索引，近邻范围跳过视频载荷，只读头定位，减少重复网络传输。
4. **原始规格与训练采样分开。** CoinVE 保留两/三任务完整有序指令，任意 Stylize 整对排除；共同区间至少 4 秒。两侧使用相同绝对时间戳，不比较时长倍数、不拉伸、不补长尾。
5. **以媒体来源组和哈希防止划分泄漏。** 不直接沿用 TripVVT 有重叠的官方 split；历史训练与保留评测的用途约束持续生效。CoinVE 两/三任务 70%/30%，首操作 Add/Remove/Replace 25%/30%/45% 是抽样配额，不是全库分布或最优比例结论。
6. **TripVVT 必须携带服装图，caption 不等于换衣指令。** 复用唯一的 `../tripvvt_research/score_pair_similarity.py`；首批 3,000 对中 8 对技术不合格，相似规则剔除 167/2,992（5.58%），各类均未超过 15%。该规则不能证明编辑失败。
7. **运行证据与数据入口分开。** 临时文件先写 `.part` 再替换；原素材可复用，探测和评分可续跑。共享盘保留数据真值，临时 ZIP 使用计算节点本地盘。计算节点不能沿用登录节点的 localhost 代理。

## 目录

```text
./
├── manifest.jsonl / manifest.csv / crops.csv  # 不变的正式清单
├── dataset_reader.py / read_example.py       # 稳定读取入口
├── HANDOFF_FOR_COLLEAGUE.txt / local_dataset_summary.json
├── scripts/          # 准备、预取、交接说明、滚动审核及对应检查脚本
├── jobs/             # 仅 testqueue 的提交入口
├── media/            # 本次下载的原视频和参考图；不进入 Git
├── metadata/         # 候选、哈希、来源、技术记录、分数、排除原因和状态
├── logs/             # 作业与重试历史
├── reviews/          # 已交付 50 对 ZIP、冻结抽样、SHA256、上传收据
└── archive/          # 历史过程说明
```

已删除可重建 smoke 素材、重复预取暂存、字节码缓存及被撤销的验收/打包/上传旧流水线；正式清单内容和媒体路径保持不变。审核包默认使用无评分表的连续滚动形式，规范见项目根目录 `DATASET_AUDIT_GUIDE.md`。Git 只追踪源码和说明，不包含原媒体、生成清单、日志、凭据和 ZIP。

后续审核包上传默认只传 ZIP，不单独传 JSONL 或哈希文件，不追加哈希、远端大小核对；上传工具报告成功即完成。历史已上传 ZIP、sidecar 和回执原样保留。该规则不删除本地用于来源追溯和分组隔离的媒体哈希。
