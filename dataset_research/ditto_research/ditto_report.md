# Ditto 随机 pair 阶段结论

## 结论

Ditto 当前不适合作为人物属性编辑数据的优先来源。本次在可低成本获取的 `local/0130` 子总体中随机抽 15 对，只有 1 条涉及人物配饰（手表改怀表），没有身份、人脸、头发、服装、首饰或表情编辑。

## 方法

- population：`local/0130` 的 472 个官方 metadata triplet。
- sampling：seed `20260821`，无放回均匀随机 15 条。
- acquisition：顺序读取 `.01` 开头约 785 MB，只保存命中的 30 个原始 MP4。
- review：H.264 左右 comparison、instruction 文本和六帧 contact sheet。

这个结果只代表 `local/0130` 子总体，不能作为 Ditto 全量精确比例。官网展示案例没有公开随机选择过程，未纳入审核。

## 审核汇总

| 字段 | 通过 | 不通过 | 存疑 |
| --- | ---: | ---: | ---: |
| person relevance | 1 | 14 | 0 |
| instruction correctness | 11 | 2 | 2 |
| logical correctness | 13 | 2 | 0 |
| content preservation | 10 | 5 | 0 |
| temporal consistency | 14 | 1 | 0 |
| visual quality | 8 | 3 | 4 |
| technical quality | 15 | 0 | 0 |

主要问题包括：合成对象像扁平贴图；目标修改范围超出 instruction；删除后引入新对象；非编辑背景被整体变黑；添加物尺度或交互不合理。唯一人物相关样本把腕表改成带银链怀表，但怀表仍贴在手腕上，佩戴逻辑和融合较生硬。

## 判断

- 优点：原始 pair 规格整齐，全部 1280×720、101 帧、20 fps，技术文件完整。
- 缺点：人物属性密度低，合成痕迹和内容保持问题明显。
- 决策：保留这 15 对作为 Ditto 的随机观察证据，不继续扩大下载，也不纳入当前人物属性训练候选。

官方来源：https://huggingface.co/datasets/QingyanBai/Ditto-1M

