# 数据说明

## MovieLens-1M

| 项目 | 值 |
| --- | --- |
| 来源 | GroupLens Research，https://files.grouplens.org/datasets/movielens/ml-1m.zip |
| 规模 | 1,000,209 条评分 / 6,040 用户 / 3,706 部电影 |
| 时间跨度 | 2000-04-25 ~ 2003-02-28 |
| 评分 | 1–5 星（整数） |
| 许可 | 仅限研究用途，详见 `ml-1m/README` |

仓库内提交的是官方压缩包 `ml-1m.zip`（5.9 MB），运行
`python scripts/prepare_data.py` 会校验 SHA-256 后解压：

```
sha256 = a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20
```

## 为什么不用 Criteo / Avazu

CTR 预估最常用的是 Criteo 与 Avazu 的广告日志，但它们要么需要登录、要么直链失效，
且体量在 GB 级，不适合放进一个"可完整复现、几分钟跑完"的仓库。

本项目的取舍是：**用 MovieLens 的评分行为构造曝光级 CTR 任务**——
把"用户是否在测试窗口里看过这部电影"当作二分类标签，负例从未交互物品中采样。
这样既能训练 FM/DeepFM，也能顺带把"负例采样会改变什么"讲清楚（见报告第 4 节）。
代价是特征里没有真实的广告竞价、出价、上下文信息，这一点在报告的"局限"里写明。