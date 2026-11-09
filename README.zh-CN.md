# HiSearch

面向知识问答的混合检索智能体：共享策略自主选择 dense 语义检索、shell 直接语料搜索、检索范围与停止时机，训练流程包含多轨迹 SFT 和可选的 rubric-based GSPO。

完整安装和运行命令见 [英文 README](README.md)。

## 实现主线

- **混合检索：**`dense_search(query)` 固定返回 top-3；`shell(command, scope)` 可以搜索完整语料或同一 rollout 内 dense 返回的候选集。两者使用同一 Wikipedia 快照，候选集保留原始 JSONL 字节并隔离清理，没有额外路由模型，也不要求每题混用工具。
- **多轨迹 SFT：**teacher 真实执行自由选择、dense 起手、shell 起手以及两类单工具策略。通过答案与可见证据审查后，导出带双工具 schema 的 messages；同一问题的轨迹留在同一数据划分。自动审查仍有漏判边界，需要人工复核。
- **Rubric RL：**沿用 GRPO 组优势与 GSPO loss，合法轨迹奖励严格为 `F1 + 0.1 × 五条标准中的通过数`，上限 1.5。每个 outer step 与全部补采冻结同一版本，更新后至多修订一条，通过语义与独立训练参考面板检查后于下一步生效。

## 当前验证范围

混合检索、SFT 和纯结果奖励 RL 有历史运行记录。动态 rubric 路径完成 CPU/mock 接线测试；真实裁判校准、正式 rubric RL 训练收益和匹配消融仍待验证。附带五条标准是人工 bootstrap 模板。参见[发布状态](docs/release_status.md)。

## 快速离线检查

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-offline.txt
export PYTHONPATH="$PWD:$PWD/rl:$PWD/verl:${PYTHONPATH:-}"
bash run_gspo_hybrid_rubric.sh --check-only
python -m unittest rl.tests.test_rubric_workflow \
  rl.tests.test_rubric_launcher rl.tests.test_rubric_cli -v
```

这些命令无需 GPU，不启动训练或服务。完整流程依次参见[训练环境](TRAINING_ENV.md)、[检索服务](retrieval/README.md)、[SFT](sft/README.md)、[RL](rl/README.md)和[五条动态 rubric](docs/rubric_rl.md)。

本仓库只发布实现源码、配置、提示与测试。模型权重、数据集、语料、FAISS 索引、生成轨迹、日志、缓存、实验历史和凭据均不随源码上传。大型资源需自行下载或生成。训练环境与检索环境分开安装，所有机器路径、服务地址和可选鉴权通过配置传入。

代码基于 GrepSeek 和 verl，保留原项目归属与 Apache-2.0 许可，详见 [NOTICE](NOTICE) 与[来源说明](docs/PROVENANCE.md)。
