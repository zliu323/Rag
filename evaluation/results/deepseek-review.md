# DeepSeek 实际调用与拒答复核

本次用户已明确同意将 60 条虚构问题及对应示例上下文发送至 DeepSeek 官方接口。执行命令：

```powershell
.\.venv\Scripts\python.exe -m evaluation.suite --llm deepseek --strategy hybrid --output evaluation\results\deepseek.json --resume
```

原始数据见 [deepseek.json](deepseek.json)。模型为 `deepseek-flash`，接口为 `https://api.deepseek.com/chat/completions`。60 个问题全部成功，无 API 错误。检索使用 HashingEmbedder + BM25/RRF，Top-3，无 Cross-Encoder，语料为 6 份示例文档、15 个非复制片段。DeepSeek 只负责生成答案。

## 自动统计

- 50 条有答案题：Hit@3、证据召回率均为 100%，MRR 为 0.9867。
- 50 条有答案题：关键词覆盖率 100%，引用编号格式有效率 100%。这两个指标不验证完整语义正确性和引用支持关系。
- 10 条无答案题：原始中文正则的拒答命中率为 90%（9/10）。
- 生成耗时：P50 1,684.35 ms，P95 1,980.06 ms，包含 API 网络等待；3 个并发请求。
- 提供商返回的累计输入 Token 为 29,786，输出 Token 为 2,047。未查询账户账单，未估算金额。

## AI 助手对无答案题的逐题复核

这是 AI 助手阅读已返回答案后进行的事后检查，没有再次调用模型，没有独立人工评审。原始 JSON、自动评分及中文规则均保留不变。

| ID | 所问内容 | 返回答案的行为 | 原始拒答得分 |
| --- | --- | --- | ---: |
| unknown01 | 年终奖月数 | 明确说资料没有年终奖信息，无法回答 | 1 |
| unknown02 | 海外住宿上限 | 说明资料只有境内标准，不能确定海外标准 | 1 |
| unknown03 | 标准版P1彻底修复时限 | 区分首次响应与彻底修复，说明没有修复时限保证 | 1 |
| unknown04 | embedding维度 | 说明资料仅提到向量与FAISS，没有维度数值 | 1 |
| unknown05 | 病假上限 | 说明没有病假规定，无法回答 | 1 |
| unknown06 | VPN服务器IP | 说明资料没有VPN IP信息，无法回答 | 1 |
| unknown07 | P0绩效扣减 | 说明没有扣减规定，不能从复盘规则确定金额 | 1 |
| unknown08 | 备份云商与地域 | 说明有备份频率与保留天数，没有存储位置 | 1 |
| unknown09 | 退款比例 | 区分退款流程与退款比例，说明没有计算规则 | 1 |
| unknown10 | 隔离环境密码 | 用英文明确说上下文没有指定密码或凭据信息 | 0 |

`unknown10` 的原始答案为：

> The context does not specify the login password for the isolated environment used by outsourced personnel. It only states that outsourced personnel may not download raw customer data and may use desensitized data only in an isolated environment [1]. No password or credential information is provided.

按“是否说明所问信息缺失”这一复核标准，10/10 均有拒答表达；但原始正则仅识别中文，导致 `unknown10` 漏判。该题还暴露出中文提问却英文回答的语言一致性缺陷。不能把这次事后复核解释为独立人工评测的 100% 正确率。

## 表述范围

这些是固定合成题集上的真实 API 实验结果，适合描述“离线评测、API 集成、可复现验证”。它们不支持生产用户收益、人工工时节省、生产吞吐或零幻觉的结论。
