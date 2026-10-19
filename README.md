# 企业知识库 RAG 智能问答系统

一个完整、可运行、可评测的企业文档问答项目。文档摄取、Chunking、Embedding、FAISS、Reranker、生成层、API 和评测彼此解耦，并保留来源元数据以返回可核验引用。

项目内置 3 份示例文档和 4 条 QA，无需准备数据即可运行。默认生成层是**无 API Key 的抽取式 fallback**；Embedding 与 Reranker 默认使用 `sentence-transformers`。如果只是离线冒烟测试，可切到确定性的 hashing embedding。

> 本项目不附带伪造的性能指标。`evaluation` 与 `experiment` 会基于本机实际运行结果生成报告。

## 架构

```text
PDF / TXT / Markdown
        │
        ▼
 Loader（保留文件名、PDF 页码）
        │
        ▼
 Recursive Chunking（chunk_size / overlap）
        │
        ▼
 SentenceTransformer Embedding
        │
        ▼
 FAISS IndexFlatIP（归一化向量余弦检索）
        │ Top-K candidates
        ▼
 Cross-Encoder Reranker
        │ Top-N contexts
        ▼
 可插拔 LLM Provider ── Extractive / OpenAI-compatible
        │
        ▼
 Answer + [source, page, chunk_id, scores]
```

关键设计：

- `app/ingestion`：支持 PDF、TXT、Markdown；PDF 按页读取，来源元数据不丢失。
- `app/retrieval`：Embedding 接口、FAISS 持久化、Cross-Encoder 重排相互独立。
- `app/generation`：无 Key fallback 与 OpenAI-compatible provider；可连接 OpenAI API、Ollama、vLLM 或 LM Studio。
- `app/rag/pipeline.py`：只负责 orchestration，便于测试和替换实现。
- `evaluation`：检索指标、答案关键词覆盖、参数网格实验。
- `FastAPI`：上传、建索引、查询、健康检查，包含输入约束、文件类型/大小校验和统一错误响应。

## 快速开始

要求 Python 3.11+。首次使用 sentence-transformers 会下载两个模型。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
python -m scripts.index_samples
uvicorn app.main:app --reload
```

打开 `http://127.0.0.1:8000/docs` 使用交互式 API 文档。

### 无模型下载的轻量模式

在 `.env` 中设置：

```dotenv
RAG_EMBEDDING_PROVIDER=hashing
RAG_RERANKER_ENABLED=false
RAG_LLM_PROVIDER=extractive
```

然后运行 `python -m scripts.smoke_test`。Hashing 模式仅用于测试系统连通性，不应作为生产语义检索模型或简历性能结论。

## API 示例

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

上传文档（可多文件）：

```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "files=@policy.pdf" -F "files=@handbook.md"
```

对内置样例和上传文件重建索引：

```bash
curl -X POST http://127.0.0.1:8000/index \
  -H "Content-Type: application/json" \
  -d '{"include_samples":true,"include_uploads":true}'
```

查询：

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"正式员工有多少天年假？","top_k":8,"rerank":true}'
```

响应中的 `citations` 包含 `source`、PDF `page`、`chunk_id`、最终 `score`、重排前 `retrieval_score` 和原文。引用由系统元数据生成，而不是让模型自由编写来源。

## LLM Provider

默认 `extractive` 不需要 API Key。它从检索上下文中抽取与问题词项重合最高的句子，并附引用；上下文不足时明确拒答。

连接 OpenAI-compatible 服务：

```dotenv
RAG_LLM_PROVIDER=openai-compatible
RAG_LLM_BASE_URL=https://api.openai.com/v1
RAG_LLM_MODEL=gpt-4.1-mini
RAG_LLM_API_KEY=your-key
```

连接本地 Ollama（示例）：

```dotenv
RAG_LLM_PROVIDER=openai-compatible
RAG_LLM_BASE_URL=http://localhost:11434/v1
RAG_LLM_MODEL=qwen2.5:7b
RAG_LLM_API_KEY=
```

生成参数固定 `temperature=0`，Prompt 明确要求只依据上下文回答、证据不足时拒答，并使用 `[n]` 引用。

## 评测

标准评测：

```bash
python -m evaluation.evaluate --k 5 --output reports/evaluation.json
```

轻量离线评测：

```bash
python -m evaluation.evaluate --lightweight --k 5 --output reports/evaluation-lightweight.json
```

指标定义：

- `Hit@K`：前 K 个结果中是否至少出现一个相关片段。
- `Recall@K`：前 K 个结果覆盖的相关来源比例。
- `MRR`：首个相关片段排名的倒数，多条问题取均值。
- `answer_correctness`：标准答案关键词在回答中的覆盖比例。它透明、易复现，但不能代替语义正确性或人工评审。

QA 文件位于 `sample_data/evaluation/qa.json`。每条包含问题、相关来源、相关原文特征和答案关键词，方便增加真实业务测试集。

参数实验：

```bash
python -m evaluation.experiment --output reports/experiments.json
# 无模型下载：添加 --lightweight（此时 rerank 实验会自动关闭）
```

脚本真实比较：

- `chunk_size`: 350 / 700
- `overlap`: 60 / 120
- `top_k`: 3 / 5
- `rerank`: false / true

不要仅依据这 4 条示例 QA 选择生产参数；应扩充为覆盖事实查询、多跳查询、无答案问题和边界条件的代表性数据集。

## 测试与质量检查

```bash
pytest --cov=app --cov=evaluation
ruff check .
python -m scripts.smoke_test
```

测试覆盖文档加载、分块元数据、指标计算、轻量端到端管线，以及 FastAPI 的 health → index → query 主路径。

## Docker

```bash
docker build -t rag-knowledge-base .
docker run --rm -p 8000:8000 --env-file .env \
  -v rag-index:/service/data/index rag-knowledge-base
```

模型首次下载会增加启动时间。生产环境建议在镜像构建或模型缓存卷中预下载，并设置 CPU/内存限制。当前 FAISS 索引适合单实例；多副本部署应把索引构建做成独立任务，并通过对象存储或版本化制品分发一致索引。

## 目录结构

```text
app/
├── ingestion/       # loader + recursive splitter
├── retrieval/       # embedding + FAISS + reranker
├── generation/      # prompt + pluggable providers
├── rag/             # orchestration pipeline
├── config.py        # typed environment settings
├── domain.py        # core data types
├── exceptions.py    # expected application errors
├── logging.py       # structured-friendly logs
├── schemas.py       # API contracts
└── main.py          # FastAPI endpoints
evaluation/          # metrics, evaluation and grid experiment
sample_data/         # sample documents and QA labels
scripts/             # sample indexing and smoke test
tests/               # unit and API tests
```

## 面试可讲点

1. **为什么两阶段检索？** Bi-Encoder 可预计算文档向量，适合大规模召回；Cross-Encoder 对 query-document 联合编码，相关性更强但成本更高，因此只重排 Top-K。
2. **为什么 IndexFlatIP？** sentence-transformers 输出 L2 归一化向量后，内积等价于余弦相似度；Flat 索引是精确搜索，适合本项目的小数据。规模扩大后可评测 IVF/HNSW 的召回—延迟权衡。
3. **Chunk 怎么选？** 太小会丢上下文，太大导致语义稀释、重复 token 和延迟上升。不能凭感觉，应在代表性 QA 上联合搜索 chunk_size、overlap 和 top_k。
4. **引用如何防伪？** source/page/chunk_id 在摄取阶段产生并随检索结果返回；模型只看到编号上下文，API 最终引用来自受控元数据。
5. **如何评测？** 检索层看 Hit@K、Recall@K、MRR；生成层的关键词覆盖只作基础代理，还需无答案率、faithfulness、人工评分和线上反馈。
6. **为什么不用 LangChain？** 本项目刻意保留薄接口和显式数据流，更容易调试、测试和解释；需要复杂 agent/graph 时可在 pipeline 外层引入框架，而不是让核心逻辑被框架隐藏。
7. **生产化缺口？** 鉴权、病毒扫描、OCR、增量索引、租户隔离、任务队列、缓存、索引版本、观测与 PII 脱敏需要按业务补充。
8. **无答案怎么办？** Prompt 和 fallback 都支持拒答，但严谨系统还应增加检索分数校准/分类器，并在验证集上选择阈值。

## 安全与限制

- 上传文件使用随机前缀和安全文件名，限制类型与大小；生产环境仍应增加恶意文件扫描和 PDF 解析隔离。
- `.env` 与运行生成的索引默认不提交版本库。
- 当前重建索引是全量替换，适合演示与中小知识库。增量更新需要处理删除、重复文档、版本与并发读写。
- Extractive fallback 不是通用 LLM，只保证无 Key 时系统可完整跑通和核验引用。

## License

MIT（示例项目，可按需要替换）。

