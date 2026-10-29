# 企业知识库 RAG 智能问答系统

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

一个模块化、可运行、可评测的企业文档问答项目。系统支持文档摄取、递归分块、BM25 与向量混合检索、Cross-Encoder 重排、答案生成和可核验引用，并通过 FastAPI 对外提供服务。

仓库内置示例文档和 QA 数据，无需额外准备数据即可体验完整流程。默认使用无需 API Key 的抽取式生成器；也可以连接 DeepSeek API、OpenAI API、Ollama、vLLM、LM Studio 等服务。

## 功能特性

- 支持 PDF、TXT 和 Markdown 文档
- 保留文件名、PDF 页码、分块 ID 等来源元数据
- 使用 Sentence Transformers 与 FAISS 或 pgvector 进行向量检索
- BM25 与向量结果通过 RRF 融合；可通过配置关闭
- 使用 Cross-Encoder 对候选结果进行二阶段重排
- 支持抽取式生成、DeepSeek API 和 OpenAI-compatible LLM
- 返回答案及对应原文引用，便于核验
- 提供检索指标、答案关键词覆盖率和参数网格实验
- 提供文档上传、索引构建、查询和健康检查 API
- 可选 Langfuse 追踪检索、重排、生成的延迟与 Token 用量
- 可选 LangGraph 外层流程：检索为空或相关性不足时问题改写、最多一次重试和拒答
- 包含自动化测试、轻量冒烟测试、Docker 配置及 GitHub Actions CI

## 系统架构

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
 FAISS IndexFlatIP / pgvector HNSW ─┐
                                   ├─ RRF 融合
 BM25 关键词检索 ─────────────────────┘
        │ Top-K candidates
        ▼
 Cross-Encoder Reranker
        │ Top-N contexts
        ▼
 Extractive / DeepSeek / OpenAI-compatible LLM
        │
        ▼
 Answer + Citations
```

`/query/graph` 在上述 RAG 流程外增加控制层：

```text
检索 ──通过相关性初筛──→ 生成器返回结构化回答/拒答状态
  ├─未通过且可改写─→ 改写问题 → 再检索（最多一次）
  │                          ├─通过初筛─→ 生成器判定回答/拒答
  │                          └─仍未通过─→ 拒答
  └─未通过且不可改写─→ 拒答
```

`/query` 与 `/query/graph` 共用相关性初筛与结构化回答状态，均返回 `decision`。图流程额外提供最多一次改写重试，不需要增加模型或 API Key。

核心模块彼此解耦：

- `app/ingestion`：文档加载与递归分块
- `app/retrieval`：Embedding、FAISS/pgvector、BM25、RRF 与重排
- `app/observability.py`：可选 Langfuse Trace
- `app/generation`：Prompt 与可插拔生成器
- `app/rag`：RAG 流程编排
- `app/rag/graph_workflow.py`：LangGraph 外层决策与有界重试
- `evaluation`：检索和答案评测、参数实验
- `app/main.py`：FastAPI 接口

## 快速开始

### 1. 准备环境

需要 Python 3.11 或更高版本。

```bash
git clone https://github.com/zliu323/Rag.git
cd Rag
python -m venv .venv
```

激活虚拟环境：

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

安装依赖并创建配置文件：

```bash
pip install -r requirements.txt
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# macOS / Linux
cp .env.example .env
```

首次运行默认配置时，Sentence Transformers 会自动下载 Embedding 和 Reranker 模型。

### 2. 构建示例索引

```bash
python -m scripts.index_samples
```

### 3. 启动 API

```bash
uvicorn app.main:app --reload
```

服务启动后可访问：

- Swagger UI：<http://127.0.0.1:8000/docs>
- ReDoc：<http://127.0.0.1:8000/redoc>
- 健康检查：<http://127.0.0.1:8000/health>

## 轻量离线模式

如果只想快速验证流程，或暂时不下载模型，可在 `.env` 中使用确定性的 Hashing Embedding，并关闭 Reranker：

```dotenv
RAG_EMBEDDING_PROVIDER=hashing
RAG_RERANKER_ENABLED=false
RAG_LLM_PROVIDER=extractive
```

运行冒烟测试：

```bash
python -m scripts.smoke_test
```

Hashing 模式仅用于测试系统连通性，不适合作为生产语义检索方案或生产性能依据。

## 配置

所有配置均可通过 `.env` 文件设置，环境变量统一使用 `RAG_` 前缀。

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_EMBEDDING_PROVIDER` | `sentence-transformers` | Embedding 实现，可选 `sentence-transformers` 或 `hashing` |
| `RAG_VECTOR_BACKEND` | `faiss` | 向量后端：`faiss` 或 `pgvector` |
| `RAG_POSTGRES_DSN` | 本地示例 DSN | pgvector 数据库连接串 |
| `RAG_HYBRID_ENABLED` | `true` | 启用 BM25 + 向量检索 + RRF |
| `RAG_OBSERVABILITY_ENABLED` | `false` | 启用 Langfuse Trace |
| `RAG_OBSERVABILITY_CAPTURE_CONTENT` | `false` | 是否上传问题、上下文和答案全文；默认只记录元数据 |
| `RAG_LANGFUSE_PUBLIC_KEY` / `RAG_LANGFUSE_SECRET_KEY` | 空 | 启用观测时必填 |
| `RAG_LANGFUSE_BASE_URL` | Langfuse Cloud | Langfuse 地址 |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding 模型 |
| `RAG_EMBEDDING_DEVICE` | `cpu` | 模型运行设备 |
| `RAG_RERANKER_ENABLED` | `true` | 是否启用 Cross-Encoder 重排 |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker 模型 |
| `RAG_LLM_PROVIDER` | `extractive` | 生成器，可选 `extractive` 或 `openai-compatible` |
| `RAG_LLM_MODEL` | `gpt-4.1-mini` | OpenAI-compatible 模型名称 |
| `RAG_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API 地址 |
| `RAG_LLM_API_KEY` | 空 | API Key；抽取式生成器不需要 |
| `RAG_CHUNK_SIZE` | `700` | 文档分块大小 |
| `RAG_CHUNK_OVERLAP` | `120` | 相邻分块重叠大小 |
| `RAG_TOP_K` | `8` | 向量检索候选数量 |
| `RAG_RERANK_TOP_N` | `4` | 重排后保留的上下文数量 |
| `RAG_CONTEXT_MIN_COVERAGE` | `0.3` | 问题有效词在单个片段中的最低覆盖率；设为 `0` 关闭词汇初筛，空上下文仍拒答 |
| `RAG_MAX_UPLOAD_MB` | `20` | 单个上传文件的大小上限 |

完整配置示例见 [`.env.example`](.env.example)。

### LangGraph 外层流程

调用 `POST /query/graph`，请求体与 `/query` 相同：

```bash
curl -X POST http://127.0.0.1:8000/query/graph \
  -H "Content-Type: application/json" \
  -d '{"question":"请问 P0 事件需要在几分钟内通知负责人？","rerank":false}'
```

两个查询接口均返回 `decision`（`answered` 或 `refused`），图接口还返回 `retrieval_attempts` 和可选的 `rewritten_question`。图流程对检索为空或问题有效词覆盖不足的结果尝试清理礼貌表达并最多重试一次；没有实质改写则直接拒答。词汇初筛不直接使用 FAISS、RRF 或重排器的数值分数，避免混用不同量纲。

初筛是可配置的相关性启发式，不能证明片段足以回答问题；它可能漏掉同义表达或跨语言的语义匹配。这类业务可调低 `RAG_CONTEXT_MIN_COVERAGE` 或设为 `0`，再通过独立评测选择阈值。现有 50 条可回答扩展题在默认阈值下均通过初筛，这不代表未知业务问题的召回保证。

DeepSeek/OpenAI-compatible 生成时要求返回包含 `answer` 与 `decision` 的 JSON，程序严格验证状态与非空答案。模型拒答时统一返回“现有知识库内容不足以回答该问题。”且引用为空；不会再通过答案措辞判断拒答。格式不合法时返回提供商错误，不将其当作成功答案。对接的自定义模型需遵守该输出协议；新增生成器需实现 `generate_result()` 并显式返回 `GenerationResult`。直接调用提供商的 `generate()`/`generate_with_metadata()` 仍保留原来的纯文本输出接口。

### pgvector 后端

本地启动 PostgreSQL（需 Docker）：

```bash
docker compose -f docker-compose.pgvector.yml up -d
```

在 `.env` 设置：

```dotenv
RAG_VECTOR_BACKEND=pgvector
RAG_POSTGRES_DSN=postgresql://rag:rag@localhost:5432/rag
```

然后运行 `python -m scripts.index_samples` 和 `uvicorn app.main:app --reload`。建索引在事务内全量替换数据，并记录 Embedding 模型标识；切换模型时应使用新的数据库或重建索引。当前 BM25 索引由每个进程从数据库分块加载到内存，适合面试演示和单实例。多副本场景应将关键词检索迁入共享服务，并在索引版本变化时统一刷新。

### Langfuse 可观测性

在 `.env` 设置：

```dotenv
RAG_OBSERVABILITY_ENABLED=true
RAG_LANGFUSE_PUBLIC_KEY=pk-lf-...
RAG_LANGFUSE_SECRET_KEY=sk-lf-...
RAG_LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

每次查询会产生主 Trace，以及向量检索、BM25、重排和答案生成的子观测。检索与重排结果包含分块 ID、分数，主 Trace 记录 Embedding、Prompt 和索引版本。模型接口返回 Token 用量时会记录输入/输出 Token；成本取决于 Langfuse 中对应模型的定价配置。默认不上传问题、文档全文和答案，若数据策略允许，可设置 `RAG_OBSERVABILITY_CAPTURE_CONTENT=true`。本项目使用 Langfuse Python SDK v5，需要兼容的 Langfuse v4 服务。

启用后 `/query` 响应会包含 `trace_id`。前端可把用户评价关联到这次请求：

```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"trace_id":"从查询响应获取的32位ID","helpful":true,"comment":"引用准确"}'
```

未启用 Langfuse 时 `trace_id` 为 `null`，反馈接口返回 503。

## LLM Provider

### 抽取式生成

默认的 `extractive` 模式不需要 API Key。它会从检索上下文中抽取与问题最相关的句子，并在证据不足时拒绝回答。

### DeepSeek API

先在项目根目录创建 `.env`（可复制 `.env.example`），设置：

```dotenv
RAG_LLM_PROVIDER=deepseek
RAG_DEEPSEEK_MODEL=deepseek-flash
RAG_DEEPSEEK_BASE_URL=https://api.deepseek.com
RAG_DEEPSEEK_API_KEY=你的DeepSeek_API_Key
```

然后按前面的启动步骤运行服务，并向 `/query` 提问。DeepSeek 只负责根据检索到的片段生成回答；文档解析、Embedding、FAISS/pgvector 检索和 Rerank 的配置不变。`deepseek-flash` 是当前官方 API 的模型标识；也可将 `RAG_DEEPSEEK_MODEL` 设为 `deepseek-v4-pro`。此模式默认关闭 DeepSeek 的思考模式，超时为 120 秒。不要将填有密钥的 `.env` 提交到 Git。

### OpenAI-compatible 服务

连接 OpenAI API：

```dotenv
RAG_LLM_PROVIDER=openai-compatible
RAG_LLM_BASE_URL=https://api.openai.com/v1
RAG_LLM_MODEL=gpt-4.1-mini
RAG_LLM_API_KEY=your-api-key
```

连接本地 Ollama：

```dotenv
RAG_LLM_PROVIDER=openai-compatible
RAG_LLM_BASE_URL=http://localhost:11434/v1
RAG_LLM_MODEL=qwen2.5:7b
RAG_LLM_API_KEY=
```

生成参数固定为 `temperature=0`。Prompt 要求模型只依据给定上下文回答，并在证据不足时明确说明。

## API 使用

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

### 上传文档

支持一次上传一个或多个 PDF、TXT 或 Markdown 文件：

```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "files=@policy.pdf" \
  -F "files=@handbook.md"
```

### 构建索引

对内置样例和已上传文件重建索引：

```bash
curl -X POST http://127.0.0.1:8000/index \
  -H "Content-Type: application/json" \
  -d '{"include_samples":true,"include_uploads":true}'
```

### 查询

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"正式员工有多少天年假？","top_k":8,"rerank":true}'
```

查询响应中的 `citations` 包含：

- `source`：来源文件
- `page`：PDF 页码
- `chunk_id`：分块 ID
- `score`：最终相关性分数
- `retrieval_score`：重排前的检索分数
- `text`：引用原文
- `trace_id`：启用 Langfuse 时用于定位请求和提交反馈

引用信息来自文档摄取阶段保留的元数据，而不是由模型自行生成。

## 评测

新增 60 题扩展评测、逐题结果及 DeepSeek 实际调用说明见 [evaluation/README.md](evaluation/README.md)。默认运行 `python -m evaluation.suite --output evaluation/results/offline.json`；它使用隔离的内存索引，并对比纯向量与 BM25/RRF 检索。

运行标准评测：

```bash
python -m evaluation.evaluate --k 5 --output reports/evaluation.json
```

运行无需下载模型的轻量评测：

```bash
python -m evaluation.evaluate --lightweight --k 5 --output reports/evaluation-lightweight.json
```

旧评测与参数网格也使用独立内存 FAISS 索引：不读取或保存应用索引、不连接配置中的业务 pgvector，并关闭业务追踪。评测默认使用本地抽取式生成，不继承 `.env` 中的外部 LLM 开关。需要实际 API 生成时显式添加 `--llm deepseek` 或 `--llm openai-compatible`，例如 `python -m evaluation.evaluate --lightweight --llm deepseek`。参数网格只测检索，不调用外部 LLM。

当前包含以下指标：

- `Hit@K`：前 K 个结果中是否至少包含一个相关片段
- `Recall@K`：前 K 个结果覆盖的相关来源比例
- `MRR`：首个相关片段排名的倒数均值
- `answer_correctness`：标准答案关键词在回答中的覆盖比例

评测数据位于 `sample_data/evaluation/qa.json`，可以替换或扩充为真实业务测试集。

运行参数网格实验：

```bash
python -m evaluation.experiment --output reports/experiments.json
```

无需下载模型时添加 `--lightweight`：

```bash
python -m evaluation.experiment --lightweight --output reports/experiments-lightweight.json
```

实验会比较不同的 `chunk_size`、`overlap`、`top_k` 和 `rerank` 组合。内置 QA 数量较少，生产参数应基于具有代表性的业务数据进行选择。

运行可复现的本地检索负载基准：

```bash
python -m evaluation.benchmark --multiplier 2000 --repeats 500 --output reports/benchmark.json
```

基准会对比纯向量与混合检索，输出索引耗时、P50/P95/P99 延迟、顺序吞吐量和检索指标。`multiplier` 通过重复示例分块构造合成负载，因此结果只能表述为本地合成压测，不能解读为生产流量或业务收益。

## 测试与代码检查

```bash
pytest --cov=app --cov=evaluation
ruff check .
python -m scripts.smoke_test
```

测试覆盖文档加载、分块元数据、BM25/RRF、评测指标、轻量端到端流程，以及 FastAPI 的 `health → index → query` 主路径。设置 `RAG_TEST_POSTGRES_DSN` 后还会运行 pgvector 集成测试；GitHub Actions 通过独立 PostgreSQL 服务执行该测试，并完成 lint、覆盖率测试、冒烟测试和 Docker 构建。

## Docker

构建镜像：

```bash
docker build -t rag-knowledge-base .
```

启动容器：

```bash
docker run --rm -p 8000:8000 --env-file .env \
  -v rag-index:/service/data/index \
  rag-knowledge-base
```

模型首次下载会增加启动时间。生产部署可在镜像构建阶段预下载模型，或挂载模型缓存卷。

## 项目结构

```text
app/
├── ingestion/       # 文档加载与递归分块
├── retrieval/       # Embedding、FAISS/pgvector、BM25、RRF 与 Reranker
├── generation/      # Prompt 与可插拔生成器
├── rag/             # RAG 流程编排
├── config.py        # 环境配置
├── domain.py        # 核心数据类型
├── exceptions.py    # 应用异常
├── logging.py       # 日志配置
├── schemas.py       # API 数据结构
└── main.py          # FastAPI 接口
evaluation/          # 评测指标与参数实验
sample_data/         # 示例文档和 QA 数据
scripts/             # 索引与冒烟测试脚本
tests/               # 单元测试和 API 测试
```

## 已知限制

- 当前索引重建为全量替换，尚未实现增量更新和文档删除同步
- PDF 暂不包含 OCR，扫描件需要先进行文字识别
- FAISS 本地索引适合单实例；pgvector 可共享向量，但 BM25 仍为进程内索引
- 生产环境仍需按业务增加鉴权、租户隔离、恶意文件扫描、任务队列、告警和 PII 脱敏
- 抽取式生成器用于保证无 API Key 时流程可运行，不等同于通用 LLM

## License

本项目基于 [MIT License](LICENSE) 开源。
