# 企业知识库 RAG 智能问答系统

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

一个模块化、可运行、可评测的企业文档问答项目。系统支持文档摄取、递归分块、向量检索、Cross-Encoder 重排、答案生成和可核验引用，并通过 FastAPI 对外提供服务。

仓库内置示例文档和 QA 数据，无需额外准备数据即可体验完整流程。默认使用无需 API Key 的抽取式生成器；也可以连接 OpenAI API、Ollama、vLLM、LM Studio 等 OpenAI-compatible 服务。

## 功能特性

- 支持 PDF、TXT 和 Markdown 文档
- 保留文件名、PDF 页码、分块 ID 等来源元数据
- 使用 Sentence Transformers 与 FAISS 进行向量检索
- 使用 Cross-Encoder 对候选结果进行二阶段重排
- 支持抽取式生成和 OpenAI-compatible LLM
- 返回答案及对应原文引用，便于核验
- 提供检索指标、答案关键词覆盖率和参数网格实验
- 提供文档上传、索引构建、查询和健康检查 API
- 包含自动化测试、轻量冒烟测试和 Docker 部署配置

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
 FAISS IndexFlatIP（归一化向量余弦检索）
        │ Top-K candidates
        ▼
 Cross-Encoder Reranker
        │ Top-N contexts
        ▼
 Extractive / OpenAI-compatible LLM
        │
        ▼
 Answer + Citations
```

核心模块彼此解耦：

- `app/ingestion`：文档加载与递归分块
- `app/retrieval`：Embedding、FAISS 向量索引与重排
- `app/generation`：Prompt 与可插拔生成器
- `app/rag`：RAG 流程编排
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
| `RAG_MAX_UPLOAD_MB` | `20` | 单个上传文件的大小上限 |

完整配置示例见 [`.env.example`](.env.example)。

## LLM Provider

### 抽取式生成

默认的 `extractive` 模式不需要 API Key。它会从检索上下文中抽取与问题最相关的句子，并在证据不足时拒绝回答。

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

引用信息来自文档摄取阶段保留的元数据，而不是由模型自行生成。

## 评测

运行标准评测：

```bash
python -m evaluation.evaluate --k 5 --output reports/evaluation.json
```

运行无需下载模型的轻量评测：

```bash
python -m evaluation.evaluate --lightweight --k 5 --output reports/evaluation-lightweight.json
```

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

## 测试与代码检查

```bash
pytest --cov=app --cov=evaluation
ruff check .
python -m scripts.smoke_test
```

测试覆盖文档加载、分块元数据、评测指标、轻量端到端流程，以及 FastAPI 的 `health → index → query` 主路径。

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
├── retrieval/       # Embedding、FAISS 与 Reranker
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
- FAISS 本地索引适合单实例；多副本部署需要统一分发版本化索引
- 生产环境仍需按业务增加鉴权、租户隔离、恶意文件扫描、任务队列、监控和 PII 脱敏
- 抽取式生成器用于保证无 API Key 时流程可运行，不等同于通用 LLM

## License

本项目基于 [MIT License](LICENSE) 开源。
