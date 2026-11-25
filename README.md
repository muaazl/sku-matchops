# SKU MatchOps

SKU MatchOps is an Automated Tagging Solution for PickMe Food and Market Products. Built on a microservice architecture combining hybrid vector search, cross-encoder re-ranking, zero-shot NER, and a deterministic rules engine.

---

## Architecture Overview

SKU MatchOps is structured around a decoupled microservice architecture separating ML inference, API orchestration, and vector/lexical retrieval:

```
                      ┌─────────────────────────────────────────┐
                      │  Frontend Web App (React 18 + Vite SPA) │
                      │          http://localhost:5173          │
                      └────────────────────┬────────────────────┘
                                           │ HTTP:8000
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │  Backend API Gateway (FastAPI 0.110)    │
                      │          http://localhost:8000          │
                      │  • Business Rules Engine (AST)          │
                      │  • Batch Job Orchestration & Workers    │
                      └────────────┬───────────────┬────────────┘
                                   │               │
            ┌──────────────────────┘               └──────────────────────┐
            │ gRPC:50051                                                  │ SQL:5432
            ▼                                                             ▼
┌──────────────────────────────────────┐                       ┌──────────────────────┐
│  ML Inference Engine (Stateless/ONNX)│                       │ PostgreSQL 16 DB     │
│          gRPC Port 50051             │                       │ • pg_trgm GIN Search │
│  • Embedding + Reranker (INT8 ONNX)  │                       │ • Jobs & Audit Trail │
│  • Native BM25 / Optional BGE Sparse │                       │     Port 5432        │
│  • GLiNER Zero-Shot Named Entity Rec │                       └──────────────────────┘
│  • INT8 Dynamic Quantization Runtime │
└──────────────────┬───────────────────┘
                   │ (HTTP:6333)
                   ▼
┌──────────────────────────────────────┐
│        Qdrant Vector Database        │
│    (Dense + Sparse HNSW Indexing)    │
│              Port 6333               │
└──────────────────────────────────────┘
```

### Core Services

| Service | Port | Description |
| :--- | :--- | :--- |
| **`frontend`** | `5173` | React 18 + Vite single-page application with interactive SKU matching and rule curation. |
| **`backend`** | `8000` | FastAPI API gateway, deterministic rules engine, batch processing worker, and audit logger. |
| **`engine`** | `50051` | Dedicated ML inference microservice (gRPC) serving INT8 ONNX models and zero-shot entity extractors. |
| **`qdrant`** | `6333` | Vector database managing dense, native BM25, and optional sparse embeddings with HNSW indexing. |
| **`postgres`** | `5432` | PostgreSQL 16 for jobs, rules, and per-SKU audit results. Managed exclusively by the backend gateway via a pooled SQLAlchemy engine; schema is initialized automatically on startup. |

---

---

## Quickstart with Docker Compose

The entire 5-service stack boots automatically with 1 command, including automatic model downloading, ONNX export, and INT8 dynamic quantization.


### 1. Clone and Configure
```bash
git clone https://github.com/muaazl/sku-matchops.git
cd sku-matchops

# Copy environment template
cp .env.example .env
```

### 2. Start the Stack
```bash
docker compose up -d
```

> [!NOTE]
> **Development vs. hardened networking.** `docker compose up` also loads `docker-compose.override.yml`, which publishes internal service ports (Qdrant `6333`, engine gRPC `50051`) to the host for local debugging. For a hardened deployment, skip the override so only the backend (`8000`) and frontend (`5173`) are exposed:
>
> ```bash
> docker compose -f docker-compose.yml up -d
> ```
>
> Services still communicate over the private compose network; use `docker compose exec <service> …` to inspect internal ones.

### 3. Ingest Demo Sample Data (Instant Offline Mode)
To seed the PostgreSQL catalog (owned by `backend`), build local caches, train classifiers, and vectorize embeddings in `engine`:
```bash
docker compose exec backend python -m scripts.sync --sample --keep-staged --db rebuild
docker compose exec engine python -m scripts.sync --from-staged --cache rebuild --qdrant rebuild --arcface rebuild
```

> [!IMPORTANT]
> **First-Run Model Downloads**: On the very first run, the system automatically downloads the configured **`EMBEDDING_MODEL`**, **`RERANKER_MODEL`**, and **GLiNER Medium** and exports/quantizes them cleanly to ONNX. Subsequent runs use cached models in `data/models/` and initialize almost instantly.

### 4. Access Services
- **Web Dashboard**: [http://localhost:5173](http://localhost:5173)
- **Backend API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Qdrant Web Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard) *(dev mode only)*

---

## Data Ingestion & Google Sheets Integration

SKU MatchOps supports two flexible modes of catalog ingestion:

### Option A: Pre-Packaged Sample Dataset (`SampleData.xlsx`)
The repository includes a ready-to-run demo dataset at [`data/sample/SampleData.xlsx`](data/sample/SampleData.xlsx) containing 500 Food dishes, 500 Market retail products, and taxonomy dictionaries.
To reset and load this sample data into the system:
```bash
docker compose exec backend python -m scripts.sync --sample --keep-staged --db rebuild
docker compose exec engine python -m scripts.sync --from-staged --cache rebuild --qdrant rebuild --arcface rebuild
```
*(Or locally outside Docker: `python -m scripts.sync --sample`)*

### Option B: Live Google Sheets Integration
You can connect your own Google Sheet catalog by following these steps:

1. **Create the Google Sheet**:
   - Upload [`data/sample/SampleData.xlsx`](data/sample/SampleData.xlsx) to Google Drive and open it as a Google Sheet.
2. **Set Sharing**:
   - Set Sheet sharing permissions to **"Anyone with the link can view"**.
3. **Configure Environment**:
   - Copy the Sheet ID from your browser URL (`https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit`) and paste it into `.env`:
     ```env
     GOOGLE_SHEET_ID=your_extracted_sheet_id_here
     ```
4. **Trigger Sync**:
   - Run the catalog sync commands to stage from Google Sheets into PostgreSQL (`backend`) and vectorize embeddings + train classifiers (`engine`):
     ```bash
     docker compose exec backend python -m scripts.sync --keep-staged --db sync
     docker compose exec engine python -m scripts.sync --from-staged --cache sync --qdrant sync --arcface sync
     ```
     *(Or locally outside Docker: `python -m scripts.sync`)*

### Google Sheet Tab Schema

| Tab Name | Domain | Required Column Headers |
| :--- | :--- | :--- |
| **`Food Catalog`** | Food | `Name`, `Description`, `Flavor`, `Price`, `SellerCategory`, `GenericKeywords`, `BasicType`, `Region` |
| **`Market Catalog`** | Market | `Name`, `Description`, `Brand`, `Price`, `SellerCategory`, `Category`, `GenericKeywords`, `BasicType` |
| **`Food_Flavors`** | Food | `Flavor Name`, `Aliases`, `Is_Meat`, `Is_Vegetable`, `Is_Seafood` |
| **`Market_Brands`** | Market | `Brand Name`, `Aliases`, `Is_Weak` |
| **`Food_GK`** | Food | Column 1: Generic keyword list (e.g. `Rice and Curry`, `Beef Burger`) |
| **`Food_BT`** | Food | Column 1: Basic type list (e.g. `Rice`, `Burger`, `Kottu`) |
| **`Food_Region`** | Food | Column 1: Region list (e.g. `Sri Lankan`, `Western`, `Chinese`) |
| **`Market_GK`** | Market | Column 1: Generic keyword list (e.g. `Soft Drink`, `Butter`) |
| **`Market_BT`** | Market | Column 1: Basic type list (e.g. `Beverages`, `Dairy`) |
| **`Market_Cat`** | Market | Column 1: Category list (e.g. `Beverages`, `Dairy`, `Produce`) |

---

## Local Development (Without Docker)

If you prefer running the Python and Node services directly on your host machine:

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- `uv` (recommended for ultra-fast pip installs) or `pip`
- Node.js 22+ and `pnpm` (install via `npm install -g pnpm`)
- Running instances of **PostgreSQL** (`localhost:5432`), **Redis** (`localhost:6379`, job queue + rule cache), **Qdrant** (`localhost:6333`). You can start just these containers using Docker:
  ```bash
  docker compose up -d postgres redis qdrant
  ```
  Tables and sequences are auto-created when the backend starts up.
- Optional: `docker compose up -d pgadmin` for a Postgres web UI at http://localhost:5050.

### 2. Python Environment Setup
```bash
# Create and activate virtual environment (using Python 3.11)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install CPU PyTorch first (fast & lightweight, required for ML engine)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies across all services:
uv pip install -r backend/requirements.txt   # FastAPI Gateway, DB, Rules Engine
uv pip install -r engine/requirements.txt    # ML Engine, Qdrant, ONNX Runtime

# Ingest sample catalog data & prepare ONNX models
python -m scripts.sync --sample
```

### 3. Run Microservices
```bash
# Terminal 1 — Start ML Inference Engine (gRPC on port 50051)
.\.venv\Scripts\Activate.ps1
python -m engine.server

# Terminal 2 — Start Backend API Gateway
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 3 — Start Frontend Development Server
cd frontend
npm install -g pnpm  # Install pnpm if not already installed
pnpm install
pnpm start
```

---

## Environment Variables

| Variable | Required | Description |
| :--- | :--- | :--- |
| `GOOGLE_SHEET_ID` | Yes (live mode) | Google Sheet ID containing catalog and taxonomy tabs. |
| `DATABASE_URL` | Yes | PostgreSQL URL (`postgresql+psycopg://user:pass@host:5432/db`). |
| `DB_POOL_SIZE` / `DB_MAX_OVERFLOW` | No | SQLAlchemy connection pool per process. Defaults to `10` / `20`. |
| `DB_BULK_CHUNK_SIZE` | No | Rows per INSERT when bulk-writing per-SKU results. Defaults to `1000`. |
| `ENGINE_GRPC_URL` | Yes | gRPC target of the ML inference engine (`127.0.0.1:50051` or `engine:50051`). |
| `BACKEND_GRPC_URL` | Yes | gRPC callback target of the backend gateway (`127.0.0.1:50052` or `backend:50052`). |
| `INTERNAL_GRPC_SECRET` | No | Optional shared secret enforced on all internal Gateway <-> Engine gRPC calls. |
| `QDRANT_URL` | Yes | Qdrant vector database URL. |
| `QDRANT_API_KEY` | No | Qdrant API key (for secured/cloud deployments). |
| `EMBEDDING_MODEL` | No | HuggingFace embedding model ID (auto-exported to INT8 ONNX). Defaults to `jinaai/jina-embeddings-v5-text-small`. |
| `RERANKER_MODEL` | No | HuggingFace reranker model ID (auto-exported to INT8 ONNX). Defaults to `jinaai/jina-reranker-v3.5`. |
| `FOOD_BT_MODEL` | No | BasicType classifier for Food (`arcface` or `logreg`). Defaults to `arcface`. |
| `MARKET_BT_MODEL` | No | BasicType classifier for Market (`arcface` or `logreg`). Defaults to `arcface`. |
| `APPS_SCRIPT_URL` | No | Google Apps Script webhook URL for push-based catalog sync. |
| `ENABLE_TUNNEL` | No | Set to `true` to auto-launch a Cloudflare quick tunnel on startup. |

---

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
