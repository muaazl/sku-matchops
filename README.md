# SKU MatchOps

SKU MatchOps is an Automated Tagging Solution for PickMe Food and Market Products. Built on a microservice architecture combining hybrid vector search, cross-encoder re-ranking, zero-shot NER, and a deterministic rules engine.

---

https://github.com/user-attachments/assets/c2354e4b-112e-4a6a-9045-d3391d1b29be

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
                      │  • SQLite Audit Trail / WAL Cache       │
                      └────────────┬───────────────┬────────────┘
                                   │               │
            ┌──────────────────────┘               └──────────────────────┐
            │ HTTP:8001                                                   │ HTTP:7700
            ▼                                                             ▼
┌──────────────────────────────────────┐                       ┌──────────────────────┐
│  ML Inference Engine (FastAPI/ONNX)  │                       │ Meilisearch Engine   │
│        http://localhost:8001         │                       │ (Typo-Tolerant BM25) │
│  • BGE-M3 Dense + Sparse Vectors     │                       │     Port 7700        │
│  • BGE-Reranker-v2-M3 Cross-Encoder  │                       └──────────────────────┘
│  • GLiNER Zero-Shot Named Entity Rec │
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
| **`engine`** | `8001` | Dedicated ML inference microservice serving INT8 ONNX models and zero-shot entity extractors. |
| **`qdrant`** | `6333` | Vector database managing BGE-M3 dense and sparse embeddings with HNSW indexing. |
| **`meilisearch`** | `7700` | Typo-tolerant lexical search engine for high-speed catalog candidate retrieval. |

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
> **Development vs. hardened networking.** `docker compose up` also loads `docker-compose.override.yml`, which publishes internal service ports (Qdrant `6333/6334`, Meilisearch `7700`, engine `8001`) to the host for local debugging. For a hardened deployment, skip the override so only the backend (`8000`) and frontend (`5173`) are exposed:
>
> ```bash
> docker compose -f docker-compose.yml up -d
> ```
>
> Services still communicate over the private compose network; use `docker compose exec <service> …` to inspect internal ones.

### 3. Ingest Demo Sample Data (Instant Offline Mode)
To seed the catalog, train classifiers, and vectorize embeddings from the pre-packaged sample dataset:
```bash
docker compose exec engine python scripts/catalog/sync.py --sample
```
*(Optional: Run `docker compose restart backend` to refresh backend memory caches with the seeded catalog.)*

> [!IMPORTANT]
> **First-Run Model Downloads**: On the very first run, the system automatically downloads **BGE-M3**, **BGE-Reranker-v2-M3**, and **GLiNER Medium** (~2.5 GB total) and applies INT8 dynamic quantization. This initial startup will take a few minutes depending on bandwidth and CPU. Subsequent runs use cached models and initialize almost instantly.

### 4. Access Services
- **Web Dashboard**: [http://localhost:5173](http://localhost:5173)
- **Backend API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ML Engine API Docs**: [http://localhost:8001/docs](http://localhost:8001/docs) *(dev mode only)*
- **Qdrant Web Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard) *(dev mode only)*

---

## Data Ingestion & Google Sheets Integration

SKU MatchOps supports two flexible modes of catalog ingestion:

### Option A: Pre-Packaged Sample Dataset (`SampleData.xlsx`)
The repository includes a ready-to-run demo dataset at [`data/sample/SampleData.xlsx`](data/sample/SampleData.xlsx) containing 500 Food dishes, 500 Market retail products, and taxonomy dictionaries.
To reset and load this sample data into the system:
```bash
docker compose exec engine python scripts/catalog/sync.py --sample
```

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
   - Run the catalog sync command to fetch from Google Sheets, index into Meilisearch, vectorize embeddings into Qdrant, and train classifiers:
     ```bash
     docker compose exec engine python scripts/catalog/sync.py
     ```
     *(Or locally outside Docker: `python scripts/catalog/sync.py`)*

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
- Node.js 18+ and `pnpm` (install via `npm install -g pnpm`)
- Running instances of **Qdrant** (`localhost:6333`) and **Meilisearch** (`localhost:7700`). You can start just these two database containers using Docker:
  ```bash
  docker compose up -d qdrant meilisearch
  ```

### 2. Python Environment Setup
```bash
# Create and activate virtual environment (using Python 3.11)
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# Install CPU PyTorch first (fast & lightweight)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install service dependencies
uv pip install -r backend/requirements.txt
uv pip install -r engine/requirements.txt

# Ingest sample catalog data & prepare ONNX models
python scripts/catalog/sync.py --sample
```

### 3. Run Microservices
```bash
# Terminal 1 — Start ML Inference Engine
.\venv\Scripts\Activate.ps1
uvicorn engine.server:app --host 0.0.0.0 --port 8001

# Terminal 2 — Start Backend API Gateway
.\venv\Scripts\Activate.ps1
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
| `ENGINE_URL` | Yes | URL of the ML inference engine microservice. |
| `BACKEND_INTERNAL_URL` | Yes | Internal backend gateway URL. |
| `QDRANT_URL` | Yes | Qdrant vector database URL. |
| `QDRANT_API_KEY` | No | Qdrant API key (for secured/cloud deployments). |
| `MEILI_URL` | Yes | Meilisearch server URL. |
| `MEILI_MASTER_KEY` | Yes | Meilisearch API master key. |
| `USE_INT8_MODELS` | No | Enables INT8 dynamic quantization for CPU speedup. Defaults to `true`. |
| `FOOD_BT_MODEL` | No | BasicType classifier for Food (`arcface` or `logreg`). Defaults to `arcface`. |
| `MARKET_BT_MODEL` | No | BasicType classifier for Market (`arcface` or `logreg`). Defaults to `arcface`. |
| `APPS_SCRIPT_URL` | No | Google Apps Script webhook URL for push-based catalog sync. |
| `ENABLE_TUNNEL` | No | Set to `true` to auto-launch a Cloudflare quick tunnel on startup. |

---

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
