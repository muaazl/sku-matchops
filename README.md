# SKU MatchOps — Intelligent Domain-Aware SKU Matching & Classification Platform

SKU MatchOps is a high-throughput, domain-aware catalog reconciliation and entity extraction platform. It pairs hybrid multi-stage retrieval (dense/sparse vector search + typo-tolerant lexical search) with cross-encoder re-ranking, zero-shot entity recognition (NER), and a deterministic business rules engine into a microservice-oriented architecture.

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
                   │ (HTTP:6333 / gRPC:6334)
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

### 3. Ingest Demo Sample Data (Instant Offline Mode)
To seed the catalog, train classifiers, and vectorize embeddings immediately from the pre-packaged sample dataset:
```bash
docker compose exec engine python -m engine.sync_catalog --sample
```
*(Optional: Run `docker compose restart backend` to immediately refresh backend memory caches with the seeded catalog).*

### 4. Access Services
- **Web Dashboard**: [http://localhost:5173](http://localhost:5173)
- **Backend API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ML Engine API Docs**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **Qdrant Web Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

## Data Ingestion & Google Sheets Integration

SKU MatchOps supports two flexible modes of catalog ingestion:

### Option A: Pre-Packaged Sample Dataset (`SampleData.xlsx`)
The repository includes a ready-to-run demo dataset at [`data/sample/SampleData.xlsx`](data/sample/SampleData.xlsx) containing 500 Food dishes, 500 Market retail products, and taxonomy dictionaries.
To reset and load this sample data into the system:
```bash
docker compose exec engine python -m engine.sync_catalog --sample
```

### Option B: Live Google Sheets Integration
You can connect your own Google Sheet catalog by following these steps:

1. **Create the Google Sheet**:
   - Upload [`data/sample/SampleData.xlsx`](data/sample/SampleData.xlsx) to Google Drive and open it as a Google Sheet (or format your sheet tabs using the schema below).
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
     docker compose exec engine python -m engine.sync_catalog
     ```
     *(Or locally outside Docker: `python -m engine.sync_catalog`)*

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
- Node.js 18+ and `pnpm` (install via `npm install -g pnpm`)
- Running instances of **Qdrant** (`localhost:6333`) and **Meilisearch** (`localhost:7700`). You can start just these two database containers using Docker:
  ```bash
  docker compose up -d qdrant meilisearch
  ```

### 2. Python Environment Setup
```bash
# Create and activate virtual environment (using Python 3.11)
py -3.11 -m venv venv        # On Linux: python3.11 -m venv venv
.\venv\Scripts\Activate.ps1  # On Linux: source venv/bin/activate

# Install CPU PyTorch first (fast & lightweight)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r backend/requirements.txt
pip install -r engine/requirements.txt

# Ingest sample catalog data & prepare ONNX models
python -m engine.sync_catalog --sample
```

### 3. Run Microservices
```bash
# Terminal 1 — Start ML Inference Engine
uvicorn engine.server:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2 — Start Backend API Gateway
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3 — Start Frontend Development Server
cd frontend
npm install -g pnpm  # Install pnpm if not already installed
pnpm install
pnpm start
```

---

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENGINE_URL` | `http://localhost:8001` | URL of the ML inference engine microservice. |
| `BACKEND_INTERNAL_URL` | `http://localhost:8000` | Internal backend gateway URL. |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector database URL. |
| `MEILI_URL` | `http://localhost:7700` | Meilisearch server URL. |
| `MEILI_MASTER_KEY` | `bismillah` | Meilisearch API master key. |
| `USE_INT8_MODELS` | `true` | Enables INT8 dynamic quantization for CPU speedup. |
| `GOOGLE_SHEET_ID` | — | Google Sheet ID containing catalog and taxonomy tabs. |

---

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
