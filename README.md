# Tempograph

<p align="center">
  <img src="./assets/logo-16-by-9.png" alt="Tempograph" width="920" />
</p>

**Agentic RAG with a temporal fact graph for Indonesian long-context chatbots**

Sistem RAG agentic di atas **SurrealDB**: graf temporal (episode → fakta → entitas), pencarian vektor pada fakta dan pada passage sesi, serta jalur **vanilla**, **agentic**, dan **hybrid**. Dataset longitudinal berbahasa Indonesia dapat dihasilkan dengan pipeline terpisah lalu di-ingest ke basis yang sama.

---

## Ringkasan

| Bagian | Peran |
|--------|--------|
| **Dataset** | `src/dataset/generator.py` — persona, graf event (`caused_by`), sesi multi-turn, ringkasan sesi, anotasi ground truth per giliran. |
| **RAG** | Ingest `scripts/ingest_agentic.py` → SurrealDB (`schema.surql`, 768-dim cosine). Evaluasi `scripts/evaluate_agentic.py` / `scripts/evaluate_vanilla.py`. |

Contoh data siap pakai ada di **`output/example_dataset/`** (`conversation_dataset.json`, `evaluation_queries_100.json`, dll.) sehingga alur Surreal dapat diuji tanpa menjalankan generator.

---

## Arsitektur

### Alur end-to-end

```mermaid
flowchart LR
    GEN[src/dataset/generator.py]
    JSON[(conversation_dataset.json)]
    ING[scripts/ingest_agentic.py]
    DB[(SurrealDB)]
    RET[src/retrieval]

    GEN --> JSON --> ING --> DB
    RET <--> DB
```

### Pipeline dataset

**Entri:** `src/dataset/generator.py` — `parse_args()`, `main()`. Membutuhkan **Google Gemini** (`GEMINI_API_KEY`).

1. **Persona** — file `--user-file` atau `--auto-generate-persona`.
2. **Life events** — daftar event bertanggal dengan **`caused_by`**; disimpan ke **`user_events.json`** di `--out-dir`.
3. **Sesi** — per tanggal sesi: `get_relevant_events` → `generate_conversation_session` → `generate_session_summary`; ringkasan kumulatif mengisi konteks berikutnya. Opsional **`--use-caching`** (context cache pada model dialog).
4. **Ground truth** — `generate_ground_truth_annotations`: per giliran `generate_ground_truth_for_turn`; mode inkremental memanggil `resolve_ground_truth_conflicts`. Keluaran utama: **`conversation_dataset.json`**.

Model Gemini per tahap dikonfigurasi lewat **variabel lingkungan** (lihat [Konfigurasi](#konfigurasi)); default mengikuti tier *structured* / *dialog* / *light* yang dipetakan di `.env.example`.

```mermaid
flowchart TB
    subgraph A1 [Persona dan dunia simulasi]
        UP[Profil user + secondary personas]
        EV[Life events + caused_by]
    end

    subgraph A2 [Loop tiap tanggal sesi]
        RV[Pilih event relevan]
        CV[Multi-turn]
        SM[Ringkasan sesi]
    end

    subgraph A3 [Setelah sesi selesai]
        HIST[conversation_history]
        GT[Ground truth per giliran]
        OUT[(conversation_dataset.json + user_events.json)]
    end

    UP --> EV --> RV --> CV --> SM
    SM --> RV
    CV --> HIST
    HIST --> GT --> OUT
```

```mermaid
flowchart LR
    subgraph causal [Graf event — field caused_by]
        E1((E1)) --> E2((E2))
        E3((E3)) --> E2
    end
```

### Ingest dan retrieval (SurrealDB)

**Ingest:** `scripts/ingest_agentic.py` — menulis episode, `extracted_fact`, `entity`, `community`, relasi **`has_fact`** / **`fact_involves`** / **`has_member`**, serta **`session_passage`** (vektor per sesi penuh untuk vanilla/hybrid). Skema dan indeks vektor: **`src/surreal/schema.surql`**.

**Skrip evaluasi:** `scripts/evaluate_agentic.py`, `scripts/evaluate_vanilla.py`, `scripts/test_agentic_hybrid_top3_questions.py` (SurrealDB harus dapat dijangkau sesuai `.env`).

```mermaid
flowchart TB
    Q[Query]

    subgraph retrieval [Retrieval]
        RA[RetrievalAgent]
        VR[VanillaRetriever]
        HY[HybridRetriever]
    end

    subgraph surreal [SurrealDB]
        FT[extracted_fact]
        SP[session_passage]
        EP[episode]
        EN[entity]
        CM[community]
        R1[has_fact]
        R2[fact_involves]
        R3[has_member]
    end

    Q --> RA
    Q --> HY
    HY --> RA
    HY --> VR
    RA --> FT
    RA --> R2
    VR --> SP
    EP --> R1 --> FT
    FT --> R2 --> EN
    CM --> R3 --> EN
```

```mermaid
erDiagram
    episode {
        string group_id
        string name
        string body
        datetime reference_time
    }
    extracted_fact {
        string group_id
        string fact_text
        string episode_name
        string embedding
        string entity_names
        datetime valid_at
        datetime invalid_at
    }
    entity {
        string group_id
        string name
    }
    community {
        string group_id
        string name
        string summary
    }
    session_passage {
        string collection
        string doc_id
        string text
        string embedding
        string metadata
    }
    episode ||--o{ extracted_fact : has_fact
    extracted_fact }o--o{ entity : fact_involves
    community ||--o{ entity : has_member
```

### Algoritma Graf Temporal Tingkat Lanjut

Untuk mengoptimalkan akurasi pada percakapan *multi-turn chatbot*, sistem ini menerapkan tiga algoritma canggih dari arsitektur Graphiti:

1. **Deduplikasi Entitas Dinamis (*Dynamic Entity Deduplication*)**:
   Penyatuan otomatis entitas alias atau variasi nama (misal: "Aisha" dan "Aisha Santoso") menggunakan metrik jarak teks (`rapidfuzz` untuk token sort & partial ratio) secara lokal. Jika kemiripan berada di batas menengah ($60\% - 84\%$), sistem meminta LLM mengklarifikasi kesamaan entitas secara kontekstual. Entitas terdeduplikasi digabung ke node yang sama di SurrealDB dengan nama terpanjang/terspesifik sebagai nama kanonis.
2. **Resolusi Kontradiksi & Invalidation Bi-temporal**:
   Pendeteksian konflik informasi baru secara real-time. Sebelum fakta disimpan, sistem membandingkan embedding fakta pada entitas yang sama menggunakan kemiripan kosinus. Jika kemiripan $\ge 0.35$ (topik sama), LLM mengklasifikasikan relasinya (*duplicate*, *contradictory*, atau *compatible*). Kontradiksi diselesaikan dengan menonaktifkan fakta lama secara temporal menggunakan atribut `invalid_at = reference_time` guna mendukung kueri sejarah bi-temporal chatbot.
3. **Deteksi Komunitas Graf (*Graph Community Detection*)**:
   Klusterisasi relasi sosial/pekerjaan pengguna secara periodik menggunakan algoritma partisi **Louvain** dari **NetworkX**. Kelompok entitas yang terbentuk kemudian dirangkum konteksnya oleh LLM menjadi `community` node yang terhubung via edge `has_member`, guna mendukung penyediaan memori makro pada turn berikutnya.

---

## Persyaratan

- Python 3.11+ (disarankan mengikuti `environment.example.yml`).
- **SurrealDB** untuk penyimpanan graf + vektor.
- **Gemini** untuk generator dataset dan untuk banyak jalur RAG (kunci + kuota).
- **Novita** (opsional): dipakai bila `LLM_PROVIDER=novita` untuk ekstraksi fakta agentic; kunci di `.env` (`NOVITAAI_API_KEY`).

---

## Instalasi

```bash
git clone <URL-repositori>
cd <direktori-repo>

conda env create -f environment.example.yml
conda activate tempograph

uv pip install -U -r requirements.txt

cp .env.example .env
# Isi GEMINI_API_KEY, SURREAL_*, dan bagian lain sesuai tabel di bawah.
```

Cek koneksi Surreal (dari root repo):

```bash
python scripts/run_with_local_surreal.py -- python scripts/test_surreal_connection.py
```

`run_with_local_surreal.py` dapat menjalankan proses Surreal lokal bila dikonfigurasi; lihat `--help` pada skrip tersebut.

---

## Konfigurasi

Salin `.env.example` ke `.env`. Ringkasan variabel penting:

### API dan penyedia

| Variabel | Keterangan |
|----------|------------|
| `GEMINI_API_KEY` | Wajib untuk generator dataset dan stack yang memakai Gemini. |
| `NOVITAAI_API_KEY` | Untuk `LLM_PROVIDER=novita` pada RAG agentic (OpenAI-compatible). |

### Stack RAG terpusat (`--setup env`)

Ingest dan evaluasi dapat memakai satu set variabel tanpa mengganti kode:

| Variabel | Nilai umum | Fungsi |
|----------|------------|--------|
| `LLM_PROVIDER` | `gemini` / `novita` | Ekstraksi fakta agentic (`gemini` = GenAI; `novita` = endpoint Novita). |
| `LLM_MODEL` | kosong = default per provider | ID model mengikuti provider. |
| `EMBED_PROVIDER` | `gemini` / `huggingface` | Embedding untuk fakta dan `session_passage`. |
| `EMBED_MODEL` | kosong = default per provider | ID model embedding. |
| `RAG_GROUP_ID` | mis. `agentic_default` | Partisi graf Surreal (`group_id`). |
| `RAG_SESSION_COLLECTION` | mis. `vanilla_default` | Nama koleksi logis untuk vektor passage sesi (vanilla / kaki vanilla hybrid). |
| `RAG_MODE` | `agentic` / `vanilla` / `hybrid` | Hanya dipakai dengan `python scripts/evaluate_agentic.py --setup env`. |

**Ingest dari env:** `python scripts/ingest_agentic.py --setup env` — `RAG_GROUP_ID` dan `RAG_SESSION_COLLECTION` harus konsisten dengan evaluasi nanti.

**Eval dari env:** `python scripts/evaluate_agentic.py --setup env` — atur `RAG_MODE` sesuai jalur yang diuji.

Preset tetap tersedia: `--setup gemini`, `gemma`, `gemini_hybrid`, `gemma_hybrid`, `vanilla_gemini`, `vanilla_gemma`, serta `ingest_agentic.py --setup gemini|gemma|all`.

### Model generator dataset (Gemini)

Tiga ID model terpisah (semua lewat API Gemini):

| Variabel | Peran dalam `generator.py` |
|----------|-----------------------------|
| `DATASET_GEMINI_MODEL_STRUCTURED` | Persona, life events, kelanjutan event (keluaran JSON terstruktur). |
| `DATASET_GEMINI_MODEL_DIALOG` | Sesi multi-turn; target model untuk context caching bila `--use-caching`. |
| `DATASET_GEMINI_MODEL_LIGHT` | Ringkasan sesi, ground truth per giliran, resolusi konflik fakta. |

Default mengacu ke nilai di `.env.example` bila variabel dikosongkan.

### SurrealDB

| Variabel | Keterangan |
|----------|------------|
| `SURREAL_URL` | Mis. `ws://127.0.0.1:8000` (tanpa `/rpc`; ditangani SDK). |
| `SURREAL_USER` / `SURREAL_PASS` | Otentikasi root atau pengguna terbatas. |
| `SURREAL_NS` / `SURREAL_DB` | Namespace dan database. |

Parameter lanjutan (path CLI, opsi storage lokal) ada di `.env.example`.

### Parameter lain

`src/config/settings.py` menggabungkan rate limit, retrieval, dan evaluasi; nilai dapat diisi dari lingkungan sesuai definisi dataclass di file tersebut.

---

## Operasi harian

### 1. Generate dataset

```bash
python src/dataset/generator.py \
  --out-dir ./data/dataset \
  --num-sessions 10 \
  --num-events 20 \
  --num-days 60
```

Opsi berguna: `--auto-generate-persona`, `--fresh-start`, `--use-caching`, `--min-turns-per-session` / `--max-turns-per-session`.

### 2. Ingest ke SurrealDB

**Satu stack dari `.env`:**

```bash
python scripts/run_with_local_surreal.py --no-start -- \
  python scripts/ingest_agentic.py --setup env --limit 10 --batch 10
```

**Preset:**

```bash
python scripts/run_with_local_surreal.py --no-start -- \
  python scripts/ingest_agentic.py --setup gemini --limit 10 --batch 10
```

`--clear` menghapus data sesuai `RAG_GROUP_ID` + `RAG_SESSION_COLLECTION` (mode `env`) atau preset yang dipilih.

### 3. Evaluasi

**Agentic / vanilla / hybrid lewat env:**

```bash
python scripts/run_with_local_surreal.py --no-start -- \
  python scripts/evaluate_agentic.py --setup env --limit 5 --no-llm-judge
```

**Preset:**

```bash
python scripts/run_with_local_surreal.py --no-start -- \
  python scripts/evaluate_agentic.py --setup gemini --limit 5 --no-llm-judge
```

Query evaluasi default membaca `output/example_dataset/evaluation_queries_100.json`; hasil ditulis ke `output/evaluation_results/`.

### 4. UI simulasi dataset (portfolio)

Simulasi **10 sesi** + graf interaktif (tanpa Surreal): lihat `web/sim/README.md` — `python scripts/export_sim_ui_dataset.py` lalu `cd web/sim && npm run dev`.

---

## Struktur repositori

```
├── assets/
├── src/
│   ├── config/              # settings, experiment_setups, runtime_setup (env RAG), dataset_generation_env
│   ├── dataset/             # generator.py
│   ├── ingestion/           # episode_ingester.py
│   ├── retrieval/           # agent.py, vanilla_retriever.py, hybrid_retriever.py, llm_reranker.py, trace.py
│   ├── surreal/             # connection.py, schema.surql, fact_graph.py, vanilla_store.py, ranking.py
│   ├── vectordb/            # Client wrapper untuk vanilla vector DB
│   ├── llm/                 # Provider LLM (gemini_provider.py, novita_provider.py)
│   ├── embedders/           # Provider Embedder (gemini_embedder.py, hf_embedder.py)
│   ├── evaluation/          # evaluator.py, metrics.py, query_schema.py
│   └── utils/               # Gemini helpers, cost tracker, rate limiter, dll.
├── scripts/
├── data/
├── output/
│   ├── example_dataset/
│   └── evaluation_results/
└── tests/
```

---

## Evaluasi (metrik skrip)

Skrip evaluasi mengukur antara lain:

- **Hit rate / MRR** terhadap sesi relevan yang diharapkan di file query.
- **Waktu retrieval** per query.
- **Context sufficiency** (opsional): penilaian dengan LLM judge — secara default memakai model Gemini (`--judge-model`), bukan GPT.

Rincian perhitungan ada di `scripts/evaluate_agentic.py` dan modul `src/evaluation/`.

---

## Referensi

- Inspirasi struktur longitudinal: [LOCOMO](https://github.com/snap-research/locomo).
- Fondasi graf temporal & pemeliharaan data: [Graphiti](https://github.com/getzep/graphiti).
- Penyimpanan temporal + vektor: **SurrealDB** (skema di repo).
- Model bahasa: **Google Gemini** (dataset + mayoritas jalur RAG); **Novita** untuk jalur ekstraksi OpenAI-compatible.

## Lisensi

MIT License
