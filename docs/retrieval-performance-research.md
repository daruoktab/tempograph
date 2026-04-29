# Penelitian performa retrieval — Tempograph (SurrealDB + RAG agentic)

**Tanggal:** 29 April 2026  
**Lingkup:** jalur vanilla, fact-graph (`TemporalGraphClient`), hybrid, agent iteratif, indeks SurrealDB, dan perbandingan ringkas dengan pola industri (termasuk inspirasi Graphiti).

---

## 1. Ringkasan eksekutif

Proyek ini menggabungkan **dense retrieval** (`session_passage`, satu vektor per sesi) dengan **retrieval berbasis fakta** (`extracted_fact` + relasi `has_fact` / `fact_involves`) di SurrealDB. Secara **fungsional** pipeline sudah utuh; secara **performa** (latensi, biaya API, recall@k, stabilitas skor) masih banyak ruang peningkatan.

Temuan utama:

| Dimensi | Status singkat |
|--------|----------------|
| **Recall** | Dua jalur (vektor + anchor entitas substring) membantu recall; tidak ada jalur sparse (BM25/fulltext) pada fakta atau passage. |
| **Presisi / ranking** | Merge skor di `fact_graph` bersifat heuristik (+0.02–0.04); bukan RRF atau reranker cross-encoder pada hasil fused. |
| **Temporal** | `valid_at` disimpan; filter temporal di `search_with_temporal_filter` dilakukan **pasca-ranking**; `RetrievalPlan.temporal_filter` tidak diisi di `create_plan`, sehingga jalur temporal di agent praktis tidak aktif. |
| **Latensi biaya** | Agent iteratif + cek sufficiency LLM + beberapa embed per iterasi → biaya dan latensi naik linear dengan iterasi. |
| **Indeks vektor** | MTREE cosine 768 pada `extracted_fact` dan `session_passage` — wajar untuk skala skripsi; opsi HNSW/waktu query perlu diverifikasi saat data membesar. |

Dokumen ini merangkum **bukti dari repo**, **referensi praktik eksternal**, dan **rencana pengembangan bertahap**.

---

## 2. Arsitektur retrieval di repo (deep dive)

### 2.1 Fact graph — `src/rag/surreal/fact_graph.py`

**Alur `search`:**

1. Embed query satu kali.
2. Ambil kandidat vektor: `fetch_lim = max(num_results * 3, num_results + 10)` lalu `ORDER BY cosine LIMIT fetch_lim`.
3. Opsional **entity graph**: `resolve_entities_in_query` — entitas yang `name` menjadi substring (case-insensitive) dari **seluruh string query** (bukan token NER), max 25.
4. `search_facts_for_entity_ids` — fakta yang terhubung ke entitas tersebut, diurut cosine terhadap query embedding pada subset itu.
5. `_merge_vector_and_graph_results` — dedupe by `fact_text`, boost tetap jika overlap vector+graph / nama.

**Implikasi performa:**

- **Substring match** pada query penuh bisa memicu false positive entitas panjang atau umum → noise graph recall.
- **Dua query Surreal** minimum (vector + mungkin entity branch); tambahan query untuk resolve entity dan graph facts.
- **Tidak ada** pembatasan `valid_at` / `created_at` di SQL — filter temporal tidak mempersempit kandidat di DB.

**`search_with_temporal_filter`:** memanggil `search(..., num_results * 2)` lalu memfilter `valid_at` di Python. Ini **tidak mengurangi kerja indeks** di tahap awal; cocok untuk prototipe, bukan optimal untuk skala besar.

### 2.2 Vanilla — `src/rag/surreal/vanilla_store.py` + `vanilla_retriever.py`

- Satu kali `ORDER BY cosine ... LIMIT n_results` pada `session_passage` (kandidat = `embedding_top_k` dari settings).
- Opsional threshold `similarity_threshold`, lalu rerank: **embedding-only** atau **LLM** (`LLMReranker`) tergantung `setup.reranker_type`.

**Implikasi:** dua tahap (broad K → narrow top-k) sudah selaras pola **two-stage retrieval** industri; bottleneck pindah ke **LLM rerank** (latency + token).

### 2.3 Agent iteratif — `src/rag/retrieval/agent.py`

- Konstanta keras: `MIN_FACTS=5`, `MAX_FACTS=15`, `MAX_ITERATIONS=5`.
- Per iterasi: satu atau lebih `execute_search` (embed per query string), lalu `expand_search` (hingga 5 entitas × `get_entity_facts`), lalu evaluasi sufficiency (LLM jika tersedia).
- `RetrievalConfig` di `settings.py` memuat `max_iterations: int = 3`, tetapi **`RetrievalAgent` memakai konstanta kelas `MAX_ITERATIONS = 5`** (`while state.iteration < self.MAX_ITERATIONS`) — field `config.max_iterations` **tidak dipakai** untuk batas iterasi saat ini. Ini mempengaruhi reproduksibilitas ablation bila hanya mengubah config.

### 2.4 Hybrid — `src/rag/retrieval/hybrid_retriever.py`

- Memanggil `graph.search(query, num_results=15)` — **bukan** loop agent penuh kecuali `graph_client` yang di-inject memang wrapper agent.
- Vanilla: `retrieve` penuh, lalu **dipotong 5** (`VANILLA_SUPPLEMENT`) — gabungan **concatenation**, bukan fusion skor (RRF / weighted).

**Implikasi:** konteks bisa panjang (hingga ~15 fakta + 5 passage); dampak ke **window LLM generator** dan biaya inference perlu dimonitor.

### 2.5 Skema indeks — `src/rag/surreal/schema.surql`

- `idx_fact_embed` / `idx_session_passage_embed`: **MTREE** dimensi **768**, cosine.
- **Risiko:** dimensi embedding **harus** konsisten dengan model; mismatch akan gagal atau kualitas buruk.
- **Tuning:** untuk dataset besar, bandingkan dokumentasi SurrealDB terkini untuk MTREE vs HNSW (latensi build vs query).

---

## 3. Literatur dan praktik industri (referensi)

Ringkasan pola yang sering dipakai untuk meningkatkan **kualitas dan efisiensi** retrieval (bukan mengikat ke satu vendor):

1. **Two-stage retrieval** — tahap 1 recall lebar (K besar), tahap 2 presisi (rerank / prune ke k kecil). Diskusi arsitektur: [Advanced RAG with Reranking (LLMversus)](https://llmversus.com/architecture/advanced-rag-reranking).
2. **Hybrid lexical + dense** — BM25/sparse melengkapi vektor untuk token eksak, nama propri, kode. Diskusi hybrid + rerank: [Adelean — hybrid reranking](https://www.adelean.com/en/blog/20250417_hybrid_reranking/).
3. **RRF (Reciprocal Rank Fusion)** — menggabungkan peringkat dari beberapa sumber tanpa normalisasi skor silang; umum di stack hybrid. Dokumentasi contoh: [Milvus — RRF Ranker](https://milvus.io/docs/rrf-ranker.md).
4. **Query expansion / HyDE** — menaikkan recall pada parafrase; trade-off biaya embed/tambahan LLM. Gambaran strategi: [Advanced Retrieval Strategies for RAG (Ailog)](https://app.ailog.fr/en/blog/guides/retrieval-strategies).

**Catatan:** angka peningkatan persentase di banyak artikel marketing **berkonteks dataset**; gunakan sebagai hipotesis, bukan jaminan, dan ukur di dataset Indonesia Anda.

---

## 4. Kesenjangan (gap) terhadap best practice

| Area | Di repo sekarang | Best practice umum |
|------|-------------------|---------------------|
| Fusion graph + vector | Heuristik skor tetap | RRF atau learned fusion pada rank |
| Sparse signal | Tidak pada `extracted_fact` / passage | BM25 atau full-text index + hybrid |
| Temporal | Pasca-filter / tidak terpakai di plan | Filter di query atau di indeks sekunder waktu |
| Entity anchor | Substring pada query | NER / gazetteer / fuzzy match terkalibrasi |
| Metrik operasional | Metrik evaluasi jawaban ada di `evaluation/` | Tambah log: p95 latensi per tahap, jumlah dokumen sebelum/sesudah threshold |
| Konfigurasi agent | Konstanta di kelas | Satu sumber kebenaran (`RetrievalConfig`) + eksperimen YAML/env |

---

## 5. Planning pengembangan (semua tahap)

Rencana dibagi **fase** agar bisa diprioritaskan untuk skripsi (bukti empiris vs fitur besar).

### Fase A — Instrumentasi dan baseline (1–2 minggu)

**Tujuan:** bisa menjawab “lembut di mana” dengan angka.

1. **Logging terstruktur** per query evaluasi:
   - waktu: embed query, query Surreal (vector / entity / graph), merge, rerank LLM (jika ada), sufficiency LLM.
   - hitungan: `fetch_lim`, jumlah baris sebelum/sesudah threshold, jumlah iterasi agent.
2. **Satukan konfigurasi** `MAX_ITERATIONS` / `MIN_FACTS` / `MAX_FACTS` dengan `RetrievalConfig` atau `ExperimentSetup` agar ablation reproducible.
3. **Checkpoint metrik** yang sudah ada (`context_recall`, `mrr`, dll.) dipasangkan dengan **log latensi** per setup (vanilla vs agentic vs hybrid).

**Deliverable:** satu skrip atau flag evaluasi yang menulis JSONL `retrieval_trace.jsonl`.

### Fase B — Quick wins performa & presisi (2–4 minggu)

1. **Pindahkan filter temporal ke SurrealQL** (optional `WHERE valid_at ...`) pada cabang vector dan/atau entity — mengurangi transfer baris dan kerja merge Python.
2. **Isi `temporal_filter` di `create_plan`** untuk pertanyaan ber-tanggal (regex / classifier ringan) agar fitur temporal benar-benar teruji di evaluasi.
3. **Kalibrasi `fetch_lim`** vs `num_results` (misalnya `2*k` vs `3*k+10`) dengan sweep kecil pada subset query.
4. **Parallel `graph.search` dan `vanilla.retrieve` di hybrid** jika keduanya independen — potong latensi wall-clock.

**Deliverable:** tabel ablation (sebelum/sesudah) pada subset `evaluation_queries`.

### Fase C — Hybrid search & fusion (4–8 minggu)

1. **Full-text / BM25** pada `fact_text` dan/atau `session_passage.text` (jika SurrealDB versi Anda mendukung pola indeks yang dipilih) — jalur recall kedua.
2. **RRF** menggabungkan rank vector dan rank sparse (dan opsional rank “entity graph”) menggantikan atau melengkapi `_merge_vector_and_graph_results`.
3. **Cross-encoder ringan** (lokal, batch) sebagai stage-2 opsional dibanding LLM rerank penuh — trade-off kualitas vs biaya.

**Deliverable:** satu mode retrieval `hybrid_bm25_rrf` dibanding baseline di skrip evaluasi.

### Fase D — Entity & graph recall (bersifat penelitian)

1. **NER** (Indonesian) atau daftar entitas dari dataset untuk mengganti/heuristik kapitalisasi di `create_plan`.
2. **Multi-hop terbatas** (1–2 lompatan) dari fakta ke entitas tetangga, dengan batasan `limit` — mirip ide BFS terbatas di graph memory frameworks.
3. **Dedup semantik** (cluster embedding fakta) untuk mengurangi redundansi konteks ke generator.

**Deliverable:** analisis error: “miss” karena entity vs karena vektor vs karena temporal.

### Fase E — Skala & infrastruktur (opsional pasca-skripsi)

1. Evaluasi **HNSW vs MTREE** pada volume data target.
2. **Caching embed query** dalam satu sesi chat multi-turn.
3. **Batch embed** fakta saat ingest jika API mendukung — menurunkan waktu ingest (bukan retrieval online tapi mempercepat eksperimen iteratif).

---

## 6. Matriks prioritas (untuk dokumentasi skripsi)

| Prioritas | Item | Dampak ke bukti skripsi |
|-----------|------|-------------------------|
| P0 | Instrumentasi + temporal di SQL + parallel hybrid | Kuat: metodologi jelas, hasil terukur |
| P1 | RRF / sparse hybrid | Sedang–tinggi: perbandingan metode modern |
| P2 | NER + multi-hop | Tinggi risiko waktu; bagus sebagai “future work” jika waktu tipis |

---

## 7. Referensi file kunci di repo

- `src/rag/surreal/fact_graph.py` — search, merge, temporal filter, ingest fakta  
- `src/rag/surreal/vanilla_store.py` — vector search passage  
- `src/rag/retrieval/vanilla_retriever.py` — two-stage + threshold + rerank  
- `src/rag/retrieval/agent.py` — loop iteratif + expand entitas  
- `src/rag/retrieval/hybrid_retriever.py` — gabungan graph + vanilla  
- `src/config/settings.py` — `RetrievalSettings` / `RetrievalConfig`  
- `src/config/experiment_setups.py` — `embedding_top_k`, `rerank_top_k`, threshold per setup  
- `src/rag/surreal/schema.surql` — indeks vektor  
- `src/evaluation/metrics.py` — metrik evaluasi jawaban  

---

## 8. Kesimpulan

Repo sudah mengimplementasikan **inti** sistem RAG dua-lapis (dense passage + fakta tergraf) dengan **merge vektor–entitas** yang masuk akal. Untuk **performa retrieval** dalam arti industri (latensi, biaya, recall hybrid, temporal konsisten), langkah paling bernilai berikutnya adalah **pengukuran**, **filter temporal di database**, **fusion yang lebih standar (RRF)**, dan **sinyal sparse** — direncanakan di atas agar bisa dijalankan bertahap tanpa me rewrite seluruh sistem sekaligus.
