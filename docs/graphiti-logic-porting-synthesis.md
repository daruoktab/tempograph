# Sintesis logika Graphiti → ide peningkatan Tempograph (Gemini + SurrealDB)

**Sumber penelusuran:** paket `graphiti_core` terpasang di conda env `porto-skripsi`  
**Path lokal (referensi):** `C:\Users\daru\anaconda3\envs\porto-skripsi\Lib\site-packages\graphiti_core\`

**Batasan (sesuai permintaan):** tidak mengusulkan ganti **model** (tetap Gemini) atau **basis data** (tetap SurrealDB). Fokus pada **algoritme, orkestrasi, format konteks, ingest**, yang bisa direimplementasikan di atas SurrealQL / Python.

---

## 1. Peta modul Graphiti yang relevan

| Area | File / modul utama | Apa yang dilakukan |
|------|-------------------|-------------------|
| Orkestrasi pencarian | `search/search.py` | Jalankan beberapa metode (BM25, cosine, BFS) per scope, **parallel** (`semaphore_gather`), gabungkan dengan reranker. |
| Resep konfigurasi | `search/search_config_recipes.py` | Pola siap pakai: hybrid BM25+cosine, RRF/MMR/cross-encoder, BFS opsional. |
| Primitif ranking | `search/search_utils.py` | `rrf`, `maximal_marginal_relevance`, `episode_mentions_reranker`, `node_distance_reranker`, pencarian fulltext/similarity/BFS. |
| Filter temporal | `search/search_filters.py` | `valid_at`, `invalid_at`, `created_at`, `expired_at` sebagai filter query terstruktur. |
| Format konteks LLM | `search/search_helpers.py` | `search_results_to_context_string` — JSON terstruktur untuk facts + rentang tanggal. |
| Dedup entitas | `utils/maintenance/dedup_helpers.py` | Normalisasi string, **entropy** nama, **MinHash/LSH** untuk kandidat fuzzy sebelum LLM. |
| Chunking ingest | `utils/content_chunking.py` + `helpers.py` | Chunk **berdasarkan ukuran + kepadatan entitas** (bukan semua teks panjang). |
| Jendela episodik | `utils/maintenance/graph_data_operations.py` | `retrieve_episodes` — `valid_at <= reference_time`, `ORDER BY valid_at DESC`, `LIMIT last_n` (`EPISODE_WINDOW_LEN = 3`). |
| Ingest bulk | `utils/bulk_utils.py` | Ekstraksi/dedup/resolusi batch ( pola pipeline ). |

---

## 2. Logika inti yang bisa diambil (tanpa Neo4j)

### 2.1 Multi-sinyal recall + oversampling

**Di Graphiti:** untuk tiap metode (mis. BM25, cosine), limit hasil sering **dikalikan ~2×** (`2 * limit`) sebelum rerank, supaya pool kandidat lebar.

**Di repo Anda:** `fetch_lim = max(num_results * 3, num_results + 10)` di `TemporalGraphClient.search` sudah mirip spirit oversampling; untuk **jalur kedua** (sparse / keyword) belum ada.

**Aksi di SurrealDB:** jalankan dua query (vektor + full-text atau `string::contains` / indeks teks jika tersedia), masing-masing `2*k`, lalu gabung di Python atau lewat RRF.

---

### 2.2 Reciprocal Rank Fusion (RRF)

**Di Graphiti:** `search_utils.rrf(results: list[list[str]], rank_const=1, min_score=0)` — menggabungkan beberapa **peringkat UUID** tanpa menyamakan skor BM25 vs cosine.

**Di repo Anda:** `_merge_vector_and_graph_results` memakai **skor absolut + boost tetap** (+0.02–0.04).

**Aksi:** pertahankan ID stabil per `extracted_fact` (RecordID / string id); bangun list rank dari (1) similarity global, (2) similarity pada subgraph entitas, (3) opsional rank keyword; terapkan RRF ke id fakta; baru map kembali ke teks. Ini **murni Python**, kompatibel Gemini.

---

### 2.3 Maximal Marginal Relevance (MMR)

**Di Graphiti:** `maximal_marginal_relevance(query_vector, candidates: dict[id, embedding], mmr_lambda, min_score)` — trade-off relevansi ke query vs **redundansi antar-kandidat** (matriks similarity antar kandidat).

**Di repo Anda:** dedupe berdasarkan string `fact` persis; tidak ada diversifikasi embedding antar fakta yang parafrase.

**Aksi:** setelah pool kandidat (mis. 30 fakta), pilih subset output dengan MMR menggunakan **embedding fakta yang sudah disimpan** di Surreal — tidak perlu model baru, hanya numpy + vektor dari DB.

---

### 2.4 Parallel scope search

**Di Graphiti:** `search()` memanggil `edge_search`, `node_search`, `episode_search`, `community_search` secara **parallel** (`semaphore_gather`).

**Di repo Anda:** urutan: embed → vector query → (optional) entity resolve → graph query; hybrid sequential graph lalu vanilla (ada catatan di `hybrid_retriever` soal jangan double-init).

**Aksi:** `asyncio.gather` untuk cabang yang independen: mis. `session_passage` search vs `extracted_fact` vector search; atau vector vs full-text pada fakta.

---

### 2.5 Filter temporal terstruktur (bukan hanya pasca-filter)

**Di Graphiti:** `SearchFilters` + `edge_search_filter_query_constructor` — `valid_at` / `invalid_at` / `created_at` / `expired_at` menjadi **predikat query**.

**Di repo Anda:** `valid_at` ada di skema; `search_with_temporal_filter` memfilter **setelah** `search`.

**Aksi:** tambahkan parameter `before`/`after` ke `SELECT ... FROM extracted_fact WHERE ... AND (valid_at ...)` (SurrealQL). Pola **invalid_at** bisa disimulasikan dengan field opsional nanti jika Anda butuh “fakta kedaluwarsa”.

---

### 2.6 Episode window & “state at time T”

**Di Graphiti:** `retrieve_episodes(reference_time, last_n=EPISODE_WINDOW_LEN)` — episodenya yang `valid_at <= reference_time`, diurut terbaru, dibatasi `n`.

**Di repo Anda:** `episode.reference_time` / `created_at` ada; belum ada API retrieval “episode terakhir sebelum T” untuk menyuplai konteks narasi mentah bersama fakta.

**Aksi:** satu fungsi Surreal: `SELECT FROM episode WHERE group_id = $g AND reference_time <= $t ORDER BY reference_time DESC LIMIT $n` — berguna untuk prompt generator atau untuk **reranking** “episode yang menyebut entitas X” (mirip ide `episode_mentions_reranker`).

---

### 2.7 Episode / mention–aware reranking (konsep)

**Di Graphiti:** `episode_mentions_reranker` — gabungan RRF antar list node, lalu **hitungan** episode yang `MENTIONS` entitas (semakin sering disebut di episod, semakin tinggi prioritas).

**Di repo Anda:** `has_fact` menghubung episode → fakta; bisa dihitung **frekuensi episode** per fakta atau per entitas tanpa mengganti DB.

**Aksi:** skor tambahan: `boost = log(1 + count_episodes_sharing_entity)` setelah traverse, atau terapkan sebagai fitur kedua dalam RRF (satu rank list dari “mention count”, satu dari vector).

---

### 2.8 Node distance reranking (konsep)

**Di Graphiti:** dari **center node**, entitas yang terhubung langsung di-graph dapat skor jarak lebih baik.

**Di repo Anda:** graf hanya `fact_involves` (fakta → entitas), belum ada relasi entitas–entitas.

**Aksi ringan:** “jarak” = **jumlah hop** lewat fakta bersama (2 entitas yang muncul di fakta yang sama → jarak 2). Itu bisa dihitung dengan query terbatas atau materialized di ingest jika diperlukan — tanpa property graph penuh seperti `RELATES_TO`.

---

### 2.9 Format konteks untuk LLM generator

**Di Graphiti:** `search_results_to_context_string` memisahkan `<FACTS>`, `<ENTITIES>`, `<EPISODES>`, `<COMMUNITIES>` dengan **valid_at / invalid_at** eksplisit di JSON.

**Di repo Anda:** `HybridRetriever.format_context` sudah memisahkan FACT vs DETAIL; fakta belum selalu menyertakan rentang validitas secara konsisten di string.

**Aksi:** seragamkan satu builder (mirip template Graphiti) dengan field `valid_at`, `source_description`, `entity_names` — membantu Gemini menjawab pertanyaan temporal tanpa mengganti model.

---

### 2.10 Dedup entitas cerdas (bukan hanya substring query)

**Di Graphiti:** `dedup_helpers` — entropy nama, panjang minimum, MinHash/Jaccard untuk **kandidat** sebelum merge; nama ber-entropy rendah tidak dipercaya untuk fuzzy match otomatis.

**Di repo Anda:** `resolve_entities_in_query` = substring `string::contains` pada query lowercased.

**Aksi:** (1) tokenisasi + kandidat dari gazetteer entitas yang pernah muncul di `entity` table untuk `group_id`; (2) skor Jaccard/MinHash antara token query dan nama entitas; (3) abaikan match entropy rendah — **logika bisa di-copy konsepnya** tanpa menyalin seluruh modul (perhatikan lisensi Apache Graphiti jika copy-paste kode).

---

### 2.11 Chunking berkepadatan entitas saat ekstraksi

**Di Graphiti:** `should_chunk` — chunk hanya jika teks **cukup panjang** dan **kepadatan entitas** tinggi (JSON rapat, dll.); teks narasi panjang tidak dipecah sembarangan.

**Di repo Anda:** `fact_graph._extract_facts` memotong `body[:120_000]` satu blok.

**Aksi:** bagi episode sangat panjang menjadi beberapa segmen dengan overlap (`CHUNK_TOKEN_SIZE`, `CHUNK_OVERLAP_TOKENS` di `helpers.py` sebagai referensi angka), **hanya** bila `estimate_tokens` + heuristik kapital/koma melebih ambang — lalu merge fakta dengan dedupe teks. Mengurangi kegagalan JSON LLM tanpa mengganti Gemini.

---

### 2.12 Tracing span untuk optimasi

**Di Graphiti:** `_trace_phase` / `Tracer` di sekitar embed, execute_scopes.

**Di repo Anda:** logging biasa.

**Aksi:** context manager sederhana yang mencatat durasi tiap fase (embed, surreal vector, surreal entity, merge) — sama sekali tidak bergantung pada DB vendor.

---

## 3. Yang sengaja *tidak* diprioritaskan (biaya vs manfaat di Surreal)

| Fitur Graphiti | Alasan deprioritisasi |
|----------------|---------------------|
| **Community** nodes & cluster | Butuh pipeline komunitas + skema tambahan besar. |
| **Cross-encoder** khusus vendor | Bisa diganti **Gemini Flash** ringan untuk rerank pasangan (query, fakta) — itu tetap Gemini, bukan model baru “asing”; opsional. |
| **Cypher / provider** spesifik | Sudah diganti SurrealQL — hanya **polanya** yang diambil. |

---

## 4. Urutan implementasi yang masuk akal (roadmap singkat)

1. **RRF** pada dua list rank (vector global vs vector-on-entity-subgraph) — dampak langsung pada kualitas ranking.  
2. **Filter `valid_at` di SurrealQL** — dampak latensi + konsistensi temporal.  
3. **`asyncio.gather`** untuk hybrid vanilla + graph search.  
4. **Format konteks** ala `search_helpers` untuk evaluasi generator.  
5. **MMR** pada kandidat fakta sebelum kirim ke LLM sufficiency / generator.  
6. **Chunking berkepadatan** pada `_extract_facts` untuk sesi sangat panjang.  
7. **Dedup entitas** bertahap (entropy + Jaccard) menggantikan atau melengkapi substring.  
8. **Episode window** + mention-boost — jika metrik eval menunjukkan recall temporal lemah.

---

## 5. Referensi cepat ke file Graphiti (untuk dibuka manual)

```
graphiti_core/search/search.py              # orkestrasi + parallel
graphiti_core/search/search_utils.py        # rrf, mmr, BFS, fulltext, episode_mentions
graphiti_core/search/search_config_recipes.py
graphiti_core/search/search_filters.py
graphiti_core/search/search_helpers.py
graphiti_core/utils/maintenance/dedup_helpers.py
graphiti_core/utils/content_chunking.py
graphiti_core/utils/maintenance/graph_data_operations.py
graphiti_core/helpers.py                    # CHUNK_* constants, lucene_sanitize pattern
```

---

## 6. Catatan lisensi

`graphiti_core` berlisensi **Apache 2.0** (header file). Jika mengadaptasi **cuplikan kode** (mis. `rrf`, rumus MMR), pertahankan notice lisensi dan atribusi sesuai ketentuan Apache 2.0. Untuk **ide algoritme** (RRF, MMR, oversampling 2k), penjelasan di dokumen ini cukup untuk implementasi ulang dari nol.

---

*Dokumen ini melengkapi `docs/retrieval-performance-research.md` dengan sudut pandang khusus “apa yang bisa diimpor dari Graphiti” tanpa mengganti Gemini atau SurrealDB.*
