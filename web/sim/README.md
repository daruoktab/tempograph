# Tempograph — UI simulasi dataset

Antarmuka **Vite + React** untuk menjelajahi **10 sesi pertama** dari `conversation_dataset` (file JSON hasil ekspor; setara isi `.toon` di repo). Termasuk:

- **Transkrip interaktif** — slider giliran + tombol langkah.
- **Graf sesi** — user, persona sekunder, entitas dari `ground_truths` hingga giliran saat ini; edge co-mention dalam satu giliran.
- **Graf life events** — `user_events.json` (subset 40 event) dengan edge `caused_by`.
- **Mock retrieval** — skor Jaccard token terhadap transcript yang sudah terbuka (proxy singkat, tanpa SurrealDB).

## Prasyarat

- Node.js 20+ (disarankan)
- Dataset ringan di `public/dataset/` (jalankan ekspor dari root repo)

## Menyiapkan data

Dari **root repositori** (bukan dari folder `web/sim`):

```bash
python scripts/export_sim_ui_dataset.py
```

Ini menulis `web/sim/public/dataset/conversation_sim_10.json` dan menyalin `user_events.json`.

## Menjalankan

```bash
cd web/sim
npm install
npm run dev
```

Buka URL yang ditampilkan Vite (default port **5174**).

## Build statis

```bash
npm run build
```

Output di `web/sim/dist/` — bisa di-serve dengan nginx / GitHub Pages (atur `base` di `vite.config.ts` jika perlu subpath).

## Catatan

- Parser **TOON** tidak di-bundle di browser; sumber kebenaran ekspor adalah **`conversation_dataset.json`**. Setelah regenerate dataset, jalankan ulang skrip ekspor.
- Ini **bukan** koneksi ke SurrealDB; untuk eval penuh gunakan `scripts/evaluate_agentic.py`.
