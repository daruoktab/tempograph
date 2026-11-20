# Tempograph — halaman showcase

Halaman portfolio interaktif (satu file, tanpa build, tanpa dependensi selain Google Fonts).

| Path | Isi |
|------|-----|
| `index.html` | Versi terbaru — yang di-deploy ke `/` |
| `archive/showcase-v1-original.html` | Versi awal (dulu `Tempograph_Portfolio_Showcase.html` di root) |
| `archive/showcase-v2.html` | Versi 2 (dulu `Tempograph_Portfolio_Showcase_v2.html`) |

Demo 10 sesi ada di [`../sim/`](../sim/) (Vite + React) dan tampil di `/sim/` pada situs.

## Isi `index.html`

Seluruh angka dan graf di halaman berasal dari data repo, bukan data karangan:

- graf 80 event kausal (`output/example_dataset/user_events.json`), bisa di-drag, di-zoom, dan di-play per tanggal
- graf co-mention entitas (29 node) dengan dedup rapidfuzz dan Louvain yang berjalan langsung di browser
- jendela validitas bi-temporal (`valid_at` / `invalid_at`) dengan kursor "as of"
- simulasi contradiction gate (ambang cosine 0.35), mode retrieval Agentic / Vanilla / Hybrid
- filter 100 eval query dan generator `.env` + perintah CLI

## Deploy

```bash
bash scripts/deploy_pages.sh
```

Skrip membuild `web/sim` bila perlu lalu mendorong `index.html`, `sim/`, dan logo ke branch `gh-pages`.
Situs: https://daruoktab.github.io/tempograph/
