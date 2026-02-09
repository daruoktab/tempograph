#!/usr/bin/env python
# scripts/ingest_agentic.py
"""
Agentic RAG Ingestion Script
============================

Ingest conversation sessions ke Neo4j untuk Agentic RAG.
Menggunakan Graphiti untuk fact extraction dan graph building.

PENTING: Ingestion dilakukan PER-SESSION, bukan per-turn!
- Sesuai real-world scenario: dalam 1 sesi, semua turn masih di context window
- RAG hanya dibutuhkan untuk retrieve info dari sesi LAIN
- LLM extraction lebih akurat karena bisa lihat full session context
- Lebih efisien: 100 LLM calls vs 1143 calls

Features:
- Sequential processing untuk fairness
- Rate limit handling dengan auto-retry untuk Gemini API
- Checkpoint/resume support
- Progress tracking
- Separate group_id untuk Gemini vs Gemma

Usage:
    python scripts/ingest_agentic.py --setup gemini
    python scripts/ingest_agentic.py --setup gemma
    python scripts/ingest_agentic.py --setup all
    python scripts/ingest_agentic.py --setup gemini --resume  # Resume dari checkpoint
"""

import asyncio
import argparse
import json
import logging
import sys
import time
import os
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# ============================================================
# SUPPRESS NOISY WARNINGS (before any imports that trigger them)
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"  # Suppress gRPC
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizers warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress TensorFlow oneDNN info
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Suppress MPS warnings
os.environ["TORCHAO_DISABLE_WARNINGS"] = "1"  # Suppress torchao warnings
os.environ["TRITON_CACHE_DIR"] = ""  # Suppress triton warnings
os.environ["TORCH_LOGS"] = "-all"  # Suppress all torch logs

# Suppress all warnings before imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*triton.*")
warnings.filterwarnings("ignore", message=".*torchao.*")
warnings.filterwarnings("ignore", message=".*Redirects.*")
warnings.filterwarnings("ignore", message=".*cpp extensions.*")
warnings.filterwarnings("ignore", message=".*tf_keras.*")
warnings.filterwarnings("ignore", message=".*sparse_softmax.*")
warnings.filterwarnings("ignore", message=".*Skipping import.*")
warnings.filterwarnings("ignore", message=".*Detected no triton.*")

# Suppress Neo4j driver notifications (very noisy schema index logs)
logging.getLogger("neo4j").setLevel(logging.CRITICAL)
logging.getLogger("neo4j.io").setLevel(logging.CRITICAL)
logging.getLogger("neo4j.pool").setLevel(logging.CRITICAL)
logging.getLogger("neo4j.notifications").setLevel(logging.CRITICAL)
logging.getLogger("graphiti_core").setLevel(logging.WARNING)
logging.getLogger("graphiti_core.driver").setLevel(logging.CRITICAL)

# Suppress ML framework loggers
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# Custom filter to suppress harmless Neo4j warnings
class Neo4jNoiseFilter(logging.Filter):
    def filter(self, record):
        msg = str(record.getMessage()).lower()
        # Suppress harmless warnings
        suppressed_patterns = [
            "equivalentschemarulealreadyexists",
            "equivalent index already exists",
            "equivalent schema rule already exists",
            "transaction failed and will be retried",
            "failed to read from defunct connection",
            "semaphore timeout period has expired",
            "connection error",
            "<connection> error",
            "error executing neo4j query",
            "neo4j_code:",
            "gql_status:",
            "create index",
            "create fulltext index",
        ]
        for pattern in suppressed_patterns:
            if pattern in msg:
                return False
        return True


# Apply filter to all relevant loggers
for logger_name in [
    "",
    "neo4j",
    "neo4j.io",
    "neo4j.pool",
    "graphiti_core",
    "graphiti_core.driver",
]:
    logging.getLogger(logger_name).addFilter(Neo4jNoiseFilter())

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATASET_PATH = Path("output/final_dataset_v1/conversation_dataset.json")
CHECKPOINT_DIR = Path("data/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class IngestionCheckpoint:
    """Checkpoint for resuming ingestion"""

    setup_name: str
    total_sessions: int
    processed_sessions: int
    last_session_id: int
    started_at: str
    updated_at: str
    status: str  # "in_progress", "completed", "failed"
    error_message: Optional[str] = None
    entities_created: int = 0
    facts_created: int = 0
    episodes_created: int = 0

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "IngestionCheckpoint":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# TOKEN BUDGET MANAGER - Fine-grained per-LLM-call budgeting
# =============================================================================


class TokenBudgetManager:
    """
    Proactive token budget management with sliding window.

    Key insight: Instead of waiting 65s for EVERY LLM call,
    we track actual token usage and only wait when approaching limit.

    OpenRouter Gemma limit: 15,000 tokens/minute
    We use 14,000 as target (1000 token buffer for safety)
    Estimated tokens per LLM call: ~5,000
    """

    TOKENS_PER_MINUTE = 14000  # Target limit (1000 buffer from 15k)
    WINDOW_SECONDS = 60

    # Estimated tokens per call type
    TOKENS_PER_CALL = 4000  # Conservative estimate (~3 calls per minute)

    def __init__(self):
        self.window_start = time.time()
        self.tokens_used = 0
        self.call_count = 0
        self.wait_count = 0
        self._lock = asyncio.Lock()

    def _reset_if_needed(self):
        """Reset window if expired"""
        now = time.time()
        if now - self.window_start >= self.WINDOW_SECONDS:
            self.window_start = now
            self.tokens_used = 0
            self.call_count = 0

    async def acquire(self, estimated_tokens: int | None = None):
        """
        Acquire tokens before making an LLM call.
        Waits if necessary to stay within budget.

        IMPORTANT: This now acquires a GLOBAL SEMAPHORE that serializes
        all LLM calls to prevent race conditions.
        """
        # First, acquire the global semaphore to serialize API calls
        # This prevents multiple concurrent calls from bypassing the budget
        await _llm_semaphore.acquire()

        try:
            async with self._lock:
                if estimated_tokens is None:
                    estimated_tokens = self.TOKENS_PER_CALL

                self._reset_if_needed()

                # Check if we would exceed budget
                if self.tokens_used + estimated_tokens > self.TOKENS_PER_MINUTE:
                    # Calculate wait time until window resets
                    elapsed = time.time() - self.window_start
                    wait_time = max(0, self.WINDOW_SECONDS - elapsed + 2)  # +2s buffer

                    if wait_time > 0:
                        self.wait_count += 1
                        logger.info(
                            f"⏸️  Token budget: {self.tokens_used}/{self.TOKENS_PER_MINUTE} used, waiting {wait_time:.0f}s for window reset"
                        )
                        await asyncio.sleep(wait_time)
                        # Reset after waiting
                        self.window_start = time.time()
                        self.tokens_used = 0
                        self.call_count = 0

                # Reserve tokens
                self.tokens_used += estimated_tokens
                self.call_count += 1
        except Exception:
            # Release semaphore if anything fails during acquire
            _llm_semaphore.release()
            raise

    def release_semaphore(self):
        """Release the global semaphore after API call completes"""
        try:
            _llm_semaphore.release()
        except ValueError:
            pass  # Already released

    def release(self, actual_tokens: int, estimated_tokens: int):
        """Adjust for actual vs estimated (optional refinement)"""
        diff = estimated_tokens - actual_tokens
        self.tokens_used = max(0, self.tokens_used - diff)

    def get_stats(self) -> dict:
        return {
            "tokens_used": self.tokens_used,
            "call_count": self.call_count,
            "wait_count": self.wait_count,
            "budget_remaining": self.TOKENS_PER_MINUTE - self.tokens_used,
        }


# Global semaphore to serialize LLM API calls (prevents race condition)
# Only 1 LLM call can be in-flight at a time to respect rate limits
_llm_semaphore = asyncio.Semaphore(1)

# Global token budget manager (shared across all LLM calls)
_token_budget = None


def get_token_budget() -> TokenBudgetManager:
    global _token_budget
    if _token_budget is None:
        _token_budget = TokenBudgetManager()
    return _token_budget


class RateLimitHandler:
    """
    Smart rate limit handling with PROACTIVE throttling.

    OpenRouter Gemma limit: 15,000 tokens/minute

    Strategy:
    1. Track estimated token usage per window (60s)
    2. Proactively slow down when approaching limit
    3. Only do long waits when actually hit limit
    4. Reset counter every 60s window
    """

    # OpenRouter Gemma limit
    TOKENS_PER_MIN = 15000
    WINDOW_SECONDS = 60

    # Estimated tokens per Graphiti operation
    # (input + output tokens)
    TOKENS_PER_EXTRACT_NODES = 2000
    TOKENS_PER_EXTRACT_EDGES = 2000
    TOKENS_PER_DEDUPE = 1500
    TOKENS_PER_SUMMARIZE = 500
    TOKENS_PER_OTHER = 800

    # Total per session: ~8-10 LLM calls
    ESTIMATED_TOKENS_PER_SESSION = 12000

    def __init__(
        self,
        base_delay: float = 2.0,
        max_delay: float = 120.0,  # Reduced from 300s
        tokens_per_min: int = TOKENS_PER_MIN,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.tokens_per_min = tokens_per_min

        # Token tracking
        self.tokens_used_in_window = 0
        self.window_start = time.time()

        # Stats
        self.consecutive_errors = 0
        self.total_rate_limit_waits = 0
        self.total_proactive_waits = 0

    def _reset_window_if_needed(self):
        """Reset token counter if window expired"""
        now = time.time()
        if now - self.window_start >= self.WINDOW_SECONDS:
            self.tokens_used_in_window = 0
            self.window_start = now

    def estimate_remaining_tokens(self) -> int:
        """How many tokens left in current window"""
        self._reset_window_if_needed()
        return max(0, self.tokens_per_min - self.tokens_used_in_window)

    def record_token_usage(self, tokens: int):
        """Record token usage"""
        self._reset_window_if_needed()
        self.tokens_used_in_window += tokens

    async def proactive_throttle(self):
        """
        Proactively wait if we're close to the limit.
        Called BEFORE each session to pace ourselves.
        """
        self._reset_window_if_needed()

        _remaining = self.estimate_remaining_tokens()
        usage_pct = (self.tokens_used_in_window / self.tokens_per_min) * 100

        # If we've used more than 80% of our budget, wait for window reset
        if usage_pct > 80:
            time_in_window = time.time() - self.window_start
            time_to_wait = max(0, self.WINDOW_SECONDS - time_in_window + 2)

            if time_to_wait > 0:
                self.total_proactive_waits += 1
                logger.info(
                    f"⏸️  Proactive throttle: {usage_pct:.0f}% used, waiting {time_to_wait:.0f}s for window reset"
                )
                await asyncio.sleep(time_to_wait)
                self._reset_window_if_needed()

        # Otherwise, just record that we're about to use tokens
        self.record_token_usage(self.ESTIMATED_TOKENS_PER_SESSION)

    def reset(self):
        """Reset after successful request"""
        self.consecutive_errors = 0

    async def wait_on_rate_limit(self, error_msg: str = ""):
        """Wait when rate limit is hit (reactive)"""
        self.consecutive_errors += 1
        self.total_rate_limit_waits += 1

        # When we hit rate limit, wait for window to reset
        time_in_window = time.time() - self.window_start
        wait_time = max(self.WINDOW_SECONDS - time_in_window + 5, 30)  # At least 30s
        wait_time = min(wait_time, self.max_delay)

        logger.warning(
            f"\n⏳ Rate limit hit! Waiting {wait_time:.0f}s for window reset..."
        )
        logger.warning(
            f"   Tokens used: {self.tokens_used_in_window}, Error: {error_msg[:100]}..."
        )

        await asyncio.sleep(wait_time)

        # Reset window after waiting
        self.tokens_used_in_window = 0
        self.window_start = time.time()


# =============================================================================
# COST TRACKING (ESTIMATED)
# =============================================================================

# Gemini 1.5 Flash Pricing (Approximate)
# Cost Tracker (Centralized)
from src.utils.cost_tracker import get_cost_tracker  # noqa: E402

# Enable file logging for cost tracking
_cost_tracker = get_cost_tracker()
_cost_tracker.set_log_file("output/ingestion_cost.jsonl")


class AgenticIngester:
    """
    Ingest sessions ke Neo4j untuk Agentic RAG.
    Menggunakan Graphiti untuk fact extraction.

    UNIT: Per-Session (bukan per-turn)
    - Setiap session = 1 episode di Graphiti
    - Graphiti akan extract facts dari seluruh session
    """

    def __init__(self, setup_name: str):
        """
        Args:
            setup_name: "gemini" or "gemma"
        """
        self.setup_name = setup_name
        self.rate_limiter = RateLimitHandler()
        self._graphiti = None
        self._setup = None
        self._group_id = None

    async def initialize(self):
        """Initialize Graphiti client"""
        from src.config.experiment_setups import (
            SETUP_1A_AGENTIC_GEMINI,
            SETUP_2A_AGENTIC_GEMMA,  # Gemma via Novita AI
        )
        from src.config.settings import get_config

        config = get_config()

        # Set setup name in cost tracker for proper logging
        _cost_tracker.setup_name = self.setup_name

        # Get setup
        if self.setup_name == "gemini":
            # Use high-detail fact extraction setup
            self._setup = SETUP_1A_AGENTIC_GEMINI
        elif self.setup_name == "gemma":
            # Gemma via Novita AI (OpenAI-compatible)
            self._setup = SETUP_2A_AGENTIC_GEMMA
        else:
            raise ValueError(f"Unknown setup: {self.setup_name}")

        self._group_id = self._setup.storage.group_id

        logger.info(f"Initializing {self._setup.name}...")
        logger.info(f"  Group ID: {self._group_id}")
        logger.info(f"  LLM Extraction: {self._setup.llm_extraction.name}")  # type: ignore[possibly-missing-attribute]
        logger.info(f"  Embedder: {self._setup.embedder.name}")

        # ============================================================
        # PATCH: Override ALL graphiti prompts to use Indonesian
        # V2 SPECIFIC: Enforce DETAIL RETENTION
        # ============================================================
        import graphiti_core.prompts.snippets as snippets_module
        import graphiti_core.prompts.extract_nodes as extract_nodes_module
        import graphiti_core.prompts.extract_edges as extract_edges_module
        import graphiti_core.prompts.summarize_nodes as summarize_nodes_module
        import graphiti_core.prompts.dedupe_nodes as dedupe_nodes_module
        import graphiti_core.prompts.dedupe_edges as dedupe_edges_module
        import graphiti_core.prompts.invalidate_edges as invalidate_edges_module
        import graphiti_core.prompts.extract_edge_dates as extract_edge_dates_module
        from graphiti_core.prompts.models import Message
        from graphiti_core.prompts.prompt_helpers import to_prompt_json

        # ============================================================
        # 1. SNIPPETS - Summary Instructions (shared)
        # ============================================================
        snippets_module.summary_instructions = (  # type: ignore[invalid-assignment]
            """
ATURAN BAHASA - WAJIB:
- SEMUA output HARUS dalam Bahasa Indonesia.
- DILARANG menggunakan Bahasa Inggris.

Panduan "HYPER-DETAIL" (WAJIB DIPATUHI):
1. JANGAN PERNAH menyensor detail sensorik (rasa, bau, warna, tekstur).
2. JANGAN meringkas nama unik menjadi kategori umum.
   - SALAH: "Aisha memesan kopi."
   - BENAR: "Aisha memesan kopi signature dengan rempah jahe dan kayu manis."
3. Ringkasan harus "VERBOSE" dan "DESKRIPTIF". Panjang maksimal 1500 karakter.
4. Setiap fakta unik (angka, tanggal, nama tempat, alasan emosional) HARUS masuk.

CONTOH STANDAR DETAIL:
✓ "Aisha memesan kopi signature dengan rempah jahe dan kayu manis di Kafe Sudut Tenang pukul 14:30. Dia merasa kopi itu unik meski awalnya aneh. Tempatnya bersih dan cocok untuk foto produk." (Sangat Bagus)
✓ "Aisha dan Dewi berdebat soal budget influencer, Dewi menyarankan alokasi 50% untuk mikro-influencer karena engagement rate mereka lebih tinggi." (Sangat Bagus)

CONTOH SALAH (DILARANG):
✗ "Aisha pergi ke kafe dan minum kopi." (TERLALU UMUM - GAGAL)
✗ "Aisha mendiskusikan budget dengan Dewi." (HILANG ANGKA/ALASAN - GAGAL)
✗ "Aisha ordered coffee..." (BAHASA INGGRIS - GAGAL)
"""
        )

        # ============================================================
        # 2. EXTRACT_NODES - Entity Extraction from Conversations
        # ============================================================
        def patched_extract_message(context):
            sys_prompt = """Kamu adalah asisten AI yang mengekstrak entitas dari percakapan harian dalam Bahasa Indonesia.
Tugas utamamu adalah mengekstrak pembicara dan entitas penting yang disebutkan dalam percakapan.
Fokus pada: nama orang, tempat, organisasi, produk, acara, dan konsep penting."""

            user_prompt = f"""
<TIPE ENTITAS>
{context["entity_types"]}
</TIPE ENTITAS>

<PESAN SEBELUMNYA>
{to_prompt_json([ep for ep in context["previous_episodes"]])}
</PESAN SEBELUMNYA>

<PESAN SAAT INI>
{context["episode_content"]}
</PESAN SAAT INI>

Instruksi:
Kamu diberikan konteks percakapan dan PESAN SAAT INI. Tugasmu adalah mengekstrak **entitas** yang disebutkan **secara eksplisit atau implisit** dalam PESAN SAAT INI.
Referensi kata ganti seperti dia/mereka harus disambiguasi ke nama entitas yang dirujuk.

1. **Ekstraksi Pembicara**: Selalu ekstrak pembicara (bagian sebelum tanda titik dua `:`) sebagai entitas pertama.
   - Jika pembicara disebutkan lagi dalam pesan, perlakukan sebagai **satu entitas**.

2. **Identifikasi Entitas**:
   - Ekstrak semua entitas, konsep, atau aktor penting yang disebutkan dalam PESAN SAAT INI.
   - **Jangan** ekstrak entitas yang hanya disebutkan dalam PESAN SEBELUMNYA (hanya untuk konteks).
   - Perhatikan nama-nama Indonesia (Aisha, Rizky, Budi, Putri, dll.)
   - Perhatikan tempat-tempat Indonesia (Jakarta, Bandung, Kafe Latte, Mall Grand Indonesia, dll.)

3. **Klasifikasi Entitas**:
   - Gunakan deskripsi dalam TIPE ENTITAS untuk mengklasifikasikan setiap entitas.
   - Berikan `entity_type_id` yang sesuai.

4. **Pengecualian**:
   - JANGAN ekstrak tanggal, waktu, atau informasi temporal.
   - JANGAN ekstrak hubungan atau aksi sebagai entitas.

5. **Format**:
   - Gunakan nama lengkap jika tersedia.
   - Eksplisit dan tidak ambigu dalam penamaan.

{context["custom_prompt"]}
"""
            return [
                Message(role="system", content=sys_prompt),
                Message(role="user", content=user_prompt),
            ]

        def patched_extract_nodes_reflexion(context):
            sys_prompt = "Kamu adalah asisten AI yang menentukan entitas mana yang belum diekstrak dari konteks yang diberikan."
            user_prompt = f"""
<PESAN SEBELUMNYA>
{to_prompt_json([ep for ep in context["previous_episodes"]])}
</PESAN SEBELUMNYA>
<PESAN SAAT INI>
{context["episode_content"]}
</PESAN SAAT INI>

<ENTITAS YANG DIEKSTRAK>
{context["extracted_entities"]}
</ENTITAS YANG DIEKSTRAK>

Berdasarkan pesan-pesan di atas dan daftar entitas yang diekstrak, tentukan apakah ada entitas yang belum diekstrak.
Fokus pada nama orang, tempat, dan konsep penting dalam percakapan Indonesia.
"""
            return [
                Message(role="system", content=sys_prompt),
                Message(role="user", content=user_prompt),
            ]

        def patched_extract_summary(context):
            return [
                Message(
                    role="system",
                    content="""Kamu adalah asisten yang mengekstrak ringkasan entitas.

PENTING - FORMAT OUTPUT JSON:
Field name HARUS "summary" (English), bukan "ringkasan".
ISI summary dalam Bahasa Indonesia.

Contoh output yang BENAR:
{"summary": "Aisha adalah Content Creator di Bandung yang bekerja dengan agensi XYZ. Dia menyukai kopi dengan rasa unik seperti jahe dan sering bekerja di kafe untuk mencari inspirasi."}

CONTOH SALAH (JANGAN GUNAKAN):
{"ringkasan": "..."} ❌ SALAH! Gunakan "summary" bukan "ringkasan"

Ringkasan HARUS MENDETAIL (maksimal 500 karakter). Jangan memotong informasi penting.""",
                ),
                Message(
                    role="user",
                    content=f"""
Berdasarkan PESAN dan ENTITAS, perbarui ringkasan yang menggabungkan informasi relevan tentang entitas.

{snippets_module.summary_instructions}

<PESAN>
{to_prompt_json(context["previous_episodes"])}
{to_prompt_json(context["episode_content"])}
</PESAN>

<ENTITAS>
{context["node"]}
</ENTITAS>

Output JSON dengan field "summary" (BUKAN "ringkasan"). Isi ringkasan dalam Bahasa Indonesia.
""",
                ),
            ]

        # Apply extract_nodes patches
        extract_nodes_module.versions["extract_message"] = patched_extract_message
        extract_nodes_module.versions["reflexion"] = patched_extract_nodes_reflexion
        extract_nodes_module.versions["extract_summary"] = patched_extract_summary

        # ============================================================
        # 3. EXTRACT_EDGES - Fact/Relation Extraction
        # ============================================================
        def patched_extract_edge(context):
            return [
                Message(
                    role="system",
                    content="""Kamu adalah pengekstrak fakta ahli yang mengekstrak tripel fakta dari percakapan dalam Bahasa Indonesia.

PENTING - ATURAN BAHASA:
- SEMUA output harus dalam Bahasa Indonesia, TERMASUK field 'fact'.
- JANGAN gunakan Bahasa Inggris sama sekali.
- Field 'fact' harus berupa kalimat Bahasa Indonesia yang natural.

PENTING - DETAIL & SPESIFISITAS (Hyper-Fidelity extraction):
- PERTAHANKAN semua detail sensorik (rasa seperti 'jahe', 'manis', bau, warna), nama tempat spesifik ('Sudut Tenang'), dan angka ('50%', '20 juta').
- JANGAN MERINGKAS detail penting menjadi generalisasi.
   - JIKA teks bilang "kopi rempah jahe", JANGAN tulis "kopi" atau "minuman". Tulis "kopi rempah jahe".
- Masukkan adjektiva/sifat ke dalam deskripsi fakta.

Contoh 'fact' yang BENAR (Kaya Detail):
- "Aisha meminum kopi signature dengan rempah jahe dan kayu manis di Kafe Sudut Tenang." (Bukan "Aisha minum kopi")
- "Aisha menyukai interior Kafe Sudut Tenang yang 'clean' dan cocok untuk foto produk." (Bukan "Aisha suka kafe")
- "Rizky menyarankan konsep 'lifestyle' karena lebih 'jual mimpi' ke anak muda." (Bukan "Rizky menyarankan konsep")

1. Fakta yang diekstrak harus menyertakan informasi waktu yang relevan.
2. Gunakan WAKTU SAAT INI sebagai waktu pesan dikirim.""",
                ),
                Message(
                    role="user",
                    content=f"""
<TIPE FAKTA>
{context["edge_types"]}
</TIPE FAKTA>

<PESAN_SEBELUMNYA>
{to_prompt_json([ep for ep in context["previous_episodes"]])}
</PESAN_SEBELUMNYA>

<PESAN_SAAT_INI>
{to_prompt_json(context["episode_content"])}
</PESAN_SAAT_INI>

<ENTITAS>
{to_prompt_json(context["nodes"])}
</ENTITAS>

<WAKTU_REFERENSI>
{context["reference_time"]}  # ISO 8601 (UTC); untuk menyelesaikan penyebutan waktu relatif
</WAKTU_REFERENSI>

# TUGAS
Ekstrak semua hubungan faktual antara ENTITAS yang diberikan berdasarkan PESAN SAAT INI.
Hanya ekstrak fakta yang:
- melibatkan dua ENTITAS BERBEDA dari daftar ENTITAS,
- dinyatakan dengan jelas atau tersirat tanpa ambigu dalam PESAN SAAT INI,
- dapat direpresentasikan sebagai edge dalam knowledge graph.
- Fakta harus menggunakan nama entitas, bukan kata ganti (dia/mereka).

INSTRUKSI KHUSUS V2 (Detail Retention):
- Jika fakta mengandung detail unik (nama menu, alasan emosi, lokasi spesifik), WAJIB dimasukkan ke dalam teks fakta.
- JANGAN menyederhanakan kalimat jika itu menghilangkan konteks penting.
- Lebih baik fakta sedikit panjang tapi lengkap daripada pendek tapi hilang konteks.

Perhatikan pola temporal Indonesia:
- "kemarin", "tadi pagi", "minggu lalu", "bulan depan"
- "jam 3 sore", "pukul 14:30", "siang ini"

{context["custom_prompt"]}

# ATURAN EKSTRAKSI

1. **Validasi ID Entitas**: `source_entity_id` dan `target_entity_id` harus menggunakan nilai `id` dari daftar ENTITAS.
   - **KRITIS**: Menggunakan ID yang tidak ada dalam daftar akan menyebabkan edge ditolak
2. Setiap fakta harus melibatkan dua entitas **berbeda**.
3. Gunakan string SCREAMING_SNAKE_CASE sebagai `relation_type` (misalnya, BERTEMU, BEKERJA_DI, TINGGAL_DI).
4. Jangan output fakta duplikat atau redundan secara semantik.
5. `fact` harus memparafrasekan kalimat sumber asli dalam Bahasa Indonesia.
6. Gunakan `WAKTU_REFERENSI` untuk menyelesaikan ekspresi temporal relatif.

# ATURAN DATETIME

- Gunakan ISO 8601 dengan sufiks "Z" (UTC) (misalnya, 2025-04-30T00:00:00Z).
- Jika fakta berlangsung (present tense), set `valid_at` ke WAKTU_REFERENSI.
- Jika perubahan/penghentian diekspresikan, set `invalid_at` ke timestamp yang relevan.
- Biarkan kedua field `null` jika tidak ada waktu yang dinyatakan atau dapat diselesaikan.
""",
                ),
            ]

        def patched_extract_edges_reflexion(context):
            sys_prompt = "Kamu adalah asisten AI yang menentukan fakta mana yang belum diekstrak dari konteks yang diberikan."
            user_prompt = f"""
<PESAN SEBELUMNYA>
{to_prompt_json([ep for ep in context["previous_episodes"]])}
</PESAN SEBELUMNYA>
<PESAN SAAT INI>
{context["episode_content"]}
</PESAN SAAT INI>

<ENTITAS YANG DIEKSTRAK>
{context["nodes"]}
</ENTITAS YANG DIEKSTRAK>

<FAKTA YANG DIEKSTRAK>
{context["extracted_facts"]}
</FAKTA YANG DIEKSTRAK>

Berdasarkan PESAN, daftar ENTITAS, dan daftar FAKTA yang diekstrak; tentukan apakah ada fakta yang belum diekstrak.
"""
            return [
                Message(role="system", content=sys_prompt),
                Message(role="user", content=user_prompt),
            ]

        # Apply extract_edges patches
        extract_edges_module.versions["edge"] = patched_extract_edge
        extract_edges_module.versions["reflexion"] = patched_extract_edges_reflexion

        # ============================================================
        # 4. SUMMARIZE_NODES - Summary Generation
        # ============================================================
        def patched_summarize_pair(context):
            return [
                Message(
                    role="system",
                    content="""Kamu adalah asisten yang menggabungkan ringkasan.

PENTING - FORMAT OUTPUT:
- Field name HARUS "summary" (English), bukan "ringkasan"
- ISI summary dalam Bahasa Indonesia
- Ringkasan BOLEH PANJANG (hingga 500 karakter) untuk mempertahankan kekayaan informasi.

Contoh output yang BENAR:
{"summary": "Aisha adalah Content Creator di Bandung yang menyukai fotografi human interest. Dia sedang mengerjakan campaign skincare dan sering mengalami creative block yang diatasi dengan diskusi bersama Rizky."}

Contoh output yang SALAH:
{"ringkasan": "..."} ❌ SALAH!""",
                ),
                Message(
                    role="user",
                    content=f"""
Sintesis informasi dari dua ringkasan berikut menjadi satu ringkasan singkat dalam Bahasa Indonesia.

Ringkasan:
{to_prompt_json(context["node_summaries"])}

Output JSON dengan field "summary" (BUKAN "ringkasan"), isi dalam Bahasa Indonesia, maksimal 500 karakter. Pertahankan detail unik dari kedua sumber.
""",
                ),
            ]

        def patched_summarize_context(context):
            return [
                Message(
                    role="system",
                    content="""Kamu adalah asisten yang menghasilkan ringkasan dan atribut dari teks percakapan.

PENTING - FORMAT OUTPUT JSON:
Field names HARUS dalam English (untuk kompatibilitas sistem), tapi ISI/VALUES dalam Bahasa Indonesia.

Contoh output yang BENAR:
```json
{
    "summary": "Aisha adalah Content Creator di Bandung yang sedang mengerjakan campaign skincare.",
    "attributes": {"lokasi": "Bandung", "pekerjaan": "Content Creator"}
}
```

Contoh output yang SALAH (jangan gunakan):
```json
{
    "ringkasan": "..."  // ❌ SALAH! Gunakan "summary" bukan "ringkasan"
}
```""",
                ),
                Message(
                    role="user",
                    content=f"""
Berdasarkan PESAN dan nama ENTITAS, buat ringkasan untuk ENTITAS dalam Bahasa Indonesia.

{snippets_module.summary_instructions}

<PESAN>
{to_prompt_json(context["previous_episodes"])}
{to_prompt_json(context["episode_content"])}
</PESAN>

<ENTITAS>
{context["node_name"]}
</ENTITAS>

<KONTEKS ENTITAS>
{context["node_summary"]}
</KONTEKS ENTITAS>

<ATRIBUT>
{to_prompt_json(context["attributes"])}
</ATRIBUT>

PENTING: Output JSON dengan field "summary" (BUKAN "ringkasan"). Isi summary dalam Bahasa Indonesia.
""",
                ),
            ]

        def patched_summary_description(context):
            return [
                Message(
                    role="system",
                    content="""Kamu adalah asisten yang mendeskripsikan konten dalam satu kalimat.

PENTING - FORMAT OUTPUT:
- Field name HARUS "description" (English)
- ISI description dalam Bahasa Indonesia
- Di bawah 500 karakter (izinkan deskripsi panjang dan lengkap)

Contoh output yang BENAR:
{"description": "Informasi mendetail tentang Aisha sebagai Content Creator di Bandung, proyek-proyek agensinya, masalah finansial yang dihadapi, serta interaksi sosialnya dengan Dewi dan Rizky."}""",
                ),
                Message(
                    role="user",
                    content=f"""
Buat deskripsi singkat satu kalimat dari ringkasan yang menjelaskan jenis informasi apa yang dirangkum.
Output dalam Bahasa Indonesia, di bawah 250 karakter.

Ringkasan:
{to_prompt_json(context["summary"])}

Output JSON dengan field "description" (English), isi dalam Bahasa Indonesia.
""",
                ),
            ]

        # Apply summarize_nodes patches
        summarize_nodes_module.versions["summarize_pair"] = patched_summarize_pair
        summarize_nodes_module.versions["summarize_context"] = patched_summarize_context
        summarize_nodes_module.versions["summary_description"] = (
            patched_summary_description
        )

        # ============================================================
        # 5. DEDUPE_NODES - Entity Deduplication
        # ============================================================
        def patched_dedupe_node(context):
            return [
                Message(
                    role="system",
                    content="""Kamu adalah asisten yang menentukan apakah ENTITAS BARU adalah duplikat dari ENTITAS YANG ADA.
Pertimbangkan variasi nama Indonesia (Rizky = Rizki = Rizqy, Aisha = Aisya).

PENTING - FORMAT OUTPUT:
Output harus berupa JSON dengan struktur PERSIS seperti ini. SEMUA field WAJIB ada:
- id: integer (dari ENTITAS BARU)
- name: string (nama lengkap entitas) - WAJIB ADA
- duplicate_idx: integer (-1 jika tidak ada duplikat, BUKAN null/None) - WAJIB INTEGER
- duplicates: list of integers ([] jika tidak ada) - WAJIB ADA""",
                ),
                Message(
                    role="user",
                    content=f"""
<PESAN SEBELUMNYA>
{to_prompt_json([ep for ep in context["previous_episodes"]])}
</PESAN SEBELUMNYA>
<PESAN SAAT INI>
{context["episode_content"]}
</PESAN SAAT INI>
<ENTITAS BARU>
{to_prompt_json(context["extracted_node"])}
</ENTITAS BARU>
<DESKRIPSI TIPE ENTITAS>
{to_prompt_json(context["entity_type_description"])}
</DESKRIPSI TIPE ENTITAS>

<ENTITAS YANG ADA>
{to_prompt_json(context["existing_nodes"])}
</ENTITAS YANG ADA>

Tentukan apakah ENTITAS BARU adalah duplikat dari salah satu ENTITAS YANG ADA.

Perhatikan variasi ejaan nama Indonesia:
- Rizky, Rizki, Rizqy = sama
- Aisha, Aisya, Aisyah = sama

TUGAS:
1. Bandingkan ENTITAS BARU dengan setiap item di ENTITAS YANG ADA.
2. Jika merujuk pada objek/konsep yang sama, catat indexnya.
3. Set duplicate_idx = index terkecil, atau -1 jika tidak ada duplikat.
4. Set duplicates = daftar semua index duplikat, atau [] jika tidak ada.

CONTOH OUTPUT YANG BENAR:
```json
{{
    "entity_resolutions": [
        {{
            "id": 0,
            "name": "Aisha",
            "duplicate_idx": -1,
            "duplicates": []
        }}
    ]
}}
```

CONTOH JIKA ADA DUPLIKAT:
```json
{{
    "entity_resolutions": [
        {{
            "id": 0,
            "name": "Rizky",
            "duplicate_idx": 2,
            "duplicates": [2, 5]
        }}
    ]
}}
```

PERINGATAN:
- duplicate_idx HARUS integer (-1 atau index), BUKAN null/None
- name HARUS ada dan berupa string
- duplicates HARUS ada dan berupa list (bisa kosong [])
""",
                ),
            ]

        def patched_dedupe_nodes(context):
            num_entities = len(context["extracted_nodes"])
            return [
                Message(
                    role="system",
                    content="""Kamu adalah asisten yang menentukan apakah ENTITAS yang diekstrak dari percakapan adalah duplikat dari entitas yang ada.
Pertimbangkan variasi nama Indonesia (Rizky = Rizki, Aisha = Aisya).

PENTING - FORMAT OUTPUT:
Untuk SETIAP entitas, output JSON dengan field berikut. SEMUA field WAJIB:
- id: integer (dari ENTITAS yang diekstrak)
- name: string (nama lengkap entitas) - WAJIB ADA
- duplicate_idx: integer (-1 jika tidak ada duplikat) - WAJIB INTEGER, BUKAN null
- duplicates: list of integers ([] jika tidak ada) - WAJIB ADA""",
                ),
                Message(
                    role="user",
                    content=f"""
<PESAN SEBELUMNYA>
{to_prompt_json([ep for ep in context["previous_episodes"]])}
</PESAN SEBELUMNYA>
<PESAN SAAT INI>
{context["episode_content"]}
</PESAN SAAT INI>

<ENTITAS>
{to_prompt_json(context["extracted_nodes"])}
</ENTITAS>

<ENTITAS YANG ADA>
{to_prompt_json(context["existing_nodes"])}
</ENTITAS YANG ADA>

Untuk setiap ENTITAS, tentukan apakah duplikat dari ENTITAS YANG ADA.

JUMLAH ENTITAS: {num_entities}
Responsmu HARUS menyertakan TEPAT {num_entities} resolusi dengan ID 0 sampai {num_entities - 1}.

CONTOH OUTPUT UNTUK 2 ENTITAS:
```json
{{
    "entity_resolutions": [
        {{
            "id": 0,
            "name": "Aisha",
            "duplicate_idx": -1,
            "duplicates": []
        }},
        {{
            "id": 1,
            "name": "Bot",
            "duplicate_idx": 3,
            "duplicates": [3]
        }}
    ]
}}
```

PERINGATAN KRITIS:
- SETIAP entity_resolution HARUS memiliki field: id, name, duplicate_idx, duplicates
- duplicate_idx HARUS integer (-1 atau index), TIDAK BOLEH null/None
- name HARUS string, TIDAK BOLEH kosong
- duplicates HARUS list integer (bisa kosong [])
- JANGAN gunakan field lain seperti "is_duplicate" 
""",
                ),
            ]

        # Apply dedupe_nodes patches
        dedupe_nodes_module.versions["node"] = patched_dedupe_node
        dedupe_nodes_module.versions["nodes"] = patched_dedupe_nodes

        # ============================================================
        # 6. DEDUPE_EDGES - Fact/Edge Deduplication
        # ============================================================
        def patched_dedupe_edge(context):
            return [
                Message(
                    role="system",
                    content="Kamu adalah asisten yang menghapus duplikat fakta dari daftar fakta.",
                ),
                Message(
                    role="user",
                    content=f"""
Berdasarkan konteks berikut, tentukan apakah Fakta Baru mewakili salah satu fakta dalam daftar Fakta Yang Ada.

<FAKTA YANG ADA>
{to_prompt_json(context["related_edges"])}
</FAKTA YANG ADA>

<FAKTA BARU>
{to_prompt_json(context["extracted_edges"])}
</FAKTA BARU>

Tugas:
Jika Fakta Baru mewakili informasi faktual yang sama dengan fakta di Fakta Yang Ada, kembalikan id fakta duplikat sebagai bagian dari daftar duplicate_facts.
Jika FAKTA BARU bukan duplikat dari FAKTA YANG ADA, kembalikan daftar kosong.

Panduan:
1. Fakta tidak perlu identik sempurna untuk menjadi duplikat, mereka hanya perlu mengekspresikan informasi yang sama.
""",
                ),
            ]

        def patched_resolve_edge(context):
            return [
                Message(
                    role="system",
                    content="Kamu adalah asisten yang menghapus duplikat fakta dari daftar fakta dan menentukan fakta mana yang bertentangan dengan fakta baru.",
                ),
                Message(
                    role="user",
                    content=f"""
Tugas:
Kamu akan menerima DUA daftar fakta terpisah. Setiap daftar menggunakan 'idx' sebagai field index, dimulai dari 0.

1. DETEKSI DUPLIKAT:
   - Jika FAKTA BARU mewakili informasi faktual identik dengan fakta di FAKTA YANG ADA, kembalikan nilai idx tersebut di duplicate_facts.
   - Fakta dengan informasi serupa yang memiliki perbedaan kunci TIDAK boleh ditandai sebagai duplikat.
   - Kembalikan nilai idx dari FAKTA YANG ADA.
   - Jika tidak ada duplikat, kembalikan daftar kosong untuk duplicate_facts.

2. KLASIFIKASI TIPE FAKTA:
   - Berdasarkan TIPE FAKTA yang ditentukan, tentukan apakah FAKTA BARU harus diklasifikasikan sebagai salah satu tipe ini.
   - Kembalikan tipe fakta sebagai fact_type atau DEFAULT jika FAKTA BARU bukan salah satu TIPE FAKTA.

3. DETEKSI KONTRADIKSI:
   - Berdasarkan KANDIDAT INVALIDASI FAKTA dan FAKTA BARU, tentukan fakta mana yang bertentangan dengan fakta baru.
   - Kembalikan nilai idx dari KANDIDAT INVALIDASI FAKTA.
   - Jika tidak ada kontradiksi, kembalikan daftar kosong untuk contradicted_facts.

<TIPE FAKTA>
{context["edge_types"]}
</TIPE FAKTA>

<FAKTA YANG ADA>
{context["existing_edges"]}
</FAKTA YANG ADA>

<KANDIDAT INVALIDASI FAKTA>
{context["edge_invalidation_candidates"]}
</KANDIDAT INVALIDASI FAKTA>

<FAKTA BARU>
{context["new_edge"]}
</FAKTA BARU>
""",
                ),
            ]

        # Apply dedupe_edges patches
        dedupe_edges_module.versions["edge"] = patched_dedupe_edge
        dedupe_edges_module.versions["resolve_edge"] = patched_resolve_edge

        # ============================================================
        # 7. INVALIDATE_EDGES - Edge Invalidation
        # ============================================================
        def patched_invalidate_v2(context):
            return [
                Message(
                    role="system",
                    content="Kamu adalah asisten AI yang menentukan fakta mana yang saling bertentangan.",
                ),
                Message(
                    role="user",
                    content=f"""
Berdasarkan FAKTA YANG ADA dan FAKTA BARU, tentukan fakta yang ada mana yang bertentangan dengan fakta baru.
Kembalikan daftar yang berisi semua id fakta yang bertentangan dengan FAKTA BARU.
Jika tidak ada fakta yang bertentangan, kembalikan daftar kosong.

<FAKTA YANG ADA>
{context["existing_edges"]}
</FAKTA YANG ADA>

<FAKTA BARU>
{context["new_edge"]}
</FAKTA BARU>
""",
                ),
            ]

        # Apply invalidate_edges patch
        invalidate_edges_module.versions["v2"] = patched_invalidate_v2

        # ============================================================
        # 8. EXTRACT_EDGE_DATES - Temporal Extraction
        # ============================================================
        def patched_extract_dates_v1(context):
            return [
                Message(
                    role="system",
                    content="Kamu adalah asisten AI yang mengekstrak informasi datetime untuk edge graph, fokus hanya pada tanggal yang terkait langsung dengan pembentukan atau perubahan hubungan dalam fakta edge.",
                ),
                Message(
                    role="user",
                    content=f"""
<PESAN SEBELUMNYA>
{context["previous_episodes"]}
</PESAN SEBELUMNYA>
<PESAN SAAT INI>
{context["current_episode"]}
</PESAN SAAT INI>
<TIMESTAMP REFERENSI>
{context["reference_timestamp"]}
</TIMESTAMP REFERENSI>

<FAKTA>
{context["edge_fact"]}
</FAKTA>

PENTING: Hanya ekstrak informasi waktu jika merupakan bagian dari fakta yang diberikan. Pastikan untuk menentukan tanggal jika hanya waktu relatif yang disebutkan.

Pola temporal Indonesia:
- "kemarin" = reference_timestamp - 1 hari
- "minggu lalu" = reference_timestamp - 7 hari
- "bulan lalu" = reference_timestamp - 1 bulan
- "tadi pagi" = reference_timestamp dengan jam pagi
- "siang ini" = reference_timestamp dengan jam siang
- "nanti malam" = reference_timestamp dengan jam malam

Definisi:
- valid_at: Tanggal dan waktu ketika hubungan yang dijelaskan oleh fakta edge menjadi benar atau terbentuk.
- invalid_at: Tanggal dan waktu ketika hubungan yang dijelaskan oleh fakta edge berhenti benar atau berakhir.

Tugas:
Analisis percakapan dan tentukan apakah ada tanggal yang merupakan bagian dari fakta edge.

Panduan:
1. Gunakan format ISO 8601 (YYYY-MM-DDTHH:MM:SS.SSSSSSZ) untuk datetime.
2. Gunakan timestamp referensi sebagai waktu saat ini.
3. Jika fakta ditulis dalam present tense, gunakan Timestamp Referensi untuk tanggal valid_at.
4. Jika tidak ada informasi temporal, biarkan field sebagai null.
5. Untuk penyebutan waktu relatif, hitung datetime aktual berdasarkan timestamp referensi.
6. Jika hanya tanggal disebutkan tanpa waktu spesifik, gunakan 00:00:00.
7. Jika hanya tahun disebutkan, gunakan 1 Januari tahun tersebut pukul 00:00:00.
""",
                ),
            ]

        # Apply extract_edge_dates patch
        extract_edge_dates_module.versions["v1"] = patched_extract_dates_v1

        # ============================================================
        # CRITICAL: Also patch the prompt_library WRAPPERS directly!
        # The PromptLibraryWrapper caches function references at import time,
        # so patching the versions dict alone has NO EFFECT.
        # We must patch the wrapper's .func attribute directly.
        # ============================================================
        from graphiti_core.prompts.lib import prompt_library

        # Patch extract_nodes wrappers
        prompt_library.extract_nodes.extract_message.func = patched_extract_message  # type: ignore[unresolved-attribute]
        prompt_library.extract_nodes.reflexion.func = patched_extract_nodes_reflexion  # type: ignore[unresolved-attribute]
        prompt_library.extract_nodes.extract_summary.func = patched_extract_summary  # type: ignore[unresolved-attribute]

        # Patch extract_edges wrappers
        prompt_library.extract_edges.edge.func = patched_extract_edge  # type: ignore[unresolved-attribute]
        prompt_library.extract_edges.reflexion.func = patched_extract_edges_reflexion  # type: ignore[unresolved-attribute]

        # Patch summarize_nodes wrappers
        prompt_library.summarize_nodes.summarize_pair.func = patched_summarize_pair  # type: ignore[unresolved-attribute]
        prompt_library.summarize_nodes.summarize_context.func = (  # type: ignore[attr-defined]
            patched_summarize_context
        )
        prompt_library.summarize_nodes.summary_description.func = (  # type: ignore[attr-defined]
            patched_summary_description
        )

        # Patch dedupe_nodes wrappers
        prompt_library.dedupe_nodes.node.func = patched_dedupe_node  # type: ignore[unresolved-attribute]
        prompt_library.dedupe_nodes.nodes.func = patched_dedupe_nodes  # type: ignore[unresolved-attribute]

        # Patch dedupe_edges wrappers
        prompt_library.dedupe_edges.edge.func = patched_dedupe_edge  # type: ignore[unresolved-attribute]
        prompt_library.dedupe_edges.resolve_edge.func = patched_resolve_edge  # type: ignore[unresolved-attribute]

        # Patch invalidate_edges wrappers
        prompt_library.invalidate_edges.v2.func = patched_invalidate_v2  # type: ignore[unresolved-attribute]

        # Patch extract_edge_dates wrappers
        prompt_library.extract_edge_dates.v1.func = patched_extract_dates_v1  # type: ignore[unresolved-attribute]

        logger.info(
            "  ✅ Patched ALL graphiti prompts to Indonesian (8 modules + 14 wrappers)"
        )

        # Initialize Graphiti
        from graphiti_core import Graphiti
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        # Shim for Neo4jDriver to handle "Index already exists" race conditions
        # Shim for Neo4jDriver to handle "Index already exists" race conditions
        class Neo4jDriverShim(Neo4jDriver):
            def __init__(
                self, uri: str, user: str, password: str, database: str = "neo4j"
            ):
                # Store credentials for reconnection
                self._uri = uri
                self._user = user
                self._password = password
                self._database = database
                # Call parent init
                super().__init__(
                    uri=uri, user=user, password=password, database=database
                )

            async def execute_query(self, cypher_query_, parameters_=None, **kwargs):
                from neo4j.exceptions import ClientError
                import sys
                import os

                # Consolidate parameters from all possible sources
                # Parent class expects 'params' in kwargs, but some callers might use 'parameters_'
                params_from_kwargs = kwargs.pop("params", {})
                parameters_from_kwargs = kwargs.pop("parameters_", {})

                params = (
                    parameters_ or params_from_kwargs or parameters_from_kwargs or {}
                )

                # IMPORTANT: Graphiti often passes parameters as direct kwargs (e.g. search_vector=...)
                # We must capture and move them into 'params' for sanitization.
                # We'll treat ANY leftover kwarg as a potential parameter,
                # except known non-param keys if any.
                keys_to_move = list(kwargs.keys())
                for k in keys_to_move:
                    params[k] = kwargs.pop(k)  # Move to params dict, remove from kwargs

                # SANITIZATION AND DEBUG LOGGING
                try:
                    # Determine log file path - ensure it is in current directory
                    log_file = os.path.join(os.getcwd(), "ingest_debug_v2.log")

                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("\n--- QUERY EXECUTION ---\n")
                        # Check for vector parameters and SANITIZE
                        sanitized_params = params.copy()
                        updates_made = False

                        for k, v in params.items():
                            # Check if it looks like a vector
                            if "vector" in k or (isinstance(v, list) and len(v) > 100):
                                v_type = type(v)
                                v_len = len(v) if isinstance(v, list) else "N/A"
                                f.write(f"🔍 PARAM '{k}': Type={v_type}, Len={v_len}\n")

                                if isinstance(v, list) and len(v) > 0:
                                    first_elem = v[0]
                                    f.write(
                                        f"    Sample[0]: {first_elem} (Type: {type(first_elem)})\n"
                                    )

                                    # CASE 1: NESTED LIST (List[List[float]]) -> Flatten
                                    if isinstance(first_elem, list):
                                        f.write(
                                            "    ⚠️ DETECTED NESTED LIST. FLATTENING...\n"
                                        )
                                        if len(v) == 1:
                                            sanitized_params[k] = v[0]
                                            updates_made = True
                                            f.write(
                                                "    ✅ FIXED: Flattened single-item nested list.\n"
                                            )
                                        else:
                                            # Flatten list of lists to single list if that was the intent...
                                            # OR just take the first one?
                                            # Let's assume it should be a flat list of floats.
                                            # If we have [[0.1, 0.2], [0.3, 0.4]], flattening might be wrong if it's batch keys.
                                            # But for `$search_vector` it MUST be a single vector.
                                            f.write(
                                                f"    ⚠️ WARNING: Nested list len={len(v)}. Taking v[0].\n"
                                            )
                                            sanitized_params[k] = v[0]
                                            updates_made = True

                                    # CASE 2: NUMPY or other types?
                                    # Ensure everything is valid float
                                    try:
                                        # Re-read potentially updated value
                                        current_vec = (
                                            sanitized_params[k] if updates_made else v
                                        )

                                        # Check for NaNs
                                        has_nan = False
                                        for i, x in enumerate(current_vec):
                                            if isinstance(x, float) and (
                                                x != x
                                                or x == float("inf")
                                                or x == float("-inf")
                                            ):
                                                has_nan = True
                                                break

                                        if has_nan:
                                            f.write(
                                                "    🚨 ERROR: VECTOR CONTAINS NaN/Inf! ZEROING OUT...\n"
                                            )
                                            sanitized_params[k] = [0.0] * len(
                                                current_vec
                                            )
                                            updates_made = True

                                    except Exception as e:
                                        f.write(
                                            f"    ⚠️ Error checking vector content: {e}\n"
                                        )

                        if updates_made:
                            params = sanitized_params
                            # Update parameters_ passed to super if possible, but execute_query takes it as arg
                            # We will pass 'params' as 'parameters_' argument explicitly
                            f.write("✅ SANITIZATION APPLIED.\n")
                        else:
                            f.write("No sanitization needed.\n")

                        f.flush()  # FORCE WRITE

                except Exception as debug_err:
                    # Last resort print
                    sys.stderr.write(f"DEBUG LOGGING FAILED: {debug_err}\n")

                try:
                    # Pass params as 'params' because parent class expects it in kwargs and pops it
                    return await super().execute_query(
                        cypher_query_, params=params, **kwargs
                    )
                except ClientError as e:
                    # Ignore "EquivalentSchemaRuleAlreadyExists" error
                    if "EquivalentSchemaRuleAlreadyExists" in str(
                        e
                    ) or "An equivalent index already exists" in str(e):
                        return [], None, []

                    try:
                        with open("ingest_debug_v2.log", "a", encoding="utf-8") as f:
                            f.write(f"NEO4J ERROR: {e}\n")
                    except Exception:
                        pass
                    raise e
                except Exception as conn_error:
                    # Handle SessionExpired and connection errors with robust retry
                    import asyncio

                    error_str = str(conn_error).lower()

                    # Check if it's a connection-related error
                    connection_errors = [
                        "sessionexpired",
                        "defunct connection",
                        "semaphore timeout",
                        "connection refused",
                        "connection reset",
                        "timed out",
                        "network is unreachable",
                        "broken pipe",
                    ]
                    is_connection_error = any(
                        err in error_str for err in connection_errors
                    )

                    if is_connection_error:
                        # Robust retry with multiple attempts
                        max_retries = 3
                        for retry_num in range(max_retries):
                            logger.warning(
                                f"⚠️ Neo4j connection error (attempt {retry_num + 1}/{max_retries}): {str(conn_error)[:80]}..."
                            )

                            # Wait before retry - longer for each attempt
                            wait_time = 10.0 * (retry_num + 1)
                            logger.info(
                                f"   Waiting {wait_time}s before reconnecting..."
                            )
                            await asyncio.sleep(wait_time)

                            try:
                                # Recreate the internal client to get fresh connection
                                from neo4j import AsyncGraphDatabase

                                if hasattr(self, "client") and self.client:
                                    try:
                                        await self.client.close()
                                    except Exception:
                                        pass

                                # Re-establish connection using stored credentials
                                self.client = AsyncGraphDatabase.driver(
                                    self._uri, auth=(self._user, self._password)
                                )

                                # Retry the query
                                return await super().execute_query(
                                    cypher_query_, params=params, **kwargs
                                )
                            except Exception as retry_error:
                                if retry_num == max_retries - 1:
                                    logger.error(
                                        f"❌ Neo4j connection failed after {max_retries} retries"
                                    )
                                    raise retry_error
                                # Continue to next retry
                                conn_error = retry_error
                    else:
                        raise conn_error

        # Create Neo4j driver using Shim
        driver = Neo4jDriverShim(
            uri=config.neo4j.uri,
            user=config.neo4j.user,
            password=config.neo4j.password,
            database=config.neo4j.database,
        )

        # Create LLM client based on setup
        from graphiti_core.llm_client.gemini_client import GeminiClient
        from graphiti_core.llm_client.config import LLMConfig

        # TRACKING WRAPPER
        class TrackingGeminiClient(GeminiClient):
            async def generate_response(self, messages, response_model=None, **kwargs):  # type: ignore[invalid-method-override]
                # Calculate input len
                input_len = 0
                if isinstance(messages, list):
                    for m in messages:
                        if hasattr(m, "content"):
                            input_len += len(str(m.content))
                        elif isinstance(m, dict):
                            input_len += len(str(m.get("content", "")))
                        else:
                            input_len += len(str(m))
                else:
                    input_len = len(str(messages))

                # Call original
                response = await super().generate_response(
                    messages, response_model, **kwargs
                )

                # Calculate output len
                output_len = len(str(response))

                # Track cost using text estimation
                # Gemini Flash: $0.30/1M input tokens, $2.50/1M output tokens
                # Estimation: 1 token ≈ 4 characters
                try:
                    from src.utils.cost_tracker import get_cost_tracker

                    tracker = get_cost_tracker()

                    input_tokens = int(input_len / 4)
                    output_tokens = int(output_len / 4)

                    await tracker.track(input_tokens, output_tokens, self.config.model or "unknown")
                except Exception as e:
                    logger.debug(f"Cost tracking failed: {e}")

                return response

        if self.setup_name == "gemini":
            # Use Gemini for extraction
            llm_config = LLMConfig(
                api_key=config.gemini.api_key,
                model=self._setup.llm_extraction.name,  # type: ignore[possibly-missing-attribute]
                temperature=0.1,  # Low temp for factual extraction
                max_tokens=8192,  # High max tokens for verbose summaries
            )
            # USE TRACKING CLIENT
            llm_client = TrackingGeminiClient(config=llm_config)

            # CRITICAL: Force small_model = main model to prevent untracked flash-lite calls
            # Graphiti internally has DEFAULT_SMALL_MODEL = 'gemini-2.5-flash-lite'
            # which causes unmonitored API calls if we don't override it
            llm_client.small_model = self._setup.llm_extraction.name  # type: ignore[possibly-missing-attribute]
            logger.info(
                f"  Forced small_model = {self._setup.llm_extraction.name} (prevents untracked flash-lite)"  # type: ignore[possibly-missing-attribute]
            )
        else:
            # Use Gemma via Novita AI (OpenAI-compatible API)
            import os

            # Get Novita API key
            novita_api_key = os.getenv("NOVITAAI_API_KEY")
            if not novita_api_key:
                raise ValueError(
                    "NOVITAAI_API_KEY environment variable required for Gemma setup"
                )

            # Custom Novita client that doesn't use structured outputs
            # (Gemma 3 27B on Novita doesn't support structured outputs)
            from graphiti_core.llm_client.client import LLMClient
            from openai import AsyncOpenAI

            class NovitaLLMClient(LLMClient):
                def __init__(self, config: LLMConfig):
                    self.config = config
                    self.client = AsyncOpenAI(
                        api_key=config.api_key,
                        base_url=config.base_url,
                    )

                async def _generate_response(  # type: ignore[override]
                    self, messages, response_model=None, **kwargs
                ):
                    """Abstract method implementation - just calls generate_response."""
                    return await self.generate_response(
                        messages, response_model, **kwargs
                    )

                async def generate_response(  # type: ignore[override]
                    self, messages, response_model=None, **kwargs
                ):
                    """Generate response without structured outputs, parse JSON manually."""
                    import json
                    import re

                    # Calculate input len for cost tracking
                    input_len = 0
                    if isinstance(messages, list):
                        for m in messages:
                            if hasattr(m, "content"):
                                input_len += len(str(m.content))
                            elif isinstance(m, dict):
                                input_len += len(str(m.get("content", "")))
                            else:
                                input_len += len(str(m))
                    else:
                        input_len = len(str(messages))

                    # Convert messages to OpenAI format
                    openai_messages = []
                    for msg in messages:
                        if hasattr(msg, "role") and hasattr(msg, "content"):
                            openai_messages.append(
                                {"role": msg.role, "content": msg.content}
                            )
                        elif isinstance(msg, dict):
                            openai_messages.append(msg)
                        else:
                            openai_messages.append(
                                {"role": "user", "content": str(msg)}
                            )

                    # Gemma 3 27B on Novita doesn't support structured outputs
                    # Add JSON schema to the last user message as instruction
                    if (
                        response_model
                        and hasattr(response_model, "model_json_schema")
                        and openai_messages
                    ):
                        schema = response_model.model_json_schema()
                        json_instruction = f"\n\nIMPORTANT: Respond with valid JSON only. No markdown code blocks, no explanation text. Start directly with {{ and end with }}. Follow this schema:\n{json.dumps(schema, indent=2)}"

                        # Append to last user message
                        for i in range(len(openai_messages) - 1, -1, -1):
                            if openai_messages[i].get("role") == "user":
                                openai_messages[i]["content"] += json_instruction
                                break

                    # Make API call WITHOUT response_format
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = await self.client.chat.completions.create(
                                model=self.config.model,  # type: ignore[invalid-argument-type]
                                messages=openai_messages,
                                temperature=getattr(self.config, "temperature", 0.1),
                                max_tokens=getattr(self.config, "max_tokens", 8192),
                            )

                            content = response.choices[0].message.content

                            # Track cost
                            try:
                                from src.utils.cost_tracker import get_cost_tracker

                                tracker = get_cost_tracker()
                                input_tokens = (
                                    response.usage.prompt_tokens
                                    if response.usage
                                    else int(input_len / 4)
                                )
                                output_tokens = (
                                    response.usage.completion_tokens
                                    if response.usage
                                    else int(len(content or "") / 4)
                                )
                                await tracker.track(
                                    input_tokens, output_tokens, self.config.model or "unknown"
                                )
                            except Exception as e:
                                logger.debug(f"Cost tracking failed: {e}")

                            # Parse JSON from response
                            if response_model:
                                # Clean markdown code blocks
                                clean_content = re.sub(
                                    r"^```json\s*", "", content or "", flags=re.MULTILINE
                                )
                                clean_content = re.sub(
                                    r"^```\s*", "", clean_content, flags=re.MULTILINE
                                )
                                clean_content = re.sub(
                                    r"\s*```$", "", clean_content, flags=re.MULTILINE
                                )
                                clean_content = clean_content.strip()

                                try:
                                    parsed = json.loads(clean_content)

                                    # Fix common issues
                                    if isinstance(parsed, list):
                                        fields = (
                                            list(response_model.model_fields.keys())
                                            if hasattr(response_model, "model_fields")
                                            else ["items"]
                                        )
                                        key = fields[0] if fields else "items"
                                        parsed = {key: parsed}

                                    # Validate with Pydantic
                                    try:
                                        _ = response_model(**parsed)
                                        return parsed
                                    except Exception as val_error:
                                        logger.warning(
                                            f"Pydantic validation failed (attempt {attempt + 1}): {str(val_error)[:100]}"
                                        )
                                        if attempt < max_retries - 1:
                                            continue
                                        return parsed  # Return anyway on last attempt

                                except json.JSONDecodeError as je:
                                    logger.warning(
                                        f"JSON parse failed (attempt {attempt + 1}): {str(je)[:50]}"
                                    )
                                    if attempt < max_retries - 1:
                                        continue
                                    return {"content": content}
                            else:
                                return content

                        except Exception as e:
                            logger.error(
                                f"Novita API error (attempt {attempt + 1}): {e}"
                            )
                            if attempt == max_retries - 1:
                                raise

                    return {}

            llm_config = LLMConfig(
                api_key=novita_api_key,
                model=self._setup.llm_extraction.name,  # type: ignore[possibly-missing-attribute]  # google/gemma-3-27b-it
                base_url="https://api.novita.ai/openai",
            )
            llm_client = NovitaLLMClient(config=llm_config)
            logger.info(
                f"  Using Novita AI (no structured outputs): {self._setup.llm_extraction.name}"  # type: ignore[possibly-missing-attribute]
            )

        # Create embedder based on setup
        if self._setup.embedder.provider == "huggingface":
            # Local HuggingFace embedder - Adapter for Graphiti
            from graphiti_core.embedder.client import EmbedderClient
            from sentence_transformers import SentenceTransformer

            class SentenceTransformerEmbedderShim(EmbedderClient):
                def __init__(self, model_name: str):
                    self.model = SentenceTransformer(model_name)

                async def create(self, input_data) -> Any:
                    import asyncio
                    import functools

                    # Run embedding in executor to avoid blocking event loop
                    loop = asyncio.get_running_loop()

                    is_single_string = isinstance(input_data, str)

                    if is_single_string:
                        func = functools.partial(self.model.encode, input_data)
                    else:
                        encoder: Any = self.model.encode
                        func = functools.partial(
                            encoder,
                            list(input_data),
                            batch_size=32,
                            show_progress_bar=False,
                        )

                    embeddings = await loop.run_in_executor(None, func)

                    # Handle return types
                    if is_single_string:
                        # Ensure we return list[float], not numpy array
                        res = embeddings.tolist()
                        # sys.stderr.write(f"  DEBUG EMBED: Single string -> Len {len(res)}\n")
                        return res
                    else:
                        # Ensure we return list[list[float]]
                        res = embeddings.tolist()
                        # sys.stderr.write(f"  DEBUG EMBED: List[{len(res)}] -> Inner Len {len(res[0]) if len(res)>0 else 0}\n")
                        return res

                async def create_batch(self, input_data) -> Any:  # type: ignore[invalid-method-override]
                    # sys.stderr.write(f"🔍 DEBUG EMBED CREATE_BATCH: Input len={len(input_data)}\n")
                    res = await self.create(input_data)
                    # sys.stderr.write(f"🔍 DEBUG EMBED RESULT: Type={type(res)}, Len={len(res)}\n")
                    return res

            embedder = SentenceTransformerEmbedderShim(
                model_name=self._setup.embedder.name
            )
            logger.info(f"  Using LOCAL embedder shim: {self._setup.embedder.name}")
        else:
            # Gemini embedder with TRACKING
            from graphiti_core.embedder.gemini import (
                GeminiEmbedder,
                GeminiEmbedderConfig,
            )

            # Tracking wrapper for embedder
            class TrackingGeminiEmbedder(GeminiEmbedder):
                async def create(self, input_data):
                    result = await super().create(input_data)

                    # Track cost
                    try:
                        if isinstance(input_data, str):
                            total_chars = len(input_data)
                        else:
                            total_chars = sum(len(str(s)) for s in input_data)

                        # Estimate tokens (4 chars per token)
                        input_tokens = total_chars // 4

                        # Gemini Embedding: $0.15/1M tokens (input only)
                        cost = (input_tokens * 0.15) / 1_000_000

                        _cost_tracker._save_entry(
                            model="gemini-embedding-001",
                            input_tokens=input_tokens,
                            output_tokens=0,
                            cost=cost,
                            call_type="embedding",
                        )

                        # Also update in-memory totals
                        async with _cost_tracker._lock:
                            _cost_tracker.total_input_tokens += input_tokens
                            _cost_tracker.total_cost_usd += cost
                            _cost_tracker.model_stats["gemini-embedding-001"][
                                "input_tokens"
                            ] += input_tokens
                            _cost_tracker.model_stats["gemini-embedding-001"][
                                "cost_usd"
                            ] += cost
                            _cost_tracker.model_stats["gemini-embedding-001"][
                                "call_count"
                            ] += 1
                    except Exception as e:
                        logger.debug(f"Embedding cost tracking failed: {e}")

                    return result

            embed_config = GeminiEmbedderConfig(
                api_key=config.gemini.api_key,
                embedding_model=config.gemini.embedding_model,
            )
            embedder = TrackingGeminiEmbedder(config=embed_config)
            logger.info("  Using Gemini API embedder (with cost tracking)")

        # Create Graphiti instance
        self._graphiti = Graphiti(
            graph_driver=driver, llm_client=llm_client, embedder=embedder
        )

        # Build indices
        await self._graphiti.build_indices_and_constraints()

        logger.info(f"✅ Initialized {self._setup.name}")

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load conversation dataset"""
        logger.info(f"Loading dataset from {DATASET_PATH}...")

        with open(DATASET_PATH) as f:
            data = json.load(f)

        sessions = data["sessions"]
        total_turns = sum(len(s["turns"]) for s in sessions)

        logger.info(f"Loaded {len(sessions)} sessions ({total_turns} total turns)")
        return sessions

    def _session_to_text(self, session: Dict[str, Any]) -> str:
        """
        Convert session to single text for Graphiti episode.

        Format:
        [Session 1 - 2024-01-15]
        [User]: Halo, apa kabar?
        [Assistant]: Baik, ada yang bisa saya bantu?
        ...
        """
        lines = []

        # Header with session info
        session_id = session["session_id"]
        first_turn = session["turns"][0] if session["turns"] else {}
        timestamp = first_turn.get("timestamp", "")[:10]  # Just date
        lines.append(f"[Session {session_id} - {timestamp}]")

        # All turns
        for turn in session["turns"]:
            speaker = turn["speaker"].capitalize()
            text = turn["text"]
            lines.append(f"[{speaker}]: {text}")

        return "\n".join(lines)

    def _get_checkpoint_path(self) -> Path:
        """Get checkpoint file path"""
        return CHECKPOINT_DIR / f"agentic_{self.setup_name}_checkpoint.json"

    def _load_checkpoint(self) -> Optional[IngestionCheckpoint]:
        """Load checkpoint if exists"""
        path = self._get_checkpoint_path()
        if path.exists():
            try:
                return IngestionCheckpoint.load(path)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return None

    def _save_checkpoint(self, checkpoint: IngestionCheckpoint):
        """Save checkpoint"""
        checkpoint.updated_at = datetime.now().isoformat()
        checkpoint.save(self._get_checkpoint_path())

    async def _add_episode_with_retry(
        self,
        content: str,
        source: str,
        reference_time: datetime,
        max_retries: int = 10,
        timeout_minutes: int = 10,
    ) -> bool:
        """
        Add episode dengan retry untuk rate limit.

        Untuk Gemini API: retry dengan backoff
        Untuk Gemma (local embedding): tetap perlu retry untuk LLM extraction

        Includes:
        - Timeout detection (default 10 minutes)
        - Heartbeat logging every 60 seconds
        """
        from graphiti_core.nodes import EpisodeType
        import asyncio

        async def _heartbeat_logger(session_name: str, interval: int = 60):
            """Log heartbeat every interval seconds"""
            elapsed = 0
            while True:
                await asyncio.sleep(interval)
                elapsed += interval
                mins = elapsed // 60
                logger.warning(
                    f"⏳ Still processing {session_name}... ({mins} min elapsed)"
                )
                if elapsed >= timeout_minutes * 60 - interval:
                    logger.error(
                        f"🚨 {session_name} taking too long! (>{timeout_minutes} min). May be stuck."
                    )

        for attempt in range(max_retries):
            try:
                # Create heartbeat task
                heartbeat_task = asyncio.create_task(_heartbeat_logger(source))

                try:
                    # Run with timeout
                    await asyncio.wait_for(
                        self._graphiti.add_episode(  # type: ignore[possibly-missing-attribute]
                            name=source,
                            episode_body=content,
                            source_description=source,
                            source=EpisodeType.message,
                            reference_time=reference_time,
                            group_id=self._group_id,
                        ),
                        timeout=timeout_minutes * 60,  # Convert to seconds
                    )
                finally:
                    # Cancel heartbeat
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

                self.rate_limiter.reset()  # Success, reset backoff
                return True

            except asyncio.TimeoutError:
                logger.error(
                    f"⏰ TIMEOUT: {source} exceeded {timeout_minutes} minutes!"
                )
                logger.error(
                    "   This usually means Neo4j connection died or Gemini API is not responding."
                )
                logger.error(f"   Attempt {attempt + 1}/{max_retries} - will retry...")
                # Continue to retry
                continue

            except Exception as e:
                error_str = str(e).lower()

                # Check if it's a rate limit error
                if any(
                    x in error_str
                    for x in ["rate", "limit", "429", "quota", "resource_exhausted"]
                ):
                    await self.rate_limiter.wait_on_rate_limit(str(e))
                    continue  # Retry
                else:
                    # Non-rate-limit error
                    logger.error(f"Non-rate-limit error: {e}")
                    raise

        raise Exception(
            f"Failed to add episode after {max_retries} retries due to rate limits/timeouts"
        )

    async def ingest(
        self, resume: bool = False, batch_size: int = 5, limit: Optional[int] = None
    ):
        """
        Ingest dataset into Graphiti.

        Args:
            resume: Resume from checkpoint
            batch_size: Process N episodes before pausing (for memory/safety)
            limit: Limit total episodes to process
        """
        # Load dataset
        sessions = self._load_dataset()

        # Limit sessions if requested
        if limit:
            sessions = sessions[:limit]
            logger.info(f"Limiting to {limit} sessions for testing.")

        total_sessions = len(sessions)

        # Check for existing checkpoint
        checkpoint = None
        start_index = 0

        if resume:
            checkpoint = self._load_checkpoint()
            # Resume from both 'in_progress', 'failed', AND 'batch_done' status
            if checkpoint and checkpoint.status in [
                "in_progress",
                "failed",
                "batch_done",
            ]:
                # Find the index to resume from
                for i, session in enumerate(sessions):
                    if session["session_id"] == checkpoint.last_session_id:
                        start_index = i + 1
                        break

                logger.info("📥 Resuming from checkpoint:")
                logger.info(
                    f"   Processed: {checkpoint.processed_sessions}/{checkpoint.total_sessions}"
                )
                logger.info(f"   Last session: {checkpoint.last_session_id}")
                logger.info(f"   Starting from index: {start_index}")
            elif checkpoint and checkpoint.status == "completed":
                logger.info(f"✅ Ingestion already completed for {self.setup_name}")
                return

        # Create new checkpoint if not resuming
        if checkpoint is None:
            checkpoint = IngestionCheckpoint(
                setup_name=self.setup_name,
                total_sessions=total_sessions,
                processed_sessions=0,
                last_session_id=0,
                started_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                status="in_progress",
            )

        # Calculate end index based on batch_size
        if batch_size:
            end_index = min(start_index + batch_size, total_sessions)
            batch_sessions = end_index - start_index
        else:
            end_index = total_sessions
            batch_sessions = total_sessions - start_index

        # Process sessions
        logger.info(f"\n{'=' * 60}")
        logger.info(f"INGESTING TO {self._setup.name.upper()}")  # type: ignore[possibly-missing-attribute]
        logger.info(f"{'=' * 60}")
        logger.info(f"Total sessions: {total_sessions}")
        logger.info(f"Starting from: {start_index}")
        if batch_size:
            logger.info(
                f"📦 BATCH: Processing {batch_sessions} sessions (until index {end_index - 1})"
            )
        logger.info(f"Group ID: {self._group_id}")
        logger.info(f"LLM Extraction: {self._setup.llm_extraction.name}")  # type: ignore[possibly-missing-attribute]
        logger.info(f"Embedder: {self._setup.embedder.name}")  # type: ignore[possibly-missing-attribute]
        logger.info("Unit: PER-SESSION (1 episode = 1 session)")
        logger.info(f"{'=' * 60}\n")

        pbar = tqdm(
            total=end_index, initial=start_index, desc=f"Ingesting ({self.setup_name})"
        )

        try:
            for i in range(start_index, end_index):
                session = sessions[i]
                session_id = session["session_id"]

                # Convert session to text
                session_text = self._session_to_text(session)

                # Parse timestamp from first turn
                first_turn = session["turns"][0] if session["turns"] else {}
                timestamp_str = first_turn.get("timestamp", "")
                try:
                    reference_time = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                except Exception:
                    reference_time = datetime.now()

                # Source identifier
                source = f"session_{session_id}"

                # PROACTIVE THROTTLE: Pace ourselves to stay within rate limits
                await self.rate_limiter.proactive_throttle()

                # Add episode with retry
                try:
                    await self._add_episode_with_retry(
                        content=session_text,
                        source=source,
                        reference_time=reference_time,
                    )
                    # Reset error counter on success
                    self.rate_limiter.reset()
                except Exception as e:
                    logger.error(f"❌ Failed to add session {session_id}: {e}")
                    checkpoint.status = "failed"
                    checkpoint.error_message = str(e)
                    self._save_checkpoint(checkpoint)
                    raise

                # Update checkpoint
                checkpoint.processed_sessions = i + 1
                checkpoint.last_session_id = session_id

                # Save checkpoint after EVERY session (not just batch)
                # This ensures proper resume on crash/interrupt
                self._save_checkpoint(checkpoint)

                # Update cost estimate in progress bar
                pbar.set_description(
                    f"Ingesting ({self.setup_name}) {get_cost_tracker().get_summary()}"
                )
                pbar.update(1)

            # Check if batch or full completion
            if end_index >= total_sessions:
                # Full completion
                checkpoint.status = "completed"
                self._save_checkpoint(checkpoint)

                pbar.close()

                # Get final stats
                stats = await self._get_stats()

                logger.info(f"\n{'=' * 60}")
                logger.info("✅ INGESTION COMPLETED!")
                logger.info(f"{'=' * 60}")
                logger.info(f"Setup: {self._setup.name}")  # type: ignore[possibly-missing-attribute]
                logger.info(f"Group ID: {self._group_id}")
                logger.info(f"Entities: {stats.get('entities', 0)}")
                logger.info(f"Facts: {stats.get('facts', 0)}")
                logger.info(f"Episodes: {stats.get('episodes', 0)}")
                logger.info(f"Facts: {stats.get('facts', 0)}")
                logger.info(f"Episodes: {stats.get('episodes', 0)}")
                logger.info(
                    f"Proactive throttles: {self.rate_limiter.total_proactive_waits}"
                )
                logger.info(
                    f"Rate limit hits: {self.rate_limiter.total_rate_limit_waits}"
                )
                logger.info(
                    f"FINAL COST ESTIMATE:\n{get_cost_tracker().get_detailed_summary()}"
                )
                logger.info(f"{'=' * 60}\n")
            else:
                # Batch done, more to process
                checkpoint.status = "batch_done"
                self._save_checkpoint(checkpoint)

                pbar.close()

                remaining = total_sessions - end_index

                # Get stats so far
                stats = await self._get_stats()

                logger.info(f"\n{'=' * 60}")
                logger.info("📦 BATCH COMPLETE!")
                logger.info(f"{'=' * 60}")
                logger.info(
                    f"Processed: {checkpoint.processed_sessions}/{total_sessions} sessions"
                )
                logger.info(f"Remaining: {remaining} sessions")
                logger.info(f"Entities so far: {stats.get('entities', 0)}")
                logger.info(f"Facts so far: {stats.get('facts', 0)}")
                logger.info(f"Facts so far: {stats.get('facts', 0)}")
                logger.info(
                    f"Proactive throttles: {self.rate_limiter.total_proactive_waits}"
                )
                logger.info(
                    f"Rate limit hits: {self.rate_limiter.total_rate_limit_waits}"
                )
                logger.info(f"COST SO FAR: {get_cost_tracker().get_summary()}")
                logger.info(f"{'=' * 60}")
                logger.info("\n🔄 To continue, run:")
                logger.info(
                    f"   python scripts/ingest_agentic.py --setup {self.setup_name} --resume --batch {batch_size or 5}"
                )
                logger.info(f"{'=' * 60}\n")

        except KeyboardInterrupt:
            logger.warning("\n⚠️ Ingestion interrupted by user")
            self._save_checkpoint(checkpoint)
            logger.info("Checkpoint saved. Resume with --resume flag.")
            raise

        except Exception as e:
            logger.error(f"\n❌ Ingestion failed: {e}")
            self._save_checkpoint(checkpoint)
            raise

    async def _get_stats(self) -> Dict[str, int]:
        """Get stats for the group"""
        try:
            from src.config.settings import get_config

            config = get_config()

            from graphiti_core.driver.neo4j_driver import Neo4jDriver

            driver = Neo4jDriver(
                uri=config.neo4j.uri,
                user=config.neo4j.user,
                password=config.neo4j.password,
                database=config.neo4j.database,
            )

            gid = self._group_id.replace("'", "\\'")  # type: ignore[possibly-missing-attribute]

            # Count entities
            entity_result = await driver.execute_query(
                f"MATCH (e:Entity {{group_id: '{gid}'}}) RETURN count(e) as count"  # type: ignore[invalid-argument-type]
            )

            # Count facts
            edge_result = await driver.execute_query(
                f"MATCH ()-[r:RELATES_TO {{group_id: '{gid}'}}]->() RETURN count(r) as count"  # type: ignore[invalid-argument-type]
            )

            # Count episodes
            episode_result = await driver.execute_query(
                f"MATCH (e:Episodic {{group_id: '{gid}'}}) RETURN count(e) as count"  # type: ignore[invalid-argument-type]
            )

            await driver.close()

            def extract_count(result):
                """Extract count from Neo4j EagerResult"""
                try:
                    # EagerResult has .records attribute
                    if hasattr(result, "records") and len(result.records) > 0:
                        record = result.records[0]
                        # Record can be accessed by key
                        return (
                            record.get("count", 0)
                            if hasattr(record, "get")
                            else record["count"]
                        )
                    # Fallback for list of dicts
                    elif isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict):
                            return result[0].get("count", 0)
                except Exception as e:
                    logger.debug(f"Error extracting count: {e}")
                return 0

            return {
                "entities": extract_count(entity_result),
                "facts": extract_count(edge_result),
                "episodes": extract_count(episode_result),
            }
        except Exception as e:
            logger.warning(f"Failed to get stats: {e}")
            return {"entities": 0, "facts": 0, "episodes": 0}

    async def close(self):
        """Close connections"""
        if self._graphiti:
            await self._graphiti.close()


async def ingest_setup(
    setup_name: str,
    resume: bool = False,
    batch_size: int = 5,
    limit: Optional[int] = None,
):
    """Run ingestion for a specific setup."""
    ingester = AgenticIngester(setup_name)
    try:
        await ingester.initialize()
        await ingester.ingest(resume=resume, batch_size=batch_size, limit=limit)
    finally:
        await ingester.close()


async def clear_group(group_id: str):
    """Clear Neo4j group"""
    from src.config.settings import get_config
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    config = get_config()
    driver = Neo4jDriver(
        uri=config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password,
        database=config.neo4j.database,
    )

    try:
        gid = group_id.replace("'", "\\'")
        await driver.execute_query(
            f"MATCH (n {{group_id: '{gid}'}}) DETACH DELETE n"  # type: ignore[invalid-argument-type]
        )
        logger.info(f"Cleared Neo4j group: {group_id}")
    finally:
        await driver.close()


async def main():
    parser = argparse.ArgumentParser(description="Ingest data for Agentic RAG")
    parser.add_argument(
        "--setup",
        choices=["gemini", "gemma", "all"],
        required=True,
        help="Which setup to ingest",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing data before ingesting"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=5,  # Changed default from None to 5 as per user's implied change
        help="Process only N sessions per run (for safer incremental ingestion)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of episodes (for testing)"
    )

    args = parser.parse_args()

    # Show batch mode info
    if args.batch:
        logger.info(f"\n📦 BATCH MODE: Processing {args.batch} sessions per run")
        logger.info("   Use --resume to continue from last checkpoint\n")

    # Clear if requested
    if args.clear:
        from src.config.experiment_setups import NEO4J_GROUP_IDS, SetupType

        if args.setup in ["gemini", "all"]:
            await clear_group(NEO4J_GROUP_IDS[SetupType.AGENTIC_GEMINI])

        if args.setup in ["gemma", "all"]:
            await clear_group(NEO4J_GROUP_IDS[SetupType.AGENTIC_GEMMA])

    # Ingest
    if args.setup == "all":
        # Sequential untuk fairness
        logger.info("\n" + "=" * 60)
        logger.info("SEQUENTIAL INGESTION: GEMINI FIRST, THEN GEMMA")
        logger.info("=" * 60 + "\n")

        await ingest_setup(
            "gemini", resume=args.resume, batch_size=args.batch, limit=args.limit
        )
        await ingest_setup(
            "gemma", resume=args.resume, batch_size=args.batch, limit=args.limit
        )
    else:
        await ingest_setup(
            args.setup, resume=args.resume, batch_size=args.batch, limit=args.limit
        )

    # Show final status
    logger.info("\n" + "=" * 60)
    logger.info("FINAL DATABASE STATUS")
    logger.info("=" * 60)

    # Show final status (manage_database.py was deleted, no longer needed)
    logger.info(
        "Use 'python scripts/clear_database.py' to view/manage database status."
    )


if __name__ == "__main__":
    asyncio.run(main())
