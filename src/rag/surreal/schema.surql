-- Episodic ingest (per session)
DEFINE TABLE IF NOT EXISTS episode SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS group_id ON episode TYPE string;
DEFINE FIELD IF NOT EXISTS name ON episode TYPE string;
DEFINE FIELD IF NOT EXISTS body ON episode TYPE string;
DEFINE FIELD IF NOT EXISTS source_description ON episode TYPE string;
DEFINE FIELD IF NOT EXISTS reference_time ON episode TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS created_at ON episode TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_episode_group ON episode FIELDS group_id;

-- Extracted atomic facts (vector retrieval)
DEFINE TABLE IF NOT EXISTS extracted_fact SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS group_id ON extracted_fact TYPE string;
DEFINE FIELD IF NOT EXISTS episode_name ON extracted_fact TYPE string;
DEFINE FIELD IF NOT EXISTS fact_text ON extracted_fact TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON extracted_fact TYPE array<float>;
DEFINE FIELD IF NOT EXISTS entity_names ON extracted_fact TYPE array<string>;
DEFINE FIELD IF NOT EXISTS valid_at ON extracted_fact TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS created_at ON extracted_fact TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS source_description ON extracted_fact TYPE option<string>;
DEFINE INDEX IF NOT EXISTS idx_fact_group ON extracted_fact FIELDS group_id;
DEFINE INDEX IF NOT EXISTS idx_fact_embed ON extracted_fact FIELDS embedding MTREE DIMENSION 768 DIST COSINE;

-- Optional entity nodes (lightweight)
DEFINE TABLE IF NOT EXISTS entity SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS group_id ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS name ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS created_at ON entity TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_entity_group_name ON entity FIELDS group_id, name;

-- Vanilla RAG chunks (768-dim embeddings)
DEFINE TABLE IF NOT EXISTS vanilla_chunk SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS collection ON vanilla_chunk TYPE string;
DEFINE FIELD IF NOT EXISTS doc_id ON vanilla_chunk TYPE string;
DEFINE FIELD IF NOT EXISTS text ON vanilla_chunk TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON vanilla_chunk TYPE array<float>;
DEFINE FIELD IF NOT EXISTS metadata ON vanilla_chunk TYPE object;
DEFINE FIELD IF NOT EXISTS created_at ON vanilla_chunk TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_vanilla_coll ON vanilla_chunk FIELDS collection;
DEFINE INDEX IF NOT EXISTS idx_vanilla_embed ON vanilla_chunk FIELDS embedding MTREE DIMENSION 768 DIST COSINE;
