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

-- Entity nodes (canonical names per group); facts link via ``fact_involves`` for graph retrieval.
DEFINE TABLE IF NOT EXISTS entity SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS group_id ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS name ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS created_at ON entity TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_entity_group_name ON entity FIELDS group_id, name;

-- Graph edges (Explorer / SELECT ->has_fact-> …); see https://surrealdb.com/docs/surrealdb/models/graph
DEFINE TABLE IF NOT EXISTS has_fact TYPE RELATION IN episode OUT extracted_fact;
DEFINE TABLE IF NOT EXISTS fact_involves TYPE RELATION IN extracted_fact OUT entity;

-- Dense session retrieval (768-dim); logical slice via ``collection`` (e.g. vanilla_gemini)
DEFINE TABLE IF NOT EXISTS session_passage SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS collection ON session_passage TYPE string;
DEFINE FIELD IF NOT EXISTS doc_id ON session_passage TYPE string;
DEFINE FIELD IF NOT EXISTS text ON session_passage TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON session_passage TYPE array<float>;
DEFINE FIELD IF NOT EXISTS metadata ON session_passage TYPE object;
DEFINE FIELD IF NOT EXISTS created_at ON session_passage TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_session_passage_coll ON session_passage FIELDS collection;
DEFINE INDEX IF NOT EXISTS idx_session_passage_embed ON session_passage FIELDS embedding MTREE DIMENSION 768 DIST COSINE;
