# Architecture

See the main README for the architecture overview and diagram.

## Data Flow

1. `remember(content, ...)` → embed via sentence-transformers → store vector in LanceDB → store metadata/salience in SQLite → optionally export to Obsidian vault
2. `recall(query, ...)` → embed query → vector search in LanceDB → rank candidates by compound salience score → return top_k

## Storage Layout

```
~/.soul-memory/           # default SOUL_MEMORY_DIR
├── lancedb/              # vector store
└── soul.db               # SQLite — salience, provenance, tiers
```

## Embedding Model

`all-MiniLM-L6-v2` via `sentence-transformers`. 384 dimensions. Fully local — no inference server required. First run downloads the model (~90MB) from HuggingFace; subsequent runs are offline.

The active model name is stored in SQLite as schema truth. Changing the model triggers a re-embedding pass or rejection.
