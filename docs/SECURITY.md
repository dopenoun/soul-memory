# Security Model

See the Security Model section in the main README for the full threat model and defense layers.

## Quick Reference

| Defense | Mechanism |
|---------|-----------|
| Provenance tagging | Every memory records source, timestamp, trust_level, source_hash |
| Tiered immutability | L0 frozen at boot; L1 session-scoped; L2 requires curation gate |
| Ethical alignment filter | Low-alignment memories score lower, surface less |
| Embedding consistency | Active model tracked in SQLite — silent swaps rejected |
| Dissolution bounds | External low-trust memories dissolve faster, limiting attack window |

## Trust Levels

| Source | Default Trust |
|--------|--------------|
| `human_provided` | 1.0 |
| `agent_generated` | 0.8 |
| `session_derived` | 0.6 |
| `external_web` | 0.3 |
