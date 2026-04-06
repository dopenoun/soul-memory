# Nine-Dimensional Salience Vector

See the main README for the full description of each dimension.

## Dimension Reference

| Dimension | Range | High Value Means |
|-----------|-------|-----------------|
| `emotional_tone` | 0–1 | Strong emotional charge |
| `relational_relevance` | 0–1 | Central to agent-human relationship |
| `task_urgency` | 0–1 | Time-sensitive |
| `novelty` | 0–1 | Surprising when recorded |
| `long_term_value` | 0–1 | Matters in 30+ days |
| `ethical_alignment` | 0–1 | Aligned with agent values |
| `coordination_impact` | 0–1 | Affects other agents |
| `memory_coherence` | 0–1 | Consistent with existing memory |
| `delight` | 0–1 | Joy-bearing |

## Compound Scoring

Retrieval weights are context-dependent. The system applies different weight profiles depending on the active retrieval mode (task-focused, relationship-focused, ambient scan).

Unspecified dimensions default to 0.5 at storage time.
