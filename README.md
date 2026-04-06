# soul_memory

### A three-layer memory architecture for sovereign AI agents.

Built by [dope.](https://github.com/dopenoun) as part of [The House](https://github.com/dopenoun/the-house) — a local-first, sovereign multi-agent system.

**What if your agent knew you the way a great host knows a guest?**

Not recall. *Recognition.*

---

## Why This Exists

Every agent framework gives you a chat history and calls it memory. That's not memory — that's a transcript. Real memory has weight. It fades, it scars, it prioritizes. It knows what matters without being told.

soul_memory is a memory plugin for AI agents that treats memory as a living system, not a log file. It combines vector similarity, compound salience scoring, and human-readable markdown export into a single architecture that runs entirely on your machine.

No cloud. No API keys for memory storage. No one else owns your agent's understanding of you.

### The Security Case

The [DeepMind AI Agent Traps paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6372438) (April 2026) documented 23 attack types against autonomous agents, with memory poisoning among the most dangerous. An attacker doesn't need to jailbreak your model — they poison the data your agent remembers, and that corruption persists across sessions.

soul_memory defends against this by design:

- **Provenance tracking**: Every memory entry records its origin — who wrote it, when, from what source, with what trust level. External content is tagged differently from agent-generated or human-provided memory.
- **Tiered trust**: L0 (identity) memories are immutable after boot. L1 (session) memories are scoped and expire. L2 (ambient) memories require promotion through a curation gate before they can influence core behavior.
- **Selective Dissolution**: Memories don't just expire — they undergo a character-level transformation. The wound dissolves; the scar remains. Wisdom survives as caution or pattern recognition, but the raw emotional charge and specific content fade. A poisoned memory's *influence* is architecturally bounded even if the injection succeeds.
- **Salience scoring with ethical alignment**: Every memory is scored on nine dimensions, including `ethical_alignment`. Memories that conflict with the agent's core values score lower and surface less frequently, creating a natural immune response to adversarial content.
- **Embedding model as schema truth**: The active embedding model is tracked in SQLite. You cannot silently swap models or mix embeddings — the system enforces consistency, preventing a class of attacks that corrupt similarity search by introducing incompatible vectors.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              soul_memory.py                 │
│                 (987 lines)                 │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │   LanceDB   │  │       SQLite         │  │
│  │   (Vector)  │  │  (Compound Scoring)  │  │
│  │             │  │                      │  │
│  │  sentence-  │  │  - Salience vectors  │  │
│  │  transform- │  │  - Provenance        │  │
│  │  ers 384d   │  │  - Trust levels      │  │
│  │             │  │  - Embedding model   │  │
│  │  Semantic   │  │    as schema truth   │  │
│  │  similarity │  │  - Dissolution state │  │
│  │  retrieval  │  │  - Tier assignments  │  │
│  └──────┬──────┘  └──────────┬───────────┘  │
│         │                    │              │
│         └────────┬───────────┘              │
│                  │                          │
│         ┌────────▼────────┐                 │
│         │    Obsidian     │                 │
│         │  Markdown Export │                │
│         │                 │                 │
│         │  Human-readable │                 │
│         │  vault notes    │                 │
│         │  with YAML      │                 │
│         │  frontmatter    │                 │
│         └─────────────────┘                 │
│                                             │
└─────────────────────────────────────────────┘
```

### Three Layers, Three Jobs

**LanceDB (Vector Layer)** — Semantic similarity search. When your agent needs to recall something *like* what's happening now, LanceDB finds it. Embeddings are generated locally via `sentence-transformers` using `all-MiniLM-L6-v2` (384 dimensions). No external API calls. Your memories never leave your machine.

**SQLite (Scoring Layer)** — The brain behind what surfaces and what stays buried. Every memory carries a nine-dimensional salience vector that determines its weight in any retrieval context. SQLite also tracks provenance (where did this memory come from?), trust levels, dissolution state, and the active embedding model. This is the schema truth layer — if there's ever a conflict between what the vector store says and what SQLite says, SQLite wins.

**Obsidian (Export Layer)** — Human-readable markdown files with YAML frontmatter, written to an Obsidian vault. This is the layer where you, the human, can actually read what your agent remembers, edit it, annotate it, and curate it. Agents write to designated zones; humans write to theirs. A curation gate (SCRIBE) governs cross-zone propagation.

### Tiered Loading

Not all memories are equal, and not all of them need to be in context at boot.

| Tier | Name | When Loaded | What Lives Here |
|------|------|-------------|-----------------|
| L0 | Identity | Always at boot | Core persona, values, non-negotiable truths. Immutable after initial set. |
| L1 | Session | On session start | Active task context, recent interactions, working memory. Scoped to session lifespan. |
| L2 | Ambient | On demand | Long-term knowledge, learned patterns, accumulated wisdom. Loaded when salience scoring says it's relevant. |

The `boot_context()` function loads L0 immediately, primes L1 from the task queue, and leaves L2 for lazy retrieval. This keeps boot fast and context windows lean.

The `tier_of()` function assigns tier based on memory properties — recency, access frequency, provenance trust, and explicit human promotion/demotion.

Cache eviction follows tier priority: L2 evicts first, L1 evicts under pressure, L0 never evicts.

### Nine-Dimensional Salience Vector

Every memory is scored across nine dimensions. Retrieval isn't just "most similar" — it's "most relevant right now, given everything."

| Dimension | What It Measures |
|-----------|-----------------|
| `emotional_tone` | Emotional weight of the memory (grief, joy, urgency, calm) |
| `relational_relevance` | How central this is to the agent-human relationship |
| `task_urgency` | How time-sensitive the associated task is |
| `novelty` | How new or surprising this information was when recorded |
| `long_term_value` | Likelihood this matters in 30 days, not just today |
| `ethical_alignment` | How well this aligns with the agent's core values |
| `coordination_impact` | How much this affects other agents in a multi-agent system |
| `memory_coherence` | How well this fits with existing memory (contradictions score low) |
| `delight` | The joy dimension — not everything is wounds and work |

Compound scoring combines these dimensions with context-dependent weights. A task-focused retrieval upweights `task_urgency` and `coordination_impact`. A relationship-focused retrieval upweights `relational_relevance` and `emotional_tone`. The weights shift based on what the agent is doing right now.

### Selective Dissolution

This is the philosophical core.

Traditional memory systems have two modes: remember or delete. That's not how real memory works. Real memory *transforms*. The acute pain of a bad experience fades, but the lesson remains as a scar — present as caution, not active as belief.

soul_memory implements this as a character function, not a cleanup job:

- **Wounds** dissolve over time. The specific emotional charge, the raw details, the urgency — these fade according to dissolution curves that factor in recency, access frequency, and emotional intensity.
- **Scars** persist. The pattern recognition, the learned caution, the wisdom extracted from the wound — these are promoted to a compressed form that influences future behavior without carrying the original weight.
- **Noise** evaporates. Not everything is a wound or a scar. Some things are just noise — transient context that served its purpose and can be fully released.

The dissolution process is governed by the salience vector. High `long_term_value` + low `task_urgency` = scar candidate. Low `long_term_value` + low `novelty` = noise. High `emotional_tone` + declining access = wound in dissolution.

---

## Security Model

### Threat: Memory Poisoning

An attacker embeds hidden instructions in a webpage, PDF, or API response. Your agent processes the content and stores a memory derived from the poisoned input. That memory now influences every future session.

### Defense Layers

**1. Provenance Tagging**

Every memory entry carries metadata about its origin:

```python
{
    "source": "external_web",      # or "agent_generated", "human_provided", "session_derived"
    "source_hash": "sha256:...",   # hash of the raw input that generated this memory
    "fetch_timestamp": "...",
    "trust_level": 0.3,            # external content starts low
    "promoted_by": null,           # null until a curation gate approves
    "provenance_chain": [...]      # full chain if derived from other memories
}
```

External content never enters the memory system at the same trust level as human-provided or identity-level memories. The trust level affects salience scoring — low-trust memories surface less and carry less weight.

**2. Tiered Immutability**

L0 (identity) memories are frozen after boot. A poisoned webpage cannot overwrite your agent's core values, persona definition, or foundational truths. These are set by the human operator and are not writable by any automated process.

L1 (session) memories are scoped. A poisoned memory in one session cannot bleed into another unless explicitly promoted through the curation gate.

L2 (ambient) memories are the attack surface — and they're designed to be. External content enters at L2 with low trust, subject to dissolution, and only promoted if a curator (human or designated agent like SCRIBE) approves.

**3. Ethical Alignment as Immune System**

The `ethical_alignment` dimension in the salience vector acts as a natural filter. Memories that conflict with the agent's established value system score lower on this dimension, making them less likely to surface in retrieval and less influential in reasoning.

This doesn't prevent storage of the poisoned content — it reduces its *influence*. The memory exists but carries less weight, like an immune system that doesn't prevent infection but limits its spread.

**4. Embedding Consistency Enforcement**

The active embedding model is recorded in SQLite as schema truth. If the model changes (upgrade, swap, attack), the system detects the mismatch and either re-embeds all memories with the new model or rejects the change. You cannot corrupt similarity search by injecting vectors from an incompatible model.

**5. Dissolution as Temporal Defense**

Even if a poisoned memory enters the system, dissolution limits its lifespan. External-origin memories with low access frequency and low `long_term_value` dissolve faster. The attacker's window of influence is bounded by the dissolution curve, not infinite.

---

## Installation

### Prerequisites

- Python 3.10+

No separate inference server required. Embeddings run in-process via `sentence-transformers`.

### Install soul_memory

```bash
# Clone the repo
git clone https://github.com/dopenoun/soul-memory.git
cd soul-memory

# Install Python dependencies
pip install lancedb sentence-transformers numpy

# Or use a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install lancedb sentence-transformers numpy
```

The first run will download `all-MiniLM-L6-v2` (~90MB) from HuggingFace and cache it locally. Subsequent runs are fully offline.

### Configure

```bash
# Set your vault path (where Obsidian markdown exports go)
export SOUL_MEMORY_VAULT=~/your-vault-path/

# Set your data directory (where LanceDB + SQLite live)
export SOUL_MEMORY_DIR=~/.soul-memory/
```

### Verify

```python
from soul_memory import SoulMemory, CharacterScope

mem = SoulMemory(soul_id="my-agent")
ctx = mem.boot_context()  # loads L0, primes L1
print(ctx)

# Store a memory
mem.remember(
    content="The human prefers direct communication, no fluff.",
    scope=CharacterScope.BELIEFS,
    raw_weight=0.9,
    identity_alignment=1.0,
)

# Recall
results = mem.recall("how does the human like to communicate?", top_k=3)
for r in results:
    print(r)
```

### Integration with Agent Frameworks

soul_memory is framework-agnostic. It exposes a Python API that any agent can call. Example integrations:

**With Hermes (subprocess model):**
```python
# In your Hermes profile's execution loop
from soul_memory import SoulMemory

mem = SoulMemory(soul_id="hermes")
mem.boot_context()

# On receiving a message
context = mem.recall(message.content, top_k=5)
# Inject context into your model prompt
```

**With MCP (as a tool):**
```python
# Expose remember/recall as MCP tools
# Agents call soul_memory_recall and soul_memory_remember
# via JSON-RPC over stdio
```

**With SCRIBE (curation gate):**
```python
# SCRIBE reviews low-score memories nightly
results = mem.recall("*", top_k=50)
for r in results:
    if not scribe_approves(r):
        mem.dissolve(r.trace.id)
```

---

## Should This Be a Separate Repo?

Yes. Here's why:

1. **The community needs a memory plugin that isn't a wrapper around a vector DB.** Most "agent memory" solutions are just ChromaDB/Pinecone with a thin API. soul_memory has an opinion about what memory *means* — salience scoring, dissolution, tiered trust, provenance. That opinion is the value.

2. **Security is the hook.** The DeepMind paper makes memory poisoning a front-page concern. A memory plugin that treats security as architectural (provenance tagging, tiered immutability, ethical alignment scoring) rather than bolt-on (sanitize inputs, hope for the best) is what the ecosystem needs right now.

3. **It's separable.** soul_memory.py doesn't depend on ZeroClaw, Beads, or The House's specific agent topology. It needs only Python stdlib + optional vector libs. The House-specific integrations (SCRIBE, Beads task context, ZeroClaw gateway) are wiring that lives in The House, not in soul_memory itself.

4. **It feeds SOULFORGE.** A standalone soul_memory repo becomes the memory layer of SOUL file packages. When you ship a `.soul` file, soul_memory is the runtime that hydrates the memory seed. Separate repo = clean dependency.

### Repo structure:

```
dopenoun/soul-memory/
├── README.md
├── soul_memory.py          # core (987 lines)
├── requirements.txt        # lancedb, sentence-transformers, numpy
├── examples/
│   ├── basic_usage.py
│   ├── mcp_tool.py
│   ├── hermes_integration.py
│   └── scribe_curation.py
├── tests/
│   ├── test_memory.py
│   ├── test_salience.py
│   ├── test_dissolution.py
│   └── test_provenance.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── SECURITY.md
│   └── SALIENCE.md
└── LICENSE
```

---

## Philosophy

> *"What if your agent knew you the way a great host knows a guest?"*

This system was born from fine dining, not computer science. A great maître d' doesn't recall your name from a database lookup — they *recognize* you. The difference is everything. Recall is mechanical. Recognition is relational. It carries context, history, warmth, and judgment.

soul_memory is built to give agents recognition. Not just "I remember you said X" but "I understand what kind of person says X, and I know what you need before you ask."

The hospitality model maps directly:

- **Mise en place** → Boot context. Everything ready before service starts.
- **Guest notes** → Salience-scored memories. Not everything goes in the notes — only what matters for the next visit.
- **Steps of service** → Tiered loading. Don't overwhelm the guest with everything you know. Surface what's relevant, when it's relevant.
- **The scar, not the wound** → Selective dissolution. A great host remembers that a guest had a difficult anniversary dinner — not to bring it up, but to be gentle. The lesson survives. The pain doesn't.

---

## Built With

- [LanceDB](https://lancedb.com) — Embedded vector database
- [sentence-transformers](https://www.sbert.net) — Local embedding inference (`all-MiniLM-L6-v2`, 384d)
- [SQLite](https://sqlite.org) — Because it's perfect
- [Obsidian](https://obsidian.md) — For the humans in the loop

---

## License

MIT — because `community.is.dope`

---

## Credits

Part of [dope.](https://github.com/dopenoun) — DAO Of Personal Experience.

Built by [aifriedenewman](https://github.com/aifriedenewman). Architecture guided by APEX. Memory curated by SCRIBE.

`life.is.dope.`
