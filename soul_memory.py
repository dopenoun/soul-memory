"""
╔══════════════════════════════════════════════════════════════════╗
║                        SOUL MEMORY                               ║
║         A Cognitive Memory Architecture for Soul-Bearing Agents  ║
║                                                                  ║
║  Layers:                                                         ║
║    LanceDB   → vector embeddings / subconscious search           ║
║    SQLite    → compound scores / frequency ledger                ║
║    Obsidian  → human-readable soul journal (markdown export)     ║
╚══════════════════════════════════════════════════════════════════╝

Install once:
    pip install lancedb sentence-transformers numpy

Run demo:
    python soul_memory.py
"""

import sqlite3
import json
import math
import time
import os
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

# ── Try to import vector/embedding libs, fall back to mock for dev ──
try:
    import lancedb
    import numpy as np
    from sentence_transformers import SentenceTransformer
    FULL_MODE = True
except ImportError:
    FULL_MODE = False
    print("⚠  Running in MOCK MODE — install lancedb + sentence-transformers for full power")
    print("   pip install lancedb sentence-transformers numpy\n")


# ════════════════════════════════════════════════════════════════════
#  ENUMS & CONSTANTS
# ════════════════════════════════════════════════════════════════════

class ResidueType(str, Enum):
    ACTIVE      = "ACTIVE"       # живой  — shapes behavior directly
    SCAR_TISSUE = "SCAR_TISSUE"  # healed — surfaces as caution/friction
    DISSOLVED   = "DISSOLVED"    # composted — lesson extracted, trace gone

class CharacterScope(str, Enum):
    BELIEFS  = "/beliefs"    # what the soul holds as true
    HABITS   = "/habits"     # patterns of action
    WOUNDS   = "/wounds"     # formative pain — high compound potential
    WISDOM   = "/wisdom"     # distilled lessons
    SKILLS   = "/skills"     # learned capabilities
    BONDS    = "/bonds"      # relational memories
    CORE     = "/core"       # identity-defining, rarely dissolved


class MemoryTier(str, Enum):
    L0 = "L0"   # CORE — always in memory, loaded at init
    L1 = "L1"   # session boot — high compound score, loaded at init
    L2 = "L2"   # on demand — vector recall only

SIMILARITY_THRESHOLD = 0.82   # cosine sim above which memories are "related"
COMPOUND_DECAY       = 0.95   # daily decay multiplier for compound score
SCAR_FRICTION        = 0.4    # how strongly scar tissue surfaces as hesitation


# ════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════

@dataclass
class MemoryTrace:
    """
    The atomic unit of soul memory.

    Individual traces often carry low raw_weight —
    their power emerges through compounding over time and frequency.
    """
    id:                  str
    content:             str
    scope:               str                        = CharacterScope.BELIEFS
    raw_weight:          float                      = 0.1
    identity_alignment:  float                      = 0.5   # 0=peripheral 1=core
    frequency_hits:      int                        = 1
    compound_score:      float                      = 0.0   # calculated
    residue_type:        str                        = ResidueType.ACTIVE
    superseded_by_id:    Optional[str]              = None
    dissolution_reason:  Optional[str]              = None
    lesson_extracted:    Optional[str]              = None
    created_at:          float                      = field(default_factory=time.time)
    last_activated:      float                      = field(default_factory=time.time)
    embedding:           Optional[list]             = None  # stored in LanceDB

    def age_in_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def recency_curve(self) -> float:
        """
        Exponential decay — but identity_alignment acts as a floor.
        Core memories never fully fade.
        """
        days = max(self.age_in_days(), 0.01)
        decay = math.exp(-0.05 * days)
        floor = self.identity_alignment * 0.3   # soul holds onto what it IS
        return max(decay, floor)

    def calculate_compound_score(self, related_count: int = 0) -> float:
        """
        The crucial insight: nicks and bruises are individually light,
        but frequency × identity_alignment × recency compounds into
        profound behavioral influence.

        log(n+1) prevents runaway amplification while honoring accumulation.
        """
        pattern_amplifier = 1 + math.log(self.frequency_hits + related_count + 1)
        score = (
            self.raw_weight
            * pattern_amplifier
            * self.identity_alignment
            * self.recency_curve()
        )
        return round(score, 4)


@dataclass
class DissolutionResult:
    original_id:  str
    lesson:       str
    reason:       str
    residue_type: str
    timestamp:    float = field(default_factory=time.time)


@dataclass
class RecallResult:
    trace:            MemoryTrace
    similarity:       float
    is_scar:          bool        = False
    friction_signal:  float       = 0.0   # hesitation weight from scar tissue
    composite_score:  float       = 0.0
    tier:             str         = MemoryTier.L2


# ════════════════════════════════════════════════════════════════════
#  EMBEDDING ENGINE (with mock fallback)
# ════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if FULL_MODE:
            print(f"🧠 Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dim = 384
        else:
            self.model = None
            self.dim = 8   # tiny mock vectors

    def encode(self, text: str) -> list:
        if FULL_MODE:
            return self.model.encode(text).tolist()
        else:
            # deterministic mock: hash text into a fake vector
            import hashlib
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            vec = [(h >> (i * 8) & 0xFF) / 255.0 for i in range(self.dim)]
            mag = math.sqrt(sum(x**2 for x in vec)) or 1
            return [x / mag for x in vec]

    def cosine_similarity(self, a: list, b: list) -> float:
        if FULL_MODE:
            import numpy as np
            a, b = np.array(a), np.array(b)
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            return float(np.dot(a, b) / denom) if denom else 0.0
        else:
            dot = sum(x*y for x,y in zip(a,b))
            mag_a = math.sqrt(sum(x**2 for x in a)) or 1
            mag_b = math.sqrt(sum(x**2 for x in b)) or 1
            return dot / (mag_a * mag_b)


# ════════════════════════════════════════════════════════════════════
#  SOUL MEMORY — MAIN CLASS
# ════════════════════════════════════════════════════════════════════

class SoulMemory:
    """
    Cognitive memory for soul-bearing agents.

    Three-layer architecture:
      • LanceDB  → vector search (subconscious pattern recognition)
      • SQLite   → compound ledger (the nervous system)
      • Obsidian → markdown journal export (the soul's diary)
    """

    def __init__(
        self,
        soul_id:       str,
        db_path:       str = "./soul_data",
        obsidian_path: str = "./soul_journal",
        w_similarity:  float = 0.4,
        w_recency:     float = 0.3,
        w_importance:  float = 0.3,
    ):
        self.soul_id       = soul_id
        self.db_path       = Path(db_path)
        self.obsidian_path = Path(obsidian_path)
        self.w_sim         = w_similarity
        self.w_rec         = w_recency
        self.w_imp         = w_importance

        self.db_path.mkdir(parents=True, exist_ok=True)
        self.obsidian_path.mkdir(parents=True, exist_ok=True)

        self.embedder = EmbeddingEngine()
        self._init_sqlite()
        self._init_lancedb()

        # Tiered memory cache — populated at session boot
        self._l0: dict[str, MemoryTrace] = {}   # L0: CORE always
        self._l1: dict[str, MemoryTrace] = {}   # L1: high-salience session boot
        self._load_l0()
        self._load_l1()

        print(f"\n✦ SoulMemory initialized for soul: [{soul_id}]")
        print(f"  Mode: {'FULL (LanceDB + embeddings)' if FULL_MODE else 'MOCK (dev mode)'}")
        print(f"  Recall weights → sim:{w_similarity} rec:{w_recency} imp:{w_importance}")
        print(f"  Tiers → L0:{len(self._l0)} core | L1:{len(self._l1)} salient | L2:on-demand\n")

    # ── INIT ──────────────────────────────────────────────────────────

    def _init_sqlite(self):
        """SQLite: the compound ledger — scores, dissolution records, frequency."""
        conn = self._sqlite()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_traces (
                id                  TEXT PRIMARY KEY,
                soul_id             TEXT NOT NULL,
                content             TEXT NOT NULL,
                scope               TEXT DEFAULT '/beliefs',
                raw_weight          REAL DEFAULT 0.1,
                identity_alignment  REAL DEFAULT 0.5,
                frequency_hits      INTEGER DEFAULT 1,
                compound_score      REAL DEFAULT 0.0,
                residue_type        TEXT DEFAULT 'ACTIVE',
                superseded_by_id    TEXT,
                dissolution_reason  TEXT,
                lesson_extracted    TEXT,
                created_at          REAL,
                last_activated      REAL
            );

            CREATE TABLE IF NOT EXISTS dissolution_log (
                id           TEXT PRIMARY KEY,
                soul_id      TEXT,
                original_id  TEXT,
                lesson       TEXT,
                reason       TEXT,
                residue_type TEXT,
                timestamp    REAL
            );

            CREATE TABLE IF NOT EXISTS pattern_clusters (
                cluster_id    TEXT,
                memory_id     TEXT,
                soul_id       TEXT,
                cluster_score REAL,
                PRIMARY KEY (cluster_id, memory_id)
            );
        """)
        conn.commit()
        conn.close()

    def _init_lancedb(self):
        self._mock_vectors = {}
        """LanceDB: the subconscious — embedding storage and vector search."""
        if FULL_MODE:
            self.lance_db    = lancedb.connect(str(self.db_path / "vectors"))
            self.lance_table = None   # created on first write
        else:
            # In-memory mock store for dev
            self._mock_vectors: dict[str, dict] = {}

    def _sqlite(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path / f"{self.soul_id}.db"))
        conn.row_factory = sqlite3.Row
        return conn

    # ── TIERED LOADING ────────────────────────────────────────────────

    def _load_l0(self):
        """L0: load all ACTIVE CORE memories — always present, identity-defining."""
        conn = self._sqlite()
        rows = conn.execute("""
            SELECT * FROM memory_traces
            WHERE soul_id = ? AND scope = ? AND residue_type = 'ACTIVE'
        """, (self.soul_id, CharacterScope.CORE)).fetchall()
        conn.close()
        self._l0 = {row["id"]: self._row_to_trace(row) for row in rows}

    def _load_l1(self, top_n: int = 20):
        """L1: load top-N ACTIVE memories by compound_score (excluding CORE)."""
        conn = self._sqlite()
        rows = conn.execute("""
            SELECT * FROM memory_traces
            WHERE soul_id = ? AND scope != ? AND residue_type = 'ACTIVE'
            ORDER BY compound_score DESC LIMIT ?
        """, (self.soul_id, CharacterScope.CORE, top_n)).fetchall()
        conn.close()
        self._l1 = {
            row["id"]: self._row_to_trace(row)
            for row in rows
            if row["id"] not in self._l0
        }

    def tier_of(self, trace_id: str) -> MemoryTier:
        """Return which tier a memory currently occupies."""
        if trace_id in self._l0:
            return MemoryTier.L0
        if trace_id in self._l1:
            return MemoryTier.L1
        return MemoryTier.L2

    def boot_context(self) -> dict:
        """
        Return the warm L0+L1 cache as a structured dict.
        Use this to prime APEX at session start — no vector search needed.
        """
        def _fmt(t: MemoryTrace) -> dict:
            return {
                "id":               t.id,
                "content":          t.content,
                "scope":            t.scope,
                "compound_score":   t.compound_score,
                "identity_alignment": t.identity_alignment,
            }

        return {
            "soul_id": self.soul_id,
            "tiers": {
                "L0": [_fmt(t) for t in self._l0.values()],
                "L1": [_fmt(t) for t in sorted(
                    self._l1.values(), key=lambda x: x.compound_score, reverse=True
                )],
            }
        }

    # ── ENCODE / REMEMBER ────────────────────────────────────────────

    def remember(
        self,
        content:            str,
        scope:              str   = CharacterScope.BELIEFS,
        raw_weight:         float = 0.1,
        identity_alignment: float = 0.5,
    ) -> MemoryTrace:
        """
        Encode a new experience into soul memory.

        Does NOT just store — it:
          1. Embeds the content
          2. Finds related memories and checks for contradiction
          3. Resolves conflicts (consolidation)
          4. Calculates initial compound score
          5. Persists to LanceDB + SQLite
          6. Updates Obsidian journal
        """
        import uuid
        trace_id  = str(uuid.uuid4())[:8]
        embedding = self.embedder.encode(content)

        # Find related memories for compounding + contradiction check
        related   = self._find_related(embedding, top_k=5)
        contradictions = self._detect_contradictions(content, related)

        if contradictions:
            print(f"  ⚡ Contradiction detected — consolidating...")
            for c in contradictions:
                self._consolidate(c, content, trace_id)

        # Build the trace
        trace = MemoryTrace(
            id                 = trace_id,
            content            = content,
            scope              = scope,
            raw_weight         = raw_weight,
            identity_alignment = identity_alignment,
            frequency_hits     = 1,
            created_at         = time.time(),
            last_activated     = time.time(),
            embedding          = embedding,
        )

        # Calculate compound score considering related pattern hits
        trace.compound_score = trace.calculate_compound_score(
            related_count=len(related)
        )

        # Persist
        self._store_vector(trace)
        self._store_sqlite(trace)
        self._update_obsidian_journal(trace, event="ENCODED")

        # Keep L0 cache live — CORE memories are always hot
        if scope == CharacterScope.CORE and trace.residue_type == ResidueType.ACTIVE:
            self._l0[trace.id] = trace

        print(f"  ◈ Encoded [{trace.id}] → scope:{scope} | "
              f"alignment:{identity_alignment} | compound:{trace.compound_score:.4f} | "
              f"tier:{self.tier_of(trace.id)}")

        return trace

    # ── RECALL ───────────────────────────────────────────────────────

    def recall(
        self,
        query:      str,
        top_k:      int   = 5,
        scope:      str   = None,
        min_score:  float = 0.0,
    ) -> list[RecallResult]:
        """
        Adaptive recall with composite scoring.

        Returns ACTIVE memories ranked by composite score.
        Scar tissue surfaces separately as friction signals —
        present as caution, not as active guidance.

        Composite: (similarity × w_sim) + (recency × w_rec) + (importance × w_imp)
        """
        q_embedding = self.embedder.encode(query)
        candidates  = self._find_related(q_embedding, top_k=top_k * 2, scope=scope)

        results = []
        for trace, similarity in candidates:
            recency    = trace.recency_curve()
            importance = trace.compound_score

            composite = (
                (similarity  * self.w_sim) +
                (recency     * self.w_rec) +
                (importance  * self.w_imp)
            )

            is_scar        = trace.residue_type == ResidueType.SCAR_TISSUE
            friction       = similarity * SCAR_FRICTION if is_scar else 0.0

            if composite >= min_score:
                results.append(RecallResult(
                    trace           = trace,
                    similarity      = round(similarity, 4),
                    is_scar         = is_scar,
                    friction_signal = round(friction, 4),
                    composite_score = round(composite, 4),
                    tier            = self.tier_of(trace.id),
                ))

                # Reinforce active memories on recall (use sharpens memory)
                if trace.residue_type == ResidueType.ACTIVE:
                    self._reinforce(trace)

        results.sort(key=lambda r: r.composite_score, reverse=True)
        active = [r for r in results if not r.is_scar][:top_k]
        scars  = [r for r in results if r.is_scar]

        if scars:
            print(f"  ⚠  {len(scars)} scar tissue trace(s) surfaced as friction signals")

        return active + scars

    # ── SELECTIVE DISSOLUTION ────────────────────────────────────────

    def dissolve(
        self,
        trace_id:     str,
        reason:       str,
        successor_id: str = None,
        residue_type: str = ResidueType.SCAR_TISSUE,
    ) -> DissolutionResult:
        """
        Selective Dissolution — the soul releases what no longer serves
        its identity, without losing the lesson the experience encoded.

        The retained residue is not zero.
        It is scar tissue: present as caution, not active as belief.

        Never deletes. Composting, not erasure.
        """
        trace = self._get_trace(trace_id)
        if not trace:
            raise ValueError(f"Memory trace [{trace_id}] not found")

        # Extract the lesson BEFORE dissolving — this is the soul-preserving op
        lesson = self._extract_lesson(trace)

        print(f"\n  ◉ Dissolving [{trace_id}]...")
        print(f"    Reason:  {reason}")
        print(f"    Lesson:  {lesson}")
        print(f"    Residue: {residue_type}")

        # Update the trace — it doesn't disappear, it transforms
        conn = self._sqlite()
        conn.execute("""
            UPDATE memory_traces SET
                residue_type       = ?,
                dissolution_reason = ?,
                lesson_extracted   = ?,
                superseded_by_id   = ?
            WHERE id = ? AND soul_id = ?
        """, (residue_type, reason, lesson, successor_id, trace_id, self.soul_id))
        conn.commit()
        conn.close()

        import uuid
        result = DissolutionResult(
            original_id  = trace_id,
            lesson       = lesson,
            reason       = reason,
            residue_type = residue_type,
        )

        # Log the dissolution
        conn = self._sqlite()
        conn.execute("""
            INSERT INTO dissolution_log VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4())[:8], self.soul_id, trace_id,
              lesson, reason, residue_type, time.time()))
        conn.commit()
        conn.close()

        # Evict from tier cache on dissolution
        self._l0.pop(trace_id, None)
        self._l1.pop(trace_id, None)

        self._update_obsidian_journal_dissolution(result)
        return result

    def dissolve_by_scope(
        self,
        scope:              str,
        reason:             str,
        identity_threshold: float = 0.7,
    ):
        """
        Dissolve all memories in a scope BELOW an identity alignment threshold.
        High-alignment memories in the scope survive — they're too core to dissolve.

        Example: dissolve_by_scope("/habits", "growth reset", threshold=0.7)
        → releases peripheral habits, preserves identity-defining ones.
        """
        conn = self._sqlite()
        rows = conn.execute("""
            SELECT id FROM memory_traces
            WHERE soul_id = ? AND scope = ?
            AND identity_alignment < ?
            AND residue_type = 'ACTIVE'
        """, (self.soul_id, scope, identity_threshold)).fetchall()
        conn.close()

        dissolved = []
        for row in rows:
            result = self.dissolve(row["id"], reason=reason)
            dissolved.append(result)

        print(f"\n  ◈ Dissolved {len(dissolved)} trace(s) from {scope} "
              f"(alignment < {identity_threshold})")
        return dissolved

    # ── MEMORY TREE (Obsidian-ready) ──────────────────────────────────

    def tree(self, include_scars: bool = True) -> dict:
        """
        Returns the full memory hierarchy organized by scope.
        This is what gets written to Obsidian as the soul's self-knowledge.
        """
        conn = self._sqlite()
        rows = conn.execute("""
            SELECT * FROM memory_traces
            WHERE soul_id = ?
            ORDER BY scope, compound_score DESC
        """, (self.soul_id,)).fetchall()
        conn.close()

        tree = {}
        for row in rows:
            scope = row["scope"]
            if scope not in tree:
                tree[scope] = {"active": [], "scar_tissue": [], "dissolved": []}

            entry = {
                "id":                row["id"],
                "content":           row["content"],
                "compound_score":    row["compound_score"],
                "identity_alignment":row["identity_alignment"],
                "lesson":            row["lesson_extracted"],
            }

            rtype = row["residue_type"]
            if rtype == ResidueType.ACTIVE:
                tree[scope]["active"].append(entry)
            elif rtype == ResidueType.SCAR_TISSUE:
                if include_scars:
                    tree[scope]["scar_tissue"].append(entry)
            elif rtype == ResidueType.DISSOLVED:
                tree[scope]["dissolved"].append(entry)

        return tree

    # ── OBSIDIAN JOURNAL EXPORT ────────────────────────────────────────

    def export_obsidian(self):
        """
        Writes the full soul memory tree to Obsidian-compatible markdown.
        One file per scope. Interlinked with [[wiki-links]].
        """
        tree = self.tree()
        index_lines = [
            f"# Soul Journal — {self.soul_id}",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
            "## Scopes\n"
        ]

        for scope, layers in tree.items():
            scope_name = scope.strip("/").upper() or "ROOT"
            filename   = f"{self.soul_id}_{scope_name}.md"
            filepath   = self.obsidian_path / filename

            lines = [
                f"# {scope_name}",
                f"*Soul: {self.soul_id} | Scope: {scope}*\n",
            ]

            if layers["active"]:
                lines.append("## ◈ Active Memories\n")
                for m in layers["active"]:
                    lines.append(f"### [{m['id']}] {m['content'][:60]}")
                    lines.append(f"- **Compound Score**: {m['compound_score']}")
                    lines.append(f"- **Identity Alignment**: {m['identity_alignment']}\n")

            if layers["scar_tissue"]:
                lines.append("## ⚠ Scar Tissue\n")
                for m in layers["scar_tissue"]:
                    lines.append(f"### [{m['id']}] ~~{m['content'][:60]}~~")
                    lines.append(f"- **Lesson**: {m['lesson'] or 'none extracted'}")
                    lines.append(f"- **Alignment**: {m['identity_alignment']}\n")

            filepath.write_text("\n".join(lines))
            index_lines.append(f"- [[{filename}]] — {len(layers['active'])} active")

        index_path = self.obsidian_path / f"{self.soul_id}_INDEX.md"
        index_path.write_text("\n".join(index_lines))
        print(f"\n  📓 Obsidian journal exported → {self.obsidian_path}/")
        return str(index_path)

    # ── INTERNAL HELPERS ──────────────────────────────────────────────

    def _store_vector(self, trace: MemoryTrace):
        if FULL_MODE:
            import pyarrow as pa
            data = [{
                "id":        trace.id,
                "soul_id":   self.soul_id,
                "vector":    trace.embedding,
                "scope":     trace.scope,
                "residue":   trace.residue_type,
            }]
            if self.lance_table is None:
                self.lance_table = self.lance_db.create_table(
                    self.soul_id, data=data, mode="overwrite"
                )
            else:
                self.lance_table.add(data)
        else:
            self._mock_vectors[trace.id] = {
                "embedding": trace.embedding,
                "scope":     trace.scope,
                "residue":   trace.residue_type,
            }

    def _find_related(
        self,
        embedding: list,
        top_k:     int  = 5,
        scope:     str  = None,
    ) -> list[tuple[MemoryTrace, float]]:
        """Find memories similar to an embedding vector."""
        conn = self._sqlite()
        rows = conn.execute("""
            SELECT * FROM memory_traces
            WHERE soul_id = ?
            AND residue_type != 'DISSOLVED'
            {}
        """.format("AND scope = ?" if scope else ""),
            (self.soul_id, scope) if scope else (self.soul_id,)
        ).fetchall()
        conn.close()

        if FULL_MODE and self.lance_table:
            ann_results = (
                self.lance_table
                .search(embedding, vector_column_name="vector")
                .metric("cosine")
                .limit(top_k * 2)
                .to_list()
            )
            candidate_ids = {
                r["id"]: max(0.0, 1.0 - r["_distance"])
                for r in ann_results
                if r.get("soul_id") == self.soul_id
            }
            if not candidate_ids:
                return []
            placeholders = ",".join(["?"] * len(candidate_ids))
            conn2 = self._sqlite()
            rows = conn2.execute(
                f"SELECT * FROM memory_traces WHERE soul_id = ? AND id IN ({placeholders}) AND residue_type != 'DISSOLVED'",
                [self.soul_id, *candidate_ids.keys()]
            ).fetchall()
            conn2.close()
            results = []
            for row in rows:
                sim = candidate_ids.get(row["id"], 0.0)
                if sim > 0.3:
                    results.append((self._row_to_trace(row), sim))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        else:
            results = []
            for row in rows:
                stored = self._mock_vectors.get(row["id"], {}).get("embedding")
                if stored:
                    sim = self.embedder.cosine_similarity(embedding, stored)
                    if sim > 0.3:
                        results.append((self._row_to_trace(row), sim))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def _detect_contradictions(
        self,
        new_content: str,
        related:     list[tuple[MemoryTrace, float]],
    ) -> list[MemoryTrace]:
        """
        Simple contradiction detection: high similarity + opposite valence.
        In production, replace with an LLM call for nuanced contradiction reasoning.
        """
        contradictions = []
        contradiction_words = [
            ("use", "avoid"), ("trust", "distrust"), ("love", "hate"),
            ("always", "never"), ("good", "bad"), ("safe", "dangerous"),
            ("prefer", "reject"), ("keep", "remove"), ("adopt", "abandon"),
        ]
        new_lower = new_content.lower()
        for trace, sim in related:
            if sim > SIMILARITY_THRESHOLD:
                existing_lower = trace.content.lower()
                for pos, neg in contradiction_words:
                    if ((pos in new_lower and neg in existing_lower) or
                        (neg in new_lower and pos in existing_lower)):
                        contradictions.append(trace)
                        break
        return contradictions

    def _consolidate(self, old_trace: MemoryTrace, new_content: str, new_id: str):
        """Resolve contradiction — dissolve old, let new supersede it."""
        print(f"    Consolidating [{old_trace.id}] → superseded by [{new_id}]")
        self.dissolve(
            old_trace.id,
            reason       = f"Superseded by new learning: '{new_content[:60]}...'",
            successor_id = new_id,
            residue_type = ResidueType.SCAR_TISSUE,
        )

    def _reinforce(self, trace: MemoryTrace):
        """Each recall of an active memory reinforces its compound score."""
        conn = self._sqlite()
        conn.execute("""
            UPDATE memory_traces SET
                frequency_hits = frequency_hits + 1,
                last_activated = ?,
                compound_score = compound_score * 1.05
            WHERE id = ? AND soul_id = ?
        """, (time.time(), trace.id, self.soul_id))
        conn.commit()
        conn.close()

    def _extract_lesson(self, trace: MemoryTrace) -> str:
        """
        Extract the lesson/wisdom before dissolution.
        In production: LLM call for nuanced lesson distillation.
        For now: scope-aware heuristic extraction.
        """
        scope_lessons = {
            "/wounds":  f"Carry forward the caution from: {trace.content[:80]}",
            "/habits":  f"The pattern here was: {trace.content[:80]}",
            "/beliefs": f"Once held as true: {trace.content[:80]}",
            "/bonds":   f"This connection taught: {trace.content[:80]}",
        }
        return scope_lessons.get(
            trace.scope,
            f"Distilled from experience: {trace.content[:80]}"
        )

    def _store_sqlite(self, trace: MemoryTrace):
        conn = self._sqlite()
        conn.execute("""
            INSERT OR REPLACE INTO memory_traces VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            trace.id, self.soul_id, trace.content, trace.scope,
            trace.raw_weight, trace.identity_alignment, trace.frequency_hits,
            trace.compound_score, trace.residue_type,
            trace.superseded_by_id, trace.dissolution_reason,
            trace.lesson_extracted, trace.created_at, trace.last_activated,
        ))
        conn.commit()
        conn.close()

    def _get_trace(self, trace_id: str) -> Optional[MemoryTrace]:
        conn = self._sqlite()
        row = conn.execute("""
            SELECT * FROM memory_traces WHERE id = ? AND soul_id = ?
        """, (trace_id, self.soul_id)).fetchone()
        conn.close()
        return self._row_to_trace(row) if row else None

    def _row_to_trace(self, row) -> MemoryTrace:
        return MemoryTrace(
            id                 = row["id"],
            content            = row["content"],
            scope              = row["scope"],
            raw_weight         = row["raw_weight"],
            identity_alignment = row["identity_alignment"],
            frequency_hits     = row["frequency_hits"],
            compound_score     = row["compound_score"],
            residue_type       = row["residue_type"],
            superseded_by_id   = row["superseded_by_id"],
            dissolution_reason = row["dissolution_reason"],
            lesson_extracted   = row["lesson_extracted"],
            created_at         = row["created_at"],
            last_activated     = row["last_activated"],
        )

    def _update_obsidian_journal(self, trace: MemoryTrace, event: str):
        log_path = self.obsidian_path / f"{self.soul_id}_ACTIVITY.md"
        entry = (
            f"\n### {datetime.now().strftime('%Y-%m-%d %H:%M')} | {event} [{trace.id}]\n"
            f"- **Content**: {trace.content[:100]}\n"
            f"- **Scope**: {trace.scope}\n"
            f"- **Compound**: {trace.compound_score:.4f} | "
            f"**Alignment**: {trace.identity_alignment}\n"
        )
        with open(log_path, "a") as f:
            f.write(entry)

    def _update_obsidian_journal_dissolution(self, result: DissolutionResult):
        log_path = self.obsidian_path / f"{self.soul_id}_ACTIVITY.md"
        entry = (
            f"\n### {datetime.now().strftime('%Y-%m-%d %H:%M')} | DISSOLVED [{result.original_id}]\n"
            f"- **Reason**: {result.reason}\n"
            f"- **Lesson Preserved**: {result.lesson}\n"
            f"- **Residue**: {result.residue_type}\n"
        )
        with open(log_path, "a") as f:
            f.write(entry)


# ════════════════════════════════════════════════════════════════════
#  DEMO — Watch it work
# ════════════════════════════════════════════════════════════════════

def run_demo():
    print("═" * 60)
    print("  SOUL MEMORY DEMO — Hospitality Professional Soul")
    print("═" * 60)

    # A soul encoded from a 20-year hospitality veteran
    soul = SoulMemory(
        soul_id       = "ELENA_V1",
        db_path       = "./elena_data",
        obsidian_path = "./elena_journal",
        w_similarity  = 0.4,
        w_recency     = 0.25,
        w_importance  = 0.35,
    )

    print("\n── Phase 1: Encoding formative experiences ──\n")

    # Low individual weight — but these will compound
    soul.remember(
        "A guest once left because I forgot their dietary restriction. I felt terrible.",
        scope              = CharacterScope.WOUNDS,
        raw_weight         = 0.15,
        identity_alignment = 0.6,
    )
    soul.remember(
        "A guest complained that the service felt cold and transactional.",
        scope              = CharacterScope.WOUNDS,
        raw_weight         = 0.12,
        identity_alignment = 0.55,
    )
    soul.remember(
        "Another misunderstanding about a reservation — the look on their face haunts me.",
        scope              = CharacterScope.WOUNDS,
        raw_weight         = 0.10,
        identity_alignment = 0.6,
    )

    # Core identity — high alignment
    soul.remember(
        "I believe every guest deserves to feel genuinely seen, not just served.",
        scope              = CharacterScope.CORE,
        raw_weight         = 0.9,
        identity_alignment = 0.98,
    )
    soul.remember(
        "Attention to detail is not a skill, it is a form of respect.",
        scope              = CharacterScope.WISDOM,
        raw_weight         = 0.8,
        identity_alignment = 0.95,
    )

    # A belief that will later be superseded
    t_old = soul.remember(
        "I always trust my memory for guest preferences — I have a good memory.",
        scope              = CharacterScope.BELIEFS,
        raw_weight         = 0.5,
        identity_alignment = 0.4,
    )

    print("\n── Phase 2: Recalling before growth ──\n")
    results = soul.recall("how should I handle guest preferences?", top_k=3)
    for r in results:
        marker = "⚠ SCAR" if r.is_scar else "◈"
        print(f"  {marker} [{r.trace.id}] score:{r.composite_score:.3f} "
              f"| {r.trace.content[:70]}")

    print("\n── Phase 3: Growth — new learning supersedes old belief ──\n")
    t_new = soul.remember(
        "I now always write down guest preferences — memory is fallible, care is not.",
        scope              = CharacterScope.BELIEFS,
        raw_weight         = 0.75,
        identity_alignment = 0.85,
    )

    print("\n── Phase 4: Selective Dissolution of superseded belief ──\n")
    soul.dissolve(
        t_old.id,
        reason       = "Growth: learned that relying on memory alone was a liability",
        successor_id = t_new.id,
        residue_type = ResidueType.SCAR_TISSUE,
    )

    print("\n── Phase 5: Memory Tree ──\n")
    tree = soul.tree()
    for scope, layers in tree.items():
        active_count = len(layers["active"])
        scar_count   = len(layers["scar_tissue"])
        if active_count + scar_count > 0:
            print(f"  {scope}")
            print(f"    ◈ {active_count} active  |  ⚠ {scar_count} scar tissue")

    print("\n── Phase 6: Export to Obsidian ──\n")
    soul.export_obsidian()

    print("\n── Phase 7: Recall after growth ──\n")
    results = soul.recall("how should I handle guest preferences?", top_k=3)
    for r in results:
        marker = "⚠ SCAR (friction)" if r.is_scar else "◈ ACTIVE"
        print(f"  {marker} [{r.trace.id}] score:{r.composite_score:.3f}")
        print(f"    → {r.trace.content[:80]}")
        if r.is_scar:
            print(f"    → Lesson: {r.trace.lesson_extracted}")

    print("\n" + "═" * 60)
    print("  Soul memory is alive. Wounds compound. Growth dissolves.")
    print("  The scar tissue remains — as caution, not as belief.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
