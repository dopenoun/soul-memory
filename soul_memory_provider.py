"""
soul_memory_provider.py — Hermes MemoryProvider wrapper for SoulMemory.

Wraps core/soul_memory.py as a Hermes MemoryProvider plugin.
Does NOT modify soul_memory.py — pure wrapper.

Plugin entry point: register(ctx) at module bottom.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Hermes ABC — graceful fallback when running outside Hermes (e.g. import test)
# ---------------------------------------------------------------------------

try:
    from agent.memory_provider import MemoryProvider
except ImportError:
    class MemoryProvider:  # type: ignore
        """Stub base class — replaced by the real ABC inside Hermes."""
        def name(self): ...
        def is_available(self) -> bool: ...
        def initialize(self, session_id: str, **kwargs) -> None: ...
        def get_tool_schemas(self) -> list: ...
        def handle_tool_call(self, name: str, args: dict) -> str: ...
        def get_config_schema(self) -> list: ...
        def save_config(self, values: dict, hermes_home: str) -> None: ...

try:
    from tools.registry import tool_error
except ImportError:
    def tool_error(msg: str) -> str:  # type: ignore
        return json.dumps({"error": msg})

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SOUL_RECALL_SCHEMA = {
    "name": "soul_recall",
    "description": (
        "Query soul memory for traces relevant to a topic. "
        "Returns ACTIVE memories ranked by composite score (similarity + recency + importance). "
        "Scar tissue surfaces separately as friction signals. "
        "Use when you need to draw on the soul's formative experiences, beliefs, or wisdom."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to recall — a topic, question, or context fragment.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results to return (default 5).",
            },
            "scope": {
                "type": "string",
                "description": (
                    "Narrow recall to a specific scope: "
                    "/beliefs, /habits, /wounds, /wisdom, /skills, /bonds, /core"
                ),
            },
        },
        "required": ["query"],
    },
}

SOUL_REMEMBER_SCHEMA = {
    "name": "soul_remember",
    "description": (
        "Encode a new experience into soul memory. "
        "Embeds the content, finds related memories, detects contradictions, "
        "calculates compound score, and persists to LanceDB + SQLite + Obsidian journal. "
        "Use when the soul learns something worth retaining across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The experience or insight to encode.",
            },
            "scope": {
                "type": "string",
                "description": (
                    "Memory scope: /beliefs, /habits, /wounds, /wisdom, "
                    "/skills, /bonds, /core (default: /beliefs)"
                ),
            },
            "raw_weight": {
                "type": "number",
                "description": "Base salience weight 0.0–1.0 (default 0.1).",
            },
            "identity_alignment": {
                "type": "number",
                "description": (
                    "How core to identity this memory is, 0.0–1.0. "
                    "High-alignment memories resist decay and surface as caution when wounded."
                ),
            },
        },
        "required": ["content"],
    },
}

SOUL_TREE_SCHEMA = {
    "name": "soul_tree",
    "description": (
        "Return the full soul memory hierarchy organized by scope. "
        "Shows active memories, scar tissue, and dissolved traces. "
        "Useful for introspection or debugging the soul's current state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "include_scars": {
                "type": "boolean",
                "description": "Include scar tissue in the output (default true).",
            }
        },
        "required": [],
    },
}

SOUL_BOOT_CONTEXT_SCHEMA = {
    "name": "soul_boot_context",
    "description": (
        "Return the warm L0+L1 memory cache as structured data. "
        "L0: CORE identity-defining memories always in RAM. "
        "L1: Top-N high-salience memories loaded at session boot. "
        "Use to prime reasoning at session start without a vector search."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

ALL_TOOL_SCHEMAS = [
    SOUL_RECALL_SCHEMA,
    SOUL_REMEMBER_SCHEMA,
    SOUL_TREE_SCHEMA,
    SOUL_BOOT_CONTEXT_SCHEMA,
]


# ---------------------------------------------------------------------------
# SoulMemoryProvider
# ---------------------------------------------------------------------------

class SoulMemoryProvider(MemoryProvider):
    """
    Hermes MemoryProvider wrapper for soul_memory.SoulMemory.

    Wraps the three-layer cognitive memory stack (LanceDB + SQLite + Obsidian)
    as a Hermes plugin. Exposes soul_recall, soul_remember, soul_tree,
    and soul_boot_context as agent tools.

    sync_turn strategy: reinforce-on-recall only. A daemon-threaded recall()
    on user_content fires natural reinforcement (recall() calls _reinforce()
    internally on active hits). No auto-remember — new traces are only encoded
    via explicit soul_remember tool calls.
    """

    def __init__(self):
        self._soul: Optional[Any] = None          # SoulMemory instance
        self._soul_id: str = ""
        self._session_id: str = ""
        self._hermes_home: Optional[Path] = None
        self._tool_log: Optional[Path] = None
        self._sync_thread: Optional[threading.Thread] = None

    # -----------------------------------------------------------------------
    # ABC: name
    # -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "soul_memory"

    # -----------------------------------------------------------------------
    # ABC: is_available — no network calls
    # -----------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check importability of LanceDB stack. No network calls."""
        try:
            import lancedb  # noqa: F401
            import numpy  # noqa: F401
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    # -----------------------------------------------------------------------
    # ABC: initialize
    # -----------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        """
        Construct SoulMemory with hermes_home-scoped paths.

        hermes_home is always passed as a kwarg by Hermes — never hardcoded.
        db_path   → hermes_home / "soul_memory" / soul_id
        obsidian_path → ~/the-house/vault  (the live vault, not hermes_home)
        """
        try:
            raw_home = kwargs.get("hermes_home")
            if not raw_home:
                logger.warning("soul_memory: hermes_home not provided — plugin inactive")
                return

            hermes_home = Path(raw_home)
            self._hermes_home = hermes_home

            # Load config from hermes_home/soul_memory.json
            config = self._load_config(hermes_home)
            self._soul_id = config.get("soul_id", "")
            if not self._soul_id:
                logger.warning("soul_memory: soul_id not configured — plugin inactive")
                return

            embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
            l1_size = int(config.get("l1_size", 20))

            db_path = hermes_home / "soul_memory" / self._soul_id
            obsidian_path = Path.home() / "the-house" / "vault"
            self._session_id = session_id
            self._tool_log = hermes_home / "soul_memory" / self._soul_id / "tool_log.jsonl"
            self._tool_log.parent.mkdir(parents=True, exist_ok=True)

            # Import here so is_available() failure doesn't break the module
            import sys
            soul_core = Path(__file__).parent
            if str(soul_core) not in sys.path:
                sys.path.insert(0, str(soul_core))

            from soul_memory import SoulMemory

            self._soul = SoulMemory(
                soul_id       = self._soul_id,
                db_path       = str(db_path),
                obsidian_path = str(obsidian_path),
            )

            # Apply optional l1_size from config
            if l1_size != 20:
                self._soul._l1.clear()
                self._soul._load_l1(top_n=l1_size)

            logger.debug(
                "soul_memory initialized: soul_id=%s db=%s obsidian=%s",
                self._soul_id, db_path, obsidian_path,
            )

        except Exception as e:
            logger.warning("soul_memory init failed: %s", e)
            self._soul = None

    def _load_config(self, hermes_home: Path) -> dict:
        config_path = hermes_home / "soul_memory.json"
        if config_path.exists():
            try:
                return json.loads(config_path.read_text())
            except Exception as e:
                logger.debug("soul_memory: config parse error: %s", e)
        return {}

    # -----------------------------------------------------------------------
    # ABC: get_config_schema
    # -----------------------------------------------------------------------

    def get_config_schema(self) -> list:
        return [
            {
                "key": "soul_id",
                "description": "Soul identifier (e.g. 'aifriedenewman')",
                "required": True,
            },
            {
                "key": "embedding_model",
                "description": "Sentence-transformers model name",
                "default": "all-MiniLM-L6-v2",
            },
            {
                "key": "l1_size",
                "description": "Number of high-salience memories to load at session boot (L1 tier)",
                "default": "20",
            },
        ]

    # -----------------------------------------------------------------------
    # ABC: save_config
    # -----------------------------------------------------------------------

    def save_config(self, values: dict, hermes_home: str) -> None:
        """Write non-secret config to hermes_home/soul_memory.json."""
        config_path = Path(hermes_home) / "soul_memory.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    # -----------------------------------------------------------------------
    # ABC: get_tool_schemas
    # -----------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._soul is None:
            return []
        return list(ALL_TOOL_SCHEMAS)

    # -----------------------------------------------------------------------
    # ABC: handle_tool_call
    # -----------------------------------------------------------------------

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._soul is None:
            return tool_error("soul_memory is not active for this session.")

        try:
            if tool_name == "soul_recall":
                query = args.get("query", "")
                if not query:
                    return tool_error("Missing required parameter: query")
                top_k = int(args.get("top_k", 5))
                scope = args.get("scope") or None
                results = self._soul.recall(query, top_k=top_k, scope=scope)
                return json.dumps({
                    "results": [
                        {
                            "id":              r.trace.id,
                            "content":         r.trace.content,
                            "scope":           r.trace.scope,
                            "similarity":      r.similarity,
                            "composite_score": r.composite_score,
                            "tier":            r.tier,
                            "is_scar":         r.is_scar,
                            "friction_signal": r.friction_signal,
                        }
                        for r in results
                    ]
                })

            elif tool_name == "soul_remember":
                content = args.get("content", "")
                if not content:
                    return tool_error("Missing required parameter: content")
                from soul_memory import CharacterScope
                scope              = args.get("scope", CharacterScope.BELIEFS)
                raw_weight         = float(args.get("raw_weight", 0.1))
                identity_alignment = float(args.get("identity_alignment", 0.5))
                trace = self._soul.remember(
                    content,
                    scope              = scope,
                    raw_weight         = raw_weight,
                    identity_alignment = identity_alignment,
                )
                return json.dumps({
                    "id":              trace.id,
                    "content":         trace.content,
                    "scope":           trace.scope,
                    "compound_score":  trace.compound_score,
                    "tier":            self._soul.tier_of(trace.id),
                })

            elif tool_name == "soul_tree":
                include_scars = bool(args.get("include_scars", True))
                tree = self._soul.tree(include_scars=include_scars)
                return json.dumps({"tree": tree})

            elif tool_name == "soul_boot_context":
                ctx = self._soul.boot_context()
                return json.dumps(ctx)

            return tool_error(f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.error("soul_memory tool %s failed: %s", tool_name, e)
            return tool_error(f"soul_memory {tool_name} failed: {e}")

    # -----------------------------------------------------------------------
    # Optional: system_prompt_block
    # -----------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        """Render L0+L1 warm cache as a plaintext block for the system prompt."""
        if self._soul is None:
            return ""

        try:
            ctx = self._soul.boot_context()
            soul_id = ctx.get("soul_id", self._soul_id)
            l0 = ctx.get("tiers", {}).get("L0", [])
            l1 = ctx.get("tiers", {}).get("L1", [])

            lines = [f"# Soul Memory — {soul_id}"]

            if l0:
                lines.append("\n## Core Identity (L0 — always present)")
                for m in l0:
                    lines.append(
                        f"  [{m['id']}] {m['content']}  "
                        f"(alignment:{m['identity_alignment']} score:{m['compound_score']})"
                    )

            if l1:
                lines.append("\n## High-Salience Memories (L1 — session boot)")
                for m in l1:
                    lines.append(
                        f"  [{m['id']}] {m['content']}  "
                        f"(scope:{m['scope']} score:{m['compound_score']})"
                    )

            if not l0 and not l1:
                lines.append("No memories encoded yet.")

            return "\n".join(lines)

        except Exception as e:
            logger.debug("soul_memory system_prompt_block failed: %s", e)
            return ""

    # -----------------------------------------------------------------------
    # Optional: prefetch — fires recall, returns formatted string
    # -----------------------------------------------------------------------

    def prefetch(self, query: str, **kwargs) -> str:
        """Run recall(query) and return results as an injected context block."""
        if self._soul is None or not query:
            return ""

        try:
            results = self._soul.recall(query, top_k=5)
            if not results:
                return ""

            active = [r for r in results if not r.is_scar]
            scars  = [r for r in results if r.is_scar]

            lines = ["## Soul Memory — Recalled Context"]
            for r in active:
                lines.append(
                    f"  [{r.trace.id}|{r.trace.scope}|score:{r.composite_score}] "
                    f"{r.trace.content}"
                )
            if scars:
                lines.append("## Friction Signals (Scar Tissue)")
                for r in scars:
                    lines.append(
                        f"  [{r.trace.id}] friction:{r.friction_signal}  "
                        f"{r.trace.content}"
                        + (f"  → lesson: {r.trace.lesson_extracted}" if r.trace.lesson_extracted else "")
                    )

            return "\n".join(lines)

        except Exception as e:
            logger.debug("soul_memory prefetch failed: %s", e)
            return ""

    # -----------------------------------------------------------------------
    # Optional: sync_turn — reinforce-on-recall only, daemon thread
    # -----------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, **kwargs) -> None:
        """
        Non-blocking passive reinforcement.

        Fires recall() on user_content in a daemon thread.
        recall() internally calls _reinforce() on every active hit —
        so frequently-touched memories compound naturally.
        No auto-remember: new traces only arrive via soul_remember tool calls.
        """
        if self._soul is None or not user_content:
            return

        def _reinforce():
            try:
                self._soul.recall(user_content, top_k=5)
            except Exception as e:
                logger.debug("soul_memory sync_turn reinforce failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(
            target=_reinforce, daemon=True, name="soul-memory-sync"
        )
        self._sync_thread.start()

    # -----------------------------------------------------------------------
    # Expeditor log — lightweight JSONL, no embedding overhead
    # -----------------------------------------------------------------------

    def _log_tool_event(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Stamp every tool call: session_id, timestamp, tool name, args hash."""
        if not self._tool_log:
            return
        try:
            args_hash = hashlib.sha256(
                json.dumps(args, sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
            entry = {
                "ts":         int(time.time()),
                "session_id": self._session_id,
                "soul_id":    self._soul_id,
                "tool":       tool_name,
                "args_hash":  args_hash,
            }
            with open(self._tool_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug("soul_memory tool_log write failed: %s", e)

    # -----------------------------------------------------------------------
    # Optional: shutdown
    # -----------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

_provider: Optional[SoulMemoryProvider] = None


def _expeditor_hook(tool_name: str, args: dict, **kwargs) -> None:
    """pre_tool_call observer — logs every tool call to the expeditor JSONL."""
    if _provider is not None:
        _provider._log_tool_event(tool_name, args)


def register(ctx) -> None:
    """Register SoulMemory as a Hermes memory provider and wire expeditor hook."""
    global _provider
    _provider = SoulMemoryProvider()
    ctx.register_memory_provider(_provider)
    # Memory plugins use a _ProviderCollector fake context where register_hook
    # is a no-op. Fall through to the real plugin manager directly.
    ctx.register_hook("pre_tool_call", _expeditor_hook)
    try:
        from hermes_cli.plugins import get_plugin_manager
        get_plugin_manager()._hooks.setdefault("pre_tool_call", []).append(_expeditor_hook)
    except Exception:
        pass
