"""
MCP tool wrapper — expose remember/recall over JSON-RPC stdio.
Stub: wire to your MCP server implementation.
"""
from soul_memory import SoulMemory, CharacterScope

mem = SoulMemory(soul_id="mcp-agent")
mem.boot_context()


def soul_memory_recall(query: str, top_k: int = 5) -> list:
    return mem.recall(query, top_k=top_k)


def soul_memory_remember(
    content: str,
    scope: str = CharacterScope.BELIEFS,
    raw_weight: float = 0.5,
    identity_alignment: float = 0.5,
) -> object:
    return mem.remember(
        content=content,
        scope=scope,
        raw_weight=raw_weight,
        identity_alignment=identity_alignment,
    )
