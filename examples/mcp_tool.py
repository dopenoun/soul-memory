"""
MCP tool wrapper — expose remember/recall over JSON-RPC stdio.
Stub: wire to your MCP server implementation.
"""
from soul_memory import SoulMemory

mem = SoulMemory()
mem.boot_context()


def soul_memory_recall(query: str, top_k: int = 5) -> list:
    return mem.recall(query, top_k=top_k)


def soul_memory_remember(content: str, source: str = "agent_generated", trust_level: float = 0.8) -> str:
    return mem.remember(content=content, source=source, trust_level=trust_level)
