"""
Hermes integration pattern — memory in a subprocess agent loop.
"""
from soul_memory import SoulMemory

mem = SoulMemory(soul_id="hermes")
mem.boot_context()


def handle_message(message_content: str) -> list:
    """Retrieve relevant memory context before passing to model."""
    return mem.recall(message_content, top_k=5)
