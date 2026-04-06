"""
SCRIBE curation gate — nightly dissolution pass on low-value memories.
"""
from soul_memory import SoulMemory


def scribe_approves(recall_result) -> bool:
    """Stub: replace with real curation logic."""
    return recall_result.trace.compound_score > 0.4


mem = SoulMemory(soul_id="scribe")
results = mem.recall("*", top_k=100)

for r in results:
    if not scribe_approves(r):
        mem.dissolve(r.trace.id)
