"""
SCRIBE curation gate — nightly L2 promotion/dissolution pass.
"""
from soul_memory import SoulMemory


def scribe_approves(memory) -> bool:
    """Stub: replace with real curation logic."""
    return memory.salience.get("long_term_value", 0) > 0.7


mem = SoulMemory()
pending = mem.get_pending_promotions()

for memory in pending:
    if scribe_approves(memory):
        mem.promote(memory.id, to_tier="L1")
    else:
        mem.dissolve(memory.id, reason="curation_rejected")
