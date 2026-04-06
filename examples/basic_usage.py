"""
Basic soul_memory usage — store and recall a memory.
"""
from soul_memory import SoulMemory

mem = SoulMemory()
mem.boot_context()

mem.remember(
    content="The human prefers direct communication, no fluff.",
    source="human_provided",
    trust_level=1.0,
    salience={
        "relational_relevance": 0.9,
        "long_term_value": 0.95,
        "ethical_alignment": 1.0,
        "delight": 0.3,
    },
)

results = mem.recall("how does the human like to communicate?", top_k=3)
for r in results:
    print(r)
