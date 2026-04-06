"""
Basic soul_memory usage — store and recall a memory.
"""
from soul_memory import SoulMemory, CharacterScope

mem = SoulMemory(soul_id="my-agent")
ctx = mem.boot_context()
print("boot context:", ctx)

mem.remember(
    content="The human prefers direct communication, no fluff.",
    scope=CharacterScope.BELIEFS,
    raw_weight=0.9,
    identity_alignment=1.0,
)

results = mem.recall("how does the human like to communicate?", top_k=3)
for r in results:
    print(r)
