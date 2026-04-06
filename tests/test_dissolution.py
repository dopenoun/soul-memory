"""Tests for selective dissolution — wound/scar/noise lifecycle."""
import pytest
from soul_memory import SoulMemory, CharacterScope, ResidueType


@pytest.fixture
def mem(tmp_path):
    return SoulMemory(
        soul_id="test-agent",
        db_path=str(tmp_path / "db"),
        obsidian_path=str(tmp_path / "journal"),
    )


def test_dissolve_changes_residue_type(mem):
    trace = mem.remember(content="temporary noise", scope=CharacterScope.BELIEFS)
    assert trace.residue_type == ResidueType.ACTIVE
    mem.dissolve(trace.id)
    # After dissolution, memory should not surface in active recall
    results = mem.recall("temporary noise", top_k=5)
    ids = [r.trace.id for r in results]
    assert trace.id not in ids
