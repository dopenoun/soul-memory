"""Tests for core memory store/recall."""
import pytest
from soul_memory import SoulMemory, CharacterScope


@pytest.fixture
def mem(tmp_path):
    return SoulMemory(
        soul_id="test-agent",
        db_path=str(tmp_path / "db"),
        obsidian_path=str(tmp_path / "journal"),
    )


def test_remember_and_recall(mem):
    mem.remember(content="test memory", scope=CharacterScope.BELIEFS)
    results = mem.recall("test memory", top_k=1)
    assert len(results) >= 1
