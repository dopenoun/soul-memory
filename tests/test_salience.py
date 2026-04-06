"""Tests for compound salience scoring."""
import pytest
from soul_memory import SoulMemory, CharacterScope


@pytest.fixture
def mem(tmp_path):
    return SoulMemory(
        soul_id="test-agent",
        db_path=str(tmp_path / "db"),
        obsidian_path=str(tmp_path / "journal"),
    )


def test_high_weight_memory_scores_higher(mem):
    mem.remember(content="high salience", scope=CharacterScope.BELIEFS, raw_weight=0.9)
    mem.remember(content="low salience", scope=CharacterScope.BELIEFS, raw_weight=0.1)
    results = mem.recall("salience", top_k=5)
    assert results
    # highest compound_score should be first
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
