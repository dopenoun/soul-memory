"""Tests for core memory store/recall."""
import pytest
from soul_memory import SoulMemory


@pytest.fixture
def mem(tmp_path):
    return SoulMemory(data_dir=str(tmp_path))


def test_remember_and_recall(mem):
    mem.remember(content="test memory", source="human_provided", trust_level=1.0)
    results = mem.recall("test memory", top_k=1)
    assert len(results) >= 1
