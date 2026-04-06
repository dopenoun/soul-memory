"""Tests for provenance tracking and trust levels."""
import pytest
from soul_memory import SoulMemory


@pytest.fixture
def mem(tmp_path):
    return SoulMemory(data_dir=str(tmp_path))


def test_external_trust_lower_than_human(mem):
    mem.remember(content="external content", source="external_web", trust_level=0.3)
    mem.remember(content="human content", source="human_provided", trust_level=1.0)
    results = mem.recall("content", top_k=5)
    assert results
