"""Tests for selective dissolution — wound/scar/noise lifecycle."""
import pytest
from soul_memory import SoulMemory


@pytest.fixture
def mem(tmp_path):
    return SoulMemory(data_dir=str(tmp_path))


def test_dissolve_removes_influence(mem):
    mem.remember(content="temporary noise", source="agent_generated", trust_level=0.3)
    results_before = mem.recall("temporary noise", top_k=5)
    assert results_before
