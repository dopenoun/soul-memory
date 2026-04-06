"""Tests for nine-dimensional salience scoring."""
import pytest
from soul_memory import SoulMemory


@pytest.fixture
def mem(tmp_path):
    return SoulMemory(data_dir=str(tmp_path))


def test_salience_vector_stored(mem):
    salience = {"relational_relevance": 0.9, "long_term_value": 0.8, "delight": 0.5}
    mem.remember(content="salience test", source="human_provided", trust_level=1.0, salience=salience)
    results = mem.recall("salience test", top_k=1)
    assert results
