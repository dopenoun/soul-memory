"""Tests for identity alignment and trust (provenance proxy)."""
import pytest
from soul_memory import SoulMemory, CharacterScope


@pytest.fixture
def mem(tmp_path):
    return SoulMemory(
        soul_id="test-agent",
        db_path=str(tmp_path / "db"),
        obsidian_path=str(tmp_path / "journal"),
    )


def test_high_alignment_stored(mem):
    trace = mem.remember(
        content="core value",
        scope=CharacterScope.BELIEFS,
        identity_alignment=1.0,
    )
    assert trace.identity_alignment == 1.0


def test_low_alignment_stored(mem):
    trace = mem.remember(
        content="external noise",
        scope=CharacterScope.BELIEFS,
        identity_alignment=0.1,
    )
    assert trace.identity_alignment == 0.1
