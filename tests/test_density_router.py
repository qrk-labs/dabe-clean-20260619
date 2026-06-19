import pytest

from src.density_router.base import Span, DensityRouter
from src.density_router.entropy_based import EntropyRouter


class TestSpan:
    def test_creation(self):
        span = Span(text="hello", start=0, end=5, density=0.8, bit_width=32)
        assert span.text == "hello"
        assert span.density == 0.8
        assert span.bit_width == 32


class TestEntropyRouter:
    @pytest.fixture
    def config(self):
        return {
            "density_lower": 0.0,
            "density_upper": 1.0,
            "min_bit_width": 8,
            "max_bit_width": 32,
        }

    def test_get_bit_width(self, config):
        router = EntropyRouter(config)
        assert router.get_bit_width(0.0) == 8
        assert router.get_bit_width(1.0) == 32
        assert router.get_bit_width(0.5) == 20

    def test_segment_empty(self, config):
        router = EntropyRouter(config)
        spans = router.segment("", [])
        assert spans == []


class TestDensityRouter:
    def test_abstract_enforces_implementation(self):
        with pytest.raises(TypeError):
            DensityRouter({})
