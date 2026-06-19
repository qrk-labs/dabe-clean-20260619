from src.data.density_sampler import DensitySampler


class TestDensitySampler:
    def test_sample_balanced(self):
        sampler = DensitySampler(density_buckets=[0.1, 0.5, 0.9])
        spans = ["a", "b", "c", "d", "e", "f"]
        densities = [0.1, 0.1, 0.5, 0.5, 0.9, 0.9]
        balanced = sampler.sample_balanced(spans, densities)
        assert len(balanced) == 6
