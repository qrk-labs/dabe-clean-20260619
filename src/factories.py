from .bitmask_encoder.base import BitmaskEncoder
from .bitmask_encoder.gumbel_encoder import GumbelEncoder
from .bitmask_encoder.lfq_encoder import LFQEncoder
from .bitmask_encoder.semantic_hash import SemanticHashEncoder
from .density_router.base import DensityRouter
from .density_router.entropy_based import EntropyRouter
from .density_router.fixed import FixedRouter
from .density_router.ib_policy import InformationBottleneckRouter
from .density_router.learned_router import LearnedRouter


def build_router(config: dict) -> DensityRouter:
    router_cfg = config.get("router", {})
    router_type = config.get("router_type", router_cfg.get("type", "entropy")).lower()

    router_map = {
        "entropy": EntropyRouter,
        "learned": LearnedRouter,
        "ib": InformationBottleneckRouter,
        "ib_policy": InformationBottleneckRouter,
        "fixed": FixedRouter,
        "uniform": FixedRouter,
    }
    router_cls = router_map.get(router_type, EntropyRouter)
    return router_cls(router_cfg)


def build_encoder(config: dict) -> BitmaskEncoder:
    encoder_cfg = config.get("encoder", {})
    encoder_type = config.get("encoder_type", encoder_cfg.get("type", "lfq")).lower()

    encoder_map = {
        "lfq": LFQEncoder,
        "gumbel": GumbelEncoder,
        "semantic_hash": SemanticHashEncoder,
        "semantic-hash": SemanticHashEncoder,
    }
    encoder_cls = encoder_map.get(encoder_type, LFQEncoder)
    return encoder_cls(encoder_cfg)
