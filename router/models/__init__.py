"""Router models. Router classes load lazily so serving can import `router.models.artifact` without sklearn."""
from router.models.artifact import LinearRouterArtifact

_ROUTERS = ("BaseRouter", "LogisticRouter", "MLPRouter", "GapRegressionRouter", "KNNRouter")
__all__ = [*_ROUTERS, "LinearRouterArtifact"]


def __getattr__(name):
    if name in _ROUTERS:
        from router.models import routers
        return getattr(routers, name)
    raise AttributeError(name)
