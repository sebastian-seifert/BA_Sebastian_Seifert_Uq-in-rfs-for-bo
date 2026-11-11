"""Registry of epistemic-uncertainty extractors, looked up by string key from the BO surrogate."""

from dyrf_bo.uq.extractors.base import BaseEpistemicExtractor


class UQExtractorRegistry:
    """Maps registration keys to :class:`BaseEpistemicExtractor` subclasses."""

    _registry: dict[str, type[BaseEpistemicExtractor]] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator registering an extractor under ``name``."""

        def decorator(subclass: type[BaseEpistemicExtractor]):
            if not issubclass(subclass, BaseEpistemicExtractor):
                raise TypeError(
                    f"Class {subclass.__name__} must subclass BaseEpistemicExtractor to be registered."
                )
            if name in cls._registry:
                raise ValueError(
                    f"UQ Extractor key '{name}' is already registered by class {cls._registry[name].__name__}."
                )
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str, model, **kwargs) -> BaseEpistemicExtractor:
        """Instantiate the extractor registered under ``name`` for ``model``.

        Parameters
        ----------
        name : str
            Registration key.
        model : RandomForestRegressor
            Fitted random forest surrogate.
        **kwargs
            Forwarded to the extractor constructor.
        """
        if name not in cls._registry:
            raise KeyError(
                f"No UQ Extractor is registered under the key '{name}'. "
                f"Available extractors: {cls.list_registered()}"
            )
        return cls._registry[name](model, **kwargs)

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return all registered keys."""
        return list(cls._registry.keys())


# Importing the modules registers their extractors.
from dyrf_bo.uq.extractors import proximity_ac, proximity_b, proximity_bc, shaker_entropy  # noqa: E402, F401
