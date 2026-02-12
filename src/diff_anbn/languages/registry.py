"""Language registry for config-based lookup."""

from typing import Type

from .base import FormalLanguage
from .anbn import AnBnLanguage
from .dyck import DyckLanguage, Dyck2Language

# Global registry
_REGISTRY: dict[str, Type[FormalLanguage]] = {}


def register_language(name: str, cls: Type[FormalLanguage]) -> None:
    """Register a language class.

    Args:
        name: Name to register under
        cls: Language class
    """
    _REGISTRY[name] = cls


def get_language(name: str, **kwargs) -> FormalLanguage:
    """Get a language instance by name.

    Args:
        name: Registered language name
        **kwargs: Arguments to pass to language constructor

    Returns:
        Language instance

    Raises:
        KeyError: If language name is not registered
    """
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(f"Unknown language '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_languages() -> list[str]:
    """List all registered language names."""
    return list(_REGISTRY.keys())


# Register built-in languages
register_language("anbn", AnBnLanguage)
register_language("dyck", DyckLanguage)
register_language("dyck1", DyckLanguage)
register_language("dyck2", Dyck2Language)
