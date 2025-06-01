"""Contains implementation of initializer factory."""

from __future__ import annotations

from typing import Callable, ClassVar, Type, TypeVar

from .base_initializer import BaseInitializer
from .initializer_config import BaseInitializerConfig, InitializerConfigDict

T = TypeVar("T", bound=BaseInitializer)


class InitializerFactory:
    """Factory for creating weight initializers from configuration.

    Maintains registries of initializer types and their corresponding
    Pydantic configuration classes. Provides a unified interface for
    building initializer instances from configuration dictionaries.

    Examples
    --------
    >>> from .initializer_factory import InitializerFactory
    >>> config = {"init_type": "normal", "mean": 0.0, "std": 0.01}
    >>> init = InitializerFactory.create_initializer(config)
    >>> init.initialize(torch.empty(10, 10))

    Notes
    -----
    All initializers must be registered using `@InitializerFactory.register_initializer(...)`
    decorator to be available via `create_initializer`.
    """

    _configs: ClassVar[dict[str, Type[BaseInitializerConfig]]] = {}
    _initializers: ClassVar[dict[str, Type[BaseInitializer]]] = {}

    @classmethod
    def register_initializer(
        cls, init_type: str, config_cls: Type[BaseInitializerConfig]
    ) -> Callable[[Type[T]], Type[T]]:
        """Inner decorator that registers the initializer implementation.

        Parameters
        ----------
        initializer_cls : Type[T]
            Class implementing the initializer logic. Must implement
            `from_config(cls, config_dict: dict)` and `initialize(x)`.

        Returns
        -------
        Type[T]
            The same class, after registering it in the factory registry.

        Raises
        ------
        RuntimeError
            If an initializer is already registered under the same `init_type`.
        """

        def _decorator(initializer_cls: Type[T]) -> Type[T]:

            if init_type in cls._initializers:
                raise RuntimeError(
                    f"Initialize for for init_type=`{init_type}` is already registered"
                )
            cls._configs[init_type] = config_cls
            cls._initializers[init_type] = initializer_cls
            return initializer_cls

        return _decorator

    @classmethod
    def create_initializer(cls, config_dict: InitializerConfigDict) -> BaseInitializer:
        """Create initializer given configuration dictionary.

        Parameters
        ----------
        config_dict : InitializerConfigDict
            initializer configuration dictionary instance.

        Returns
        -------
        BaseInitializer
            initializer instance created using configuration dictionary.
        """
        base_config = BaseInitializerConfig.model_validate(config_dict)
        if base_config.init_type not in cls._configs:
            raise RuntimeError(
                f"No registered config class found for initializer type=`{base_config.init_type}`"
            )
        builder_cls = cls._initializers[base_config.init_type]
        return builder_cls.from_config(config_dict=config_dict)
