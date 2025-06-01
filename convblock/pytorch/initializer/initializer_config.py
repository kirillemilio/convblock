"""Contains implementation of initializer configuration dictionaries."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel


class InitializerConfigDict(TypedDict):
    """Define base initializer config structure for legacy style usage.

    Represents a flexible dictionary-based configuration for weight
    initializers. Used when strong validation is not required.

    Attributes
    ----------
    init_type : str
        Type of initializer (e.g., 'zeros', 'normal', etc.).
    val : float, optional
        Constant value used for 'constant' initializer.
    a : float, optional
        Lower bound for 'uniform' initializer.
    b : float, optional
        Upper bound for 'uniform' initializer.
    mean : float, optional
        Mean value for 'normal' initializer.
    std : float, optional
        Standard deviation for 'normal' initializer.
    gain : float, optional
        Gain factor for Xavier initializers.
    """

    init_type: str
    val: NotRequired[float]
    a: NotRequired[float]
    b: NotRequired[float]
    mean: NotRequired[float]
    std: NotRequired[float]
    gain: NotRequired[float]


class BaseInitializerConfig(BaseModel):
    """Define base Pydantic model for initializer configuration.

    Used as a base class for all structured initializer configs.

    Attributes
    ----------
    init_type : str
        Type of initializer (must be specified in derived classes).
    """

    init_type: str


class ZerosInitializerConfig(BaseInitializerConfig):
    """Configure zero-initialization of parameters.

    Sets all weights to zero.

    Attributes
    ----------
    init_type : Literal["zeros"]
        Type of initializer, fixed as 'zeros'.
    """

    init_type: Literal["zeros"] = "zeros"


class OnesInitializerConfig(BaseInitializerConfig):
    """Configure one-initialization of parameters.

    Sets all weights to one.

    Attributes
    ----------
    init_type : Literal["ones"]
        Type of initializer, fixed as 'ones'.
    """

    init_type: Literal["ones"] = "ones"


class ConstantInitializerConfig(BaseInitializerConfig):
    """Configure constant initialization of parameters.

    Sets all weights to a user-defined constant value.

    Attributes
    ----------
    init_type : Literal["constant"]
        Type of initializer, fixed as 'constant'.
    val : float, default=0.0
        Constant value to assign to all weights.
    """

    init_type: Literal["constant"] = "constant"
    val: float = 0.0


class UniformInitializerConfig(BaseInitializerConfig):
    """Configure uniform distribution initialization.

    Samples weights uniformly in the interval [a, b].

    Attributes
    ----------
    init_type : Literal["uniform"]
        Type of initializer, fixed as 'uniform'.
    a : float, default=0.0
        Lower bound of the uniform distribution.
    b : float, default=1.0
        Upper bound of the uniform distribution.
    """

    init_type: Literal["uniform"] = "uniform"
    a: float = 0.0
    b: float = 1.0


class NormalInitializerConfig(BaseInitializerConfig):
    """Configure normal distribution initialization.

    Samples weights from a normal distribution with given mean and std.

    Attributes
    ----------
    init_type : Literal["normal"]
        Type of initializer, fixed as 'normal'.
    mean : float, default=0.0
        Mean of the normal distribution.
    std : float, default=1.0
        Standard deviation of the normal distribution.
    """

    init_type: Literal["normal"] = "normal"
    mean: float = 0.0
    std: float = 1.0


class TruncatedNormalInitializerConfig(BaseInitializerConfig):
    """Pydantic config model for truncated normal initializer."""

    init_type: Literal["tnormal"] = "tnormal"
    mean: float = 0.0
    std: float = 1.0
    a: float = -2.0
    b: float = 2.0


class XavierNormalInitializerConfig(BaseInitializerConfig):
    """Configure Xavier normal initialization.

    Uses gain-scaled normal distribution based on fan-in and fan-out.

    Attributes
    ----------
    init_type : Literal["nxavier"]
        Type of initializer, fixed as 'nxavier'.
    gain : float, default=1.0
        Scaling factor for the weights.
    """

    init_type: Literal["nxavier"]
    gain: float = 1.0


class XavierUniformInitializerConfig(BaseInitializerConfig):
    """Configure Xavier uniform initialization.

    Uses gain-scaled uniform distribution based on fan-in and fan-out.

    Attributes
    ----------
    init_type : Literal["uxavier"]
        Type of initializer, fixed as 'uxavier'.
    gain : float, default=1.0
        Scaling factor for the weights.
    """

    init_type: Literal["uxavier"]
    gain: float = 1.0


class KaimingNormalInitializerConfig(BaseInitializerConfig):
    """Configuration for Kaiming normal initializer."""

    init_type: Literal["nkaiming"] = "nkaiming"
    a: float = 0.0
    mode: Literal["fan_in", "fan_out"] = "fan_in"
    nonlinearity: str = "leaky_relu"


class KaimingUniformInitializerConfig(BaseInitializerConfig):
    """Configuration for Kaiming uniform initializer."""

    init_type: Literal["ukaiming"] = "ukaiming"
    a: float = 0.0
    mode: Literal["fan_in", "fan_out"] = "fan_in"
    nonlinearity: str = "leaky_relu"


InitializerConfigUnion = (
    ZerosInitializerConfig
    | OnesInitializerConfig
    | ConstantInitializerConfig
    | UniformInitializerConfig
    | NormalInitializerConfig
    | TruncatedNormalInitializerConfig
    | XavierUniformInitializerConfig
    | XavierNormalInitializerConfig
    | KaimingNormalInitializerConfig
    | KaimingUniformInitializerConfig
)
