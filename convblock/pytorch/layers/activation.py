"""Contains various custom pytorch activation functions."""

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812

from ..utils import ArrayLike
from .conv_block import ConvBlock
from .torch_module import TorchModule


class Swish(torch.nn.Module):
    """Implementation for swish(silu) activation function.

    Attributes
    ----------
    inpace : bool
        whether to apply activation inplace or not.
    """

    def __init__(self, inplace: bool = True):
        """Construct swish activation function.

        Parameters
        ----------
        inplace : bool
            whether to apply swish activation function
            inplace or not. Default is True
        """
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through swish activation function.

        Parameters
        ----------
        x : torch.Tensor
            input tensor over which silu
            activation function will be applied.
        """
        return F.silu(x, inplace=self.inplace)


class HSwish(torch.nn.Module):
    """Implementation of hard swish activation function.

    Attributes
    ----------
    inplace : bool
        whether to apply activation inplace or not.
    """

    def __init__(self, inplace: bool = True):
        """Construct hard swish activation function.

        Parameters
        ----------
        inplace : bool
            whether to apply hard swish activation function
            inplace or not. Default is True.
        """
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through hard swish activation function.

        Parameters
        ----------
        x : torch.Tensor
            input tensor over which hard
            silu activation will be applied.
        """
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class HSigmoid(torch.nn.Module):
    """Hard sigmoid activation function implementation.

    Attributes
    ----------
    inplace : bool
        whether to apply hard sigmoid activation function
        inplace or not.
    """

    def __init__(self, inplace: bool = True):
        """Construct hard sigmoid activation function.

        Parameters
        ----------
        inplace : bool
            whether to apply hard sigmoid activation function
            inplace or not. Default is True.
        """
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through hard sigmoid activation function.

        Parameters
        ----------
        x : torch.Tensor
            input tensor over which hard sigmoid activation function
            will be applied.
        """
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


ACTIVATIONS = Literal[
    "relu",
    "relu6",
    "leaky_relu",
    "prelu",
    "sigmoid",
    "hsigmoid",
    "silu",
    "swish",
    "hswish",
    "tanh",
    "prelu",
    "elu",
    "selu",
    "softmax",
    "softmin",
    "softplus",
    "linear",
]


@ConvBlock.register_option(name="a")
class Activation(TorchModule):
    """Generalized activation layer."""

    def __init__(
        self,
        input_shape: ArrayLike[int],
        activation: ACTIVATIONS = "relu",
        alpha: float = 1.0,
        beta: float = 1.0,
        inplace: bool = True,
        init: float = 0.1,
        negative_slope: float = 0.01,
        num_parameters: int = 1,
        **kwargs,
    ):
        """Generalized activation layer.

        Parameters
        ----------
        input_shape : int, Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that batch dimension is not
            taken into account.
        activation : str
            possible values: 'relu', 'prelu', 'elu',
            'selu', 'sigmoid', 'softmax' or 'leaky_relu'.
            Default is 'relu'.
        alpha : float
            alpha value for 'elu' activation function. Default is 1.0.
        inplace : bool
            put activation inplace. This parameter
            exists only for 'relu', 'leaky_relu',
            'elu' and 'selu' activation functions. Default is True.
        init : float
            init argument required by 'prelu' activation. Default is 0.1.
        negative_slope : float
            slope in negative halfspace. This argument required by 'leaky_relu'.
            Default is 0.01.
        num_parameters : int
            required by 'prelu' activation. Default is 1.
        **kwargs : dict
            these parameters will be ignored.

        Raises
        ------
        ValueError
            if argument 'activation' is not str or None value.
        """
        super().__init__(input_shape=input_shape, output_shape=input_shape)
        if not (isinstance(activation, str) or activation is None):
            raise ValueError("Argument 'activation' must have " + "type 'str' or be None.")
        activation = "linear" if activation is None else activation
        activation = activation.lower()
        activation = activation.strip()

        self.params = []
        match activation:
            case "relu":
                self.layer = torch.nn.ReLU(inplace=inplace)
            case "relu6":
                self.layer = torch.nn.ReLU6(inplace=inplace)
            case "leaky_relu":
                self.layer = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
            case "prelu":
                self.layer = torch.nn.PReLU(num_parameters=num_parameters)
            case "sigmoid":
                self.layer = torch.nn.Sigmoid()
            case "hsigmoid":
                self.layer = HSigmoid(inplace=inplace)
            case "silu":
                self.layer = torch.nn.SiLU(inplace=inplace)
            case "swish":
                self.layer = Swish(inplace=inplace)
            case "hswish":
                self.layer = HSwish(inplace=inplace)
            case "tanh":
                self.layer = torch.nn.Tanh()
            case "prelu":
                self.layer = torch.nn.PReLU()
            case "elu":
                self.layer = torch.nn.ELU(alpha=alpha, inplace=inplace)
                self.params = [float(alpha)]
            case "selu":
                self.layer = torch.nn.SELU(inplace=inplace)
                self.params = [1.6732, 1.0507]
            case "softmax":
                self.layer = torch.nn.Softmax(dim=1)
            case "softmin":
                self.layer = torch.nn.Softmin(dim=1)
            case "softplus":
                self.layer = torch.nn.Softplus(beta=beta, threshold=kwargs.get("threshold"))
                self.params = [beta, kwargs.get("threshold")]
            case "linear":
                self.layer = None
                self.params = []
            case _:

                raise ValueError(
                    "Argument 'activation' must be one "
                    + "of following values: 'relu', 'leaky_relu', "
                    + "'sigmoid', 'linear', 'elu' or None."
                )
        self.activation = activation

    def forward(self, inputs: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        """Forward pass method.

        Parameters
        ----------
        inputs : torch.Tensor
            input tensor.

        Returns
        -------
        Tensor
            result of activaton function application.
        """
        if self.layer is None:
            return inputs
        return self.layer.forward(inputs)
