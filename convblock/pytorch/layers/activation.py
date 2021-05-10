""" Contains various custom pytorch activation functions. """

import torch
import torch.nn.functional as F

from ..bases import Layer
from .conv_block import ConvBlock


def softmax1d(inputs: 'Tensor') -> 'Tensor':
    """ Apply softmax along channels dimension to batch of 1D signals.

    Parameters
    ----------
    inputs : Tensor
        3D input tensor. First dimension is considered to be associated with
        batch items, second with channels.

    Returns
    -------
    Tensor
        tensor containing result of softmax operation.
    """
    batch_size, num_channels = int(inputs.size(0)), int(inputs.size(1))
    size = int(inputs.size(2))
    x = (
        inputs
        .permute(0, 2, 1)
        .contiguous()
        .view(-1, num_channels)
    )
    x = (
        F.softmax(x, dim=1)
        .view(batch_size, size, num_channels)
        .permute(0, 2, 1)
    )
    return x.contiguous()


def softmax2d(inputs: 'Tensor') -> 'Tensor':
    """ Apply softmax along channels dimension to batch of 2D images.

    Parameters
    ----------
    inputs : Tensor
        4D input tensor. First dimension is considered to be associated with
        batch items, second with channels.

    Returns
    -------
    Tensor
        tensor containing result of softmax operation.
    """
    batch_size, num_channels = int(inputs.size(0)), int(inputs.size(1))
    shape = (int(inputs.size(2)), int(inputs.size(3)))
    x = (
        inputs
        .permute(0, 2, 3, 1)
        .contiguous()
        .view(-1, num_channels)
    )
    x = (
        F.softmax(x, dim=1)
        .view(batch_size, *shape, num_channels)
        .permute(0, 3, 1, 2)
    )
    return x.contiguous()


def softmax3d(inputs: 'Tensor') -> 'Tensor':
    """ Apply softmax along channels dimension to batch of 3D images.

    Parameters
    ----------
    inputs : Tensor
        5D input tensor. First dimension is considered to be associated with
        batch items, second with channels.

    Returns
    -------
    Tensor
        tensor containing result of softmax operation.
    """
    batch_size, num_channels = int(inputs.size(0)), int(inputs.size(1))
    shape = (int(inputs.size(2)), int(inputs.size(3)), int(inputs.size(4)))
    x = (
        inputs
        .permute(0, 2, 3, 4, 1)
        .contiguous()
        .view(-1, num_channels)
    )
    x = (
        F.softmax(x, dim=1)
        .view(batch_size, *shape, num_channels)
        .permute(0, 4, 1, 2, 3)
    )
    return x.contiguous()


def softmax(inputs: 'Tensor') -> 'Tensor':
    """ Apply softmax to input ndimage represented by torch Tensor. """
    if len(inputs.size()) == 2:
        return F.softmax(inputs, dim=1)
    elif len(inputs.size()) == 3:
        return softmax1d(inputs)
    elif len(inputs.size()) == 4:
        return softmax2d(inputs)
    elif len(inputs.size()) == 5:
        return softmax3d(inputs)


class Softmax(torch.nn.Module):
    """ Softmax activation function generalized for 1D, 2D, 3D images. """

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for Softmax activation function.

        Parameters
        ----------
        inputs : Tensor
            input tensor.

        Returns
        -------
        Tensor
        """
        return softmax(inputs)


class Softmin(torch.nn.Module):
    """ Softmin activation function generalized for 1D, 2D, 3D images. """

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for Softmin activation function.

        Parameters
        ----------
        inputs : Tensor
            input Tensor

        Returns
        -------
        Tensor
        """
        return softmax(-inputs)
    

class Swish(torch.nn.Module):
    
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return x * F.sigmoid(x, inplace=self.inplace)
    

class HSwish(torch.nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class HSigmoid(torch.nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


@ConvBlock.register_option(name='a')
class Activation(Layer):
    """ Generalized activation layer. """

    def __init__(self, input_shape, activation='relu', alpha=1.0, beta=1.0, inplace=True,
                 init=0.1, negative_slope=0.01, num_parameters=1, **kwargs):
        """ Generalized activation layer.

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
        super().__init__(input_shape)
        if not (isinstance(activation, str) or activation is None):
            raise ValueError("Argument 'activation' must have "
                             + "type 'str' or be None.")
        activation = 'linear' if activation is None else activation
        activation = activation.lower()
        activation = activation.strip()

        dim = kwargs.get('dim', None)
        self.params = []
        if activation == 'relu':
            self.layer = torch.nn.ReLU(inplace)
        elif activation == 'relu6':
            self.layer = torch.nn.ReLU6(inplace)
        elif activation == 'sigmoid':
            self.layer = torch.nn.Sigmoid()
        elif activation == 'tanh':
            self.layer = torch.nn.Tanh()
            self.params = []
        elif activation == 'leaky_relu':
            self.layer = torch.nn.LeakyReLU(negative_slope, inplace)
            self.params = [float(negative_slope)]
        elif activation == 'elu':
            self.layer = torch.nn.ELU(alpha, inplace)
            self.params = [float(alpha)]
        elif activation == 'selu':
            self.layer = torch.nn.SELU(inplace)
            self.params = [1.6732, 1.0507]
        elif activation == 'prelu':
            self.layer = torch.nn.PReLU(num_parameters, init)
            self.params = []
        elif activation == 'softmax':
            self.layer = torch.nn.Softmax(1)
        elif activation == 'softmin':
            self.layer = torch.nn.Softmin(1)
        elif activation == 'softplus':
            self.layer = torch.nn.Softplus(beta,
                                           kwargs.get('threshold'))
        elif activation == 'softsign':
            self.layer = torch.nn.Softsign()
        elif activation == 'swish':
            self.layer = Swish(inplace)
        elif activation == 'hsigmoid':
            self.layer = HSigmoid(inplace)
        elif activation == 'hswish':
            self.layer = HSwish(inplace)
        elif activation == 'linear':
            self.layer = None
            self.params = [1.0, 0.0]
        else:
            raise ValueError("Argument 'activation' must be one "
                             + "of following values: 'relu', 'leaky_relu', "
                             + "'sigmoid', 'linear', 'elu' or None.")
        self.activation = activation

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method.

        Parameters
        ----------
        inputs : Tensor
            input tensor.

        Returns
        -------
        Tensor
            result of activaton function application.
        """
        if self.layer is None:
            return inputs
        return self.layer.forward(inputs)
