import numpy as np
import torch
from torch.autograd import Variable
from ..config import Config
from ..layers import ConvBlock


class BaseModel(torch.nn.Module):
    """ Base class for all models

    Attributes
    ----------
    name : str
        a model name
    config : dict
        configuration parameters

    Notes
    -----

    **Configuration**:

    * build : bool
        whether to build a model by calling `self.build()`. Default is True.
    * load : dict
        parameters for model loading. If present, a model will be loaded
        by calling `self.load(**config['load'])`.

    """

    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        self.config = self.default_config() @ Config(config)
        self.data_format = 'channels_first'
        self._device = None
        self.input_shape = self.config.get('input_shape')
        if 'conv_block' in self.config:
            self.conv_block = ConvBlock.partial(**self.config['conv_block'])
        else:
            self.conv_block = ConvBlock

        self.input = None
        self.body = None
        self.head = None
        if self.config.get('build', default=True):
            self.build(*args, **kwargs)

    @property
    def encoders(self):
        return self.body

    def count_parameters(self) -> int:
        """ Count number of models parameters. """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def cuda(self, device: int = 0) -> 'PretrainedModel':
        """ Put model on cuda device.

        Parameters
        ----------
        device : int
            id of cuda device. Default is 0.

        Returns
        -------
        PretrainedModel
        """
        model = super().cuda(device)
        model._device = int(device)
        return model

    def cpu(self) -> 'PretrainedModel':
        """ Put model on cpu.

        Returns
        -------
        PretrainedModel
        """
        model = super().cpu()
        model._device = None
        return model

    def dummy_batch(self, batch_size: int = 2) -> 'Tensor':
        """ Get dummy input batch for the mdoel.

        Parameters
        ----------
        batch_size : int
            size of dummy batch.

        Returns
        -------
        Tensor
        """
        x = torch.rand(batch_size, *[int(d) for d in self.input_shape])
        if self._device is not None:
            return Variable(x.cuda(int(self._device)), requires_grad=False)
        else:
            return Variable(x, requires_grad=False)

    @classmethod
    def disable_grad(cls, model: 'Module') -> 'Module':
        """ Set all parameters of input pytorch model as non-requiring gradient.

        Parameters
        ----------
        model : torch.nn.Module
            input model represented by pytorch Module.

        Returns
        -------
        torch.nn.Module
            input model with all parameters marked as non-requiring gradient
            computation.
        """
        for param in model.parameters():
            param.requires_grad = False
        return model

    @classmethod
    def enable_grad(cls, model: 'Module') -> 'Module':  # noqa: N805
        """ Set all parameters of input pytorch model as requiring gradient.

        Parameters
        ----------
        model : torch.nn.Module
            input model represented by pytorch Module.

        Returns
        -------
        torch.nn.Module
            input model with all parameters marked as requiring gradient
            computation.
        """
        for param in model.parameters():
            param.requires_grad = True
        return model

    @classmethod
    def as_torch(cls, data, dtype='float32', device=None, grad=False, async=False):
        if isinstance(data, np.ndarray):
            x = torch.from_numpy(data)
        elif torch.is_tensor(data):
            x = data
        elif isinstance(data, Variable):
            x = data.data

        if dtype == 'float16':
            x = x.type(torch.HalfTensor)
        elif dtype == 'float32':
            x = x.type(torch.FloatTensor)
        elif dtype == 'float64' or dtype == 'float':
            x = x.type(torch.DoubleTensor)

        elif dtype == 'int8':
            x = x.type(torch.ByteTensor)
        elif dtype == 'int16':
            x = x.type(torch.ShortTensor)
        elif dtype == 'int32':
            x = x.type(torch.IntTensor)
        elif dtype == 'int64' or dtype == 'int':
            x = x.type(torch.LongTensor)
        else:
            raise ValueError("Argument 'dtype' must be str.")

        if device is None:
            return Variable(x, requires_grad=grad)
        elif isinstance(device, int):
            return Variable(x.cuda(device=device, async=async), requires_grad=grad)

    @classmethod
    def as_numpy(cls, data, dtype='float32'):
        if isinstance(data, np.ndarray):
            return data.astype(dtype)
        elif torch.is_tensor(data):
            x = data
        elif isinstance(data, Variable):
            x = data.data
        return x.cpu().numpy()

    @property
    def default_name(self):
        """: str - the class name (serve as a default for a model name) """
        return self.__class__.__name__

    @classmethod
    def default_config(cls):
        """ Define model defaults. """
        return Config({
            'conv_block': {},
            'input': {},
            'body': {},
            'head': {}
        })

    def build_input(self, input_shape, config=None, **kwargs):
        if config:
            return self.conv_block(input_shape=input_shape, **config)
        return None

    def build_head(self, input_shape, config=None, **kwargs):
        if config:
            return self.conv_block(input_shape=input_shape, **config)
        return None

    def build_body(self, input_shape, config=None, **kwargs):
        if config:
            return self.conv_block(input_shape=input_shape, **config)
        return None

    def build(self, *args, **kwargs):
        config = self.config
        input_shape = config.get('input_shape')
        self.input = self.build_input(input_shape=input_shape,
                                      config=config.get('input'))
        if self.input is not None:
            input_shape = self.input.output_shape

        self.body = self.build_body(input_shape=input_shape,
                                    config=config.get('body'))
        if self.body is not None:
            input_shape = self.body.output_shape

        self.head = self.build_head(input_shape=input_shape,
                                    config=config.get('head'))
        if self.head is not None:
            input_shape = self.head.output_shape

        self.output_shape = input_shape

    @classmethod
    def load(cls, path, **kwargs):
        """ Load the model """
        return torch.load(path)

    def save(self, path, **kwargs):
        """ Save the model """
        torch.save(self, path)

    def save_onnx(self, path: str):
        """ Save model in onnx format.

        Parameters
        ----------
        path : str
            path to file where onnx model will be saved.
        """
        torch.onnx.export(self, self.dummy_batch(), path)

    def save_coreml(self, path: str,
                    image_inputs: list = None,
                    verbose: bool = False):
        """ Save model in coreml format.

        Parameters
        ----------
        path : str
            path file where coreml model will be saved.
        image_inputs : List[str]
            list containing names of nodes that
            corespond to input images. Default is None
            meaning that empty list will be passed.
            Another common value is ["0"] because
            commonly model accepts only one image.
        verbose : bool
            whether to print info about onnx => coreml convertation
            progress or not. Default is False meaning that no info will
            be showed.
        """
        self.as_coreml(image_inputs=image_inputs,
                       verbose=verbose).save(path)

    def as_coreml(self, image_inputs: list = None, verbose: bool = False):
        """ Convert model to coreml model.

        Parameters
        ----------
        image_inputs : List[str]
            list containing names of nodes that
            corespond to input images. Default is None
            meaning that empty list will be passed.
            Another common value is ["0"] because
            commonly model accepts only one image.
        verbose : bool
            whether to print info about onnx => coreml
            convertation progress or not. Default is False
            meaning that no info will be showed.

        Returns
        -------
        MLModel
            coreml MLModel
        """
        import tempfile
        import onnx
        import onnx_coreml
        from coremltools.models import MLModel

        image_inputs = [] if image_inputs is None else ["0"]
        with tempfile.NamedTemporaryFile() as file:
            torch.onnx.export(self, self.dummy_batch(), file.name)
            mlmodel = MLModel(onnx_coreml.convert(onnx.load(file),
                                                  image_input_names=image_inputs).get_spec())
        return mlmodel

    def forward_encoders(self, inputs: 'Tensor'):
        x = self.input(inputs) if self.input else inputs
        outputs = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            outputs.append(x)
        return outputs

    def forward(self, inputs):
        x = self.input(inputs) if self.input is not None else inputs
        x = self.body(x) if self.body is not None else x
        x = self.head(x) if self.head is not None else x
        return x
