import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from ..layers import ConvBlock
from ..bases import Module
from ..bases import Sequential
from .res_block import BaseResBlock
from ..utils import transform_to_int_tuple
from ..config import Config


def softmax_2d(x):
    x = x.permute(1, 0)
    xmax, _ = torch.max(x, 0)
    return (torch.exp(x - xmax) / torch.sum(torch.exp(x - xmax), 0)).permute(1, 0)


def iou_torch(x: 'Tensor(..., k, 4)', y: 'Tensor(..., l, 4)') -> 'Tensor(..., k, l)':
    """ Compute intersection over union values for two sets of bounding boxes. """
    lb = torch.max(torch.unsqueeze(x[..., :, :2], -2),
                   torch.unsqueeze(y[..., :, :2], -3))

    ub = torch.min(torch.unsqueeze(x[..., :, 2:], -2),
                   torch.unsqueeze(y[..., :, 2:], -3))

    inter = torch.prod(torch.clamp(ub - lb, 0), -1)

    area_x = torch.unsqueeze((x[..., :, 2] - x[..., :, 0])
                             * (x[..., :, 3] - x[..., :, 1]), -1)
    area_y = torch.unsqueeze((y[..., :, 2] - y[..., :, 0])
                             * (y[..., :, 3] - y[..., :, 1]), -2)
    return inter / (area_x + area_y - inter)


def nms_torch(bboxes: 'Tensor',
              scores: 'Tensor',
              threshold: float = 0.5) -> 'Tensor':
    """ Non maximum suppression.

    Parameters
    ----------
    bboxes: Tensor
        bounding boxes, sized[N, 4].
    scores: Tensor
        bbox scores, sized[N, ].
    threshold: float
        overlap threshold.

    Returns
    -------
    keep: Tensor
        selected indices.

    Note
    ----
    Inspired by https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 2], bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=float(x1[i]))
        yy1 = y1[order[1:]].clamp(min=float(y1[i]))
        xx2 = x2[order[1:]].clamp(max=float(x2[i]))
        yy2 = y2[order[1:]].clamp(max=float(y2[i]))

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)

        ids = (overlap <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


def encode(bboxes: 'Tensor(l, 4)',
           priors: 'Tensor(l, 4)',
           labels: 'Tensor(l)',
           variance: tuple = (0.1, 0.2),
           iou_threshold: float = 0.5) -> 'Tuple[Tensor(n, 4), Tensor(n)]':
    priors_centers = 0.5 * (priors[:, :2] + priors[:, 2:])
    priors_sizes = priors[:, 2:] - priors[:, :2]

    iou = iou_torch(bboxes, priors)
    iou_max, idx_max = iou.max(0)
    idx_max.squeeze_(0)
    iou_max.squeeze_(0)

    bboxes = bboxes[idx_max, :]

    cxcy = 0.5 * (bboxes[:, :2] + bboxes[:, 2:]) - priors_centers
    cxcy /= variance[0] * priors_sizes

    wh = (bboxes[:, 2:] - bboxes[:, :2]) / prior_sizes
    wh = torch.log(wh + 10e-7) / variance[1]

    return (torch.cat([cxcy, wh], 1),
            torch.where(iou_max < iou_threshold,
                        0, 1 + labels[idx_max]))


def decode(loc: 'Tensor(n, 4)',
           priors: 'Tensor(k, 4)',
           conf: 'Tensor(n, l)',
           variance: tuple = (0.1, 0.2),
           iou_threshold: float = 0.5) -> 'Tuple[Tensor(n, 4), Tensor(n)]':
    priors_centers = 0.5 * (priors[:, :2] + priors[:, 2:])
    priors_sizes = priors[:, 2:] - priors[:, :2]

    wh = torch.exp(loc[:, 2:] * variance[1]) * priors_sizes
    cxcy = loc[:, :2] * variance[0] * priors_sizes + priors_centers
    bboxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)

    conf = F.Softmax(conf)
    max_conf, max_labels = conf.max(1)
    ids = labels.nonzero()
    if len(ids) == 0:
        return None, None
    else:
        ids = ids.squeeze(1)

    keep = nms_torch(bboxes[ids, :], max_conf[ids],
                     threshold=iou_threshold)
    return bboxes[ids][keep], conf[ids][keep]


class BaseDetectionEncoder(torch.nn.Module):

    def __init__(self, priors: 'Tensor', variance: tuple = (0.1, 0.2)):
        super().__init__()
        self.priors = torch.nn.Parameter(priors, requires_grad=False)
        self.variance = [float(v) for v in variance]

    @property
    def prior_centers(self) -> 'Tensor':
        """ Get prior bounding boxes centers. """
        return 0.5 * (self.priors[:, 2:] + self.priors[:, :2])

    @property
    def prior_sizes(self) -> 'Tensor':
        """ Get prior bounding boxes sizes. """
        return self.priors[:, 2:] - self.priors[:, :2]

    @classmethod
    def stack_encoders(cls, encoders):
        return cls(torch.cat([encoder.priors for encoder in encoders], 0))

    @classmethod
    def iou(cls, x: 'Tensor', y: 'Tensor') -> 'Tensor':
        return iou_torch(x, y)

    @classmethod
    def nms_torch(cls, bboxes: 'Tensor',
                  scores: 'Tensor',
                  threshold: float = 0.5) -> 'Tensor':
        return nms_torch(bboxes, scores, threshold)

    @classmethod
    def generate_anchors(cls,
                         fmap_size: 'Tuple[int]',
                         tile_size: 'Tuple[int]',
                         stride: 'Tuple[int]',
                         aspect_ratios: 'Tuple[int]',
                         scales: 'Tuple[int]'):

        if isinstance(fmap_size, (list, tuple, np.ndarray)):
            fmap_size = np.array(fmap_size, 'int64').flatten()
        else:
            fmap_size = np.array([int(fmap_size)] * 2, 'int64')

        if isinstance(tile_size, (list, tuple, np.ndarray)):
            tile_size = np.array(tile_size, 'int64').flatten()
        else:
            tile_size = np.array([int(tile_size)] * 2, 'int64')

        if isinstance(stride, (list, tuple, np.ndarray)):
            stride = np.array(stride, 'int64').flatten()
        else:
            stride = np.array([int(stride)] * 2, 'int64')

        if isinstance(aspect_ratios, (list, tuple, np.ndarray)):
            aspect_ratios = np.array(aspect_ratios, 'float')
        else:
            aspect_ratios = np.array([float(aspect_ratios)], 'float')

        if isinstance(scales, (list, tuple, np.ndarray)):
            scales = np.array(scales, 'float')
        else:
            scales = np.array([float(scales)], 'float')

        gx, gy, gr, gs = np.meshgrid(np.arange(fmap_size[0]),
                                     np.arange(fmap_size[1]),
                                     np.arange(len(aspect_ratios)),
                                     np.arange(len(scales)),
                                     indexing='ij')

        z = np.zeros(shape=(*fmap_size,
                            len(aspect_ratios),
                            len(scales), 4))

        z[gx, gy, gr, gs, 0] = ((gx + 0.5) * stride[0]
                                - 0.5 * tile_size[0]
                                      * scales[gs]
                                      * np.sqrt(aspect_ratios[gr]))

        z[gx, gy, gr, gs, 1] = ((gy + 0.5) * stride[1]
                                - 0.5 * tile_size[1]
                                      * scales[gs]
                                      * np.sqrt(1.0 / aspect_ratios[gr]))

        z[gx, gy, gr, gs, 2] = ((gx + 0.5) * stride[0]
                                + 0.5 * tile_size[0]
                                      * scales[gs]
                                      * np.sqrt(aspect_ratios[gr]))

        z[gx, gy, gr, gs, 3] = ((gy + 0.5) * stride[1]
                                + 0.5 * tile_size[1]
                                      * scales[gs]
                                      * np.sqrt(1.0 / aspect_ratios[gr]))

        return z.reshape(-1, 4)

    def encode(self, bboxes: 'Tensor',
               classes: 'Tensor',
               threshold: float = 0.5):
        iou = self.iou(bboxes, self.priors)

        iou, max_idx = iou.max(0)
        max_idx.squeeze_(0)
        iou.squeeze_(0)

        bboxes = bboxes[max_idx]

        cxcy = 0.5 * (bboxes[:, :2] + bboxes[:, 2:]) - self.prior_centers
        cxcy /= self.variance[0] * self.prior_sizes

        wh = (bboxes[:, 2:] - bboxes[:, :2]) / self.prior_sizes
        wh = torch.log(wh + 10e-7) / self.variance[1]
        loc = torch.cat([cxcy, wh], 1)

        conf = 1 + classes[max_idx]
        conf[iou < threshold] = 0
        return loc, conf

    def decode(self,
               loc: 'Tensor',
               conf: 'Tensor',
               threshold: float = 0.5):
        var_0 = self.variance[0]
        var_1 = self.variance[1]

        wh = torch.exp(loc[:, 2:] * var_1) * self.prior_sizes
        cxcy = loc[:, :2] * var_0 * self.prior_sizes + self.prior_centers
        bboxes = torch.cat([cxcy - wh / 2,
                            cxcy + wh / 2], 1)

        conf = softmax_2d(conf)
        max_conf, labels = conf.max(1)
        ids = labels.nonzero()
        if len(ids) == 0:
            return None, None
        else:
            ids = ids.squeeze(1)
        keep = self.nms(bboxes[ids, :],
                        max_conf[ids],
                        threshold=threshold)
        return bboxes[ids][keep], conf[ids][keep]


class DetectionEncoder(BaseDetectionEncoder):

    @classmethod
    def default_config(cls):
        return Config({
            'input_shape': (None, 512, 512),
            'num_levels': 5,
            'tile_size': [32, 64, 128, 256, 512],
            'fmap_size': [64, 32, 16, 8, 4],
            'stride': [8, 16, 32, 64, 128],
            'scales': [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
            'aspect_ratios': [1.0 / 2.0, 1.0, 2.0],
            'variance': [0.1, 0.2],
            'normalize': True
        })

    @classmethod
    def _norm_param(cls, param, dtype=int, num_levels=1, is_spatial=False):
        if not isinstance(param, (tuple, list, np.ndarray)):
            if is_spatial:
                param = np.array([(param, param)], dtype=dtype)
            else:
                param = np.array([param], dtype=dtype)
        else:
            param = np.array(param, dtype=dtype)
        if param.ndim == 1 and not isinstance(param[0], (tuple, list, np.ndarray)):
            if is_spatial:
                param = np.tile(param[:, np.newaxis], [1, 2])
            else:
                param = np.tile(param[np.newaxis, :],
                                [num_levels, 1])
        if param.shape[0] != num_levels:
            raise ValueError("First dimension of array-like"
                             + " parameter '{}'".format(param.shape[0])
                             + " must be equal to number of levels '{}'".format(num_levels))
        return param

    def __init__(self, **config: dict):
        config = self.default_config() @ Config(config)

        input_shape = config['input_shape']

        num_levels = config.get('num_levels')
        aspect_ratios = np.array(config.get('aspect_ratios'))
        scales = np.array(config.get('scales'))

        if not isinstance(aspect_ratios[0], (tuple, list, np.ndarray)):
            aspect_ratios = np.tile(aspect_ratios[np.newaxis, :],
                                    [num_levels, 1])
        if not isinstance(scales[0], (tuple, list, np.ndarray)):
            scales = np.tile(scales[np.newaxis, :],
                             [num_levels, 1])

        stride = self._norm_param(config.get('stride', 8),
                                  int, num_levels, True)
        fmap_size = self._norm_param(config.get('fmap_size', 64),
                                     int, num_levels, True)

        tile_size = self._norm_param(config.get('tile_size', 0),
                                     int, num_levels, True)

        if np.all(fmap_size == fmap_size[0, :]):
            fmap_size = np.ceil(fmap_size.T / 2 ** np.arange(num_levels)).T
            fmap_size = fmap_size.astype(np.int)

        if np.all(stride == stride[0, :]):
            stride = np.ceil(stride.T * 2 ** np.arange(num_levels)).T
            stride = stride.astype(np.int)

        if np.all(tile_size == tile_size[0, :]):
            tile_size = np.ceil(tile_size.T * 2 ** np.arange(num_levels)).T
            tile_size = tile_size.astype(np.int)

        variance = config['variance']

        self.input_shape = input_shape
        self.fmap_size = fmap_size
        self.tile_size = tile_size
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.stride = stride

        priors = []
        for i in range(num_levels):
            x = self.generate_anchors(fmap_size[i, :],
                                      tile_size[i, :],
                                      stride[i, :],
                                      aspect_ratios[i],
                                      scales[i])
            priors.append(x)
        priors = np.concatenate(priors)
        if config.get('normalize', True):
            priors[:, [0, 2]] /= self.input_shape[1]
            priors[:, [1, 3]] /= self.input_shape[2]
        super().__init__(Variable(torch.from_numpy(priors)),
                         list(variance))


class BaseLocClfPredictor(Module):

    @classmethod
    def default_config(cls):
        return Config({
            'loc': {
                'layout': 'ca',
                'c': {
                    'kernel_size': 3,
                    'bias': True
                },
                'a': {
                    'activation': 'relu'
                }
            },
            'clf': {
                'layout': 'ca',
                'c': {
                    'kernel_size': 3,
                    'bias': True
                },
                'a': {
                    'activation': 'sigmoid'
                }
            }
        })

    def __init__(self,
                 input_shape,
                 num_classes: int,
                 num_anchors: int,
                 body_block=None,
                 config: dict = None):
        super().__init__(input_shape)
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)

        self.config = self.default_config() @ Config(config) @ {
            'loc/c/filters': 4 * self.num_anchors,
            'clf/c/filters': (self.num_classes
                              * self.num_anchors)
        }

        if body_block is not None:
            self.body = body_block(self.input_shape,
                                   **self.config.get('body', {}))
            shape = self.body.output_shape
        else:
            self.body = None
            shape = self.input_shape

        self.loc_head = ConvBlock(
            input_shape=shape,
            **self.config['loc']
        )
        self.clf_head = ConvBlock(
            input_shape=shape,
            **self.config['clf']
        )

    @property
    def output_shape(self):
        return np.stack([self.loc_head.output_shape,
                         self.clf_head.output_shape], axis=0)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        if self.body is None:
            x = inputs
        else:
            x = self.body(inputs)
        return (
            self.loc_head(x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size, -1, 4),

            self.clf_head(x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size, -1,
                  self.num_classes)
        )


class SimpleLocClfPredictor(BaseLocClfPredictor):

    @classmethod
    def default_config(cls):
        return Config({
            'body': {
                'layout': 'ca ca ca ca',
                'c': {
                    'kernel_size': 3,
                    'bias': True,
                    'filters': [256, 256, 256, 256]
                },
                'a': {
                    'activation': 'relu'
                }
            },
            **BaseLocClfPredictor.default_config()
        })

    def __init__(self,
                 input_shape,
                 num_classes: int,
                 num_anchors: int,
                 config: dict = None):
        super(SimpleLocClfPredictor, self).__init__(input_shape,
                                                    num_classes,
                                                    num_anchors,
                                                    ConvBlock,
                                                    config)


class ResLocClfPredictor(BaseLocClfPredictor):

    @classmethod
    def default_config(cls):
        return Config({
            'body': {
                'num_blocks': 2,
                'filters': [1024, 256, 256],
                'layout': 'cna cna cn',
                'shortcut_layout': 'cn',
                'kernel_size': 3,
                'post_activation': True
            },
            **BaseLocClfPredictor.default_config()
        })

    @classmethod
    def body_block(cls, input_shape, **config):
        layout, filters = config['layout'], config['filters']
        kernel_size = config['kernel_size']
        shortcut_layout = config['shortcut_layout']
        post_activation = config['post_activation']

        num_conv = layout.lower().count('c')
        kernel_size = transform_to_int_tuple(kernel_size,
                                             'kernel_size',
                                             num_conv)

        filters = transform_to_int_tuple(filters,
                                         'filters',
                                         num_conv)

        body = ConvBlock.partial(
            layout=layout,
            c=dict(kernel_size=kernel_size,
                   filters=filters)
        )

        shortcut = ConvBlock.partial(
            layout=shortcut_layout,
            c=dict(kernel_size=1, stride=1,
                   filters=filters[-1])
        )

        if post_activation:
            head = ConvBlock.partial(layout='an')
        else:
            head = None

        layers = Sequential()
        shape = input_shape
        for i in range(config['num_blocks']):
            iblock = BaseResBlock(shape, body, shortcut, head, '+')
            layers.add_module('ResBlock_{}'.format(i), iblock)
            shape = iblock.output_shape
        return layers

    def __init__(self,
                 input_shape,
                 num_classes: int,
                 num_anchors: int,
                 config: dict = None):
        super(ResLocClfPredictor, self).__init__(input_shape,
                                                 num_classes, num_anchors,
                                                 self.body_block, config)
