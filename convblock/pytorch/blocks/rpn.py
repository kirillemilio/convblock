import numpy as np
import torch
import torch.nn.functional as F
from ..bases import Module
from ..layers import ConvBlock


class RPN(Module):

    @classmethod
    def generate_anchors(cls, image_shape, fmap_size, tile_size,
                         stride, ratios=None, scales=None):

        if ratios is None:
            ratios = 1.0
        if scales is None:
            scales = 1.0

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

        if isinstance(ratios, (list, tuple, np.ndarray)):
            ratios = np.array(ratios, 'float')
        else:
            ratios = np.array([float(ratios)], 'float')

        if isinstance(scales, (list, tuple, np.ndarray)):
            scales = np.array(scales, 'float')
        else:
            scales = np.array([float(scales)], 'float')

        gx, gy, gr, gs = np.meshgrid(np.arange(fmap_size[0]),
                                     np.arange(fmap_size[1]),
                                     np.arange(len(ratios)),
                                     np.arange(len(scales)),
                                     indexing='ij')

        z = np.zeros(shape=(*fmap_size,
                            len(ratios),
                            len(scales), 4))

        z[gx, gy, gr, gs, 0] = ((gx + 0.5) * stride[0]
                                - 0.5 * tile_size[0]
                                      * scales[gs]
                                      * np.sqrt(ratios[gr]))

        z[gx, gy, gr, gs, 1] = ((gy + 0.5) * stride[1]
                                - 0.5 * tile_size[1]
                                      * scales[gs]
                                      * np.sqrt(1.0 / ratios[gr]))

        z[gx, gy, gr, gs, 2] = ((gx + 0.5) * stride[0]
                                + 0.5 * tile_size[0]
                                      * scales[gs]
                                      * np.sqrt(ratios[gr]))

        z[gx, gy, gr, gs, 3] = ((gy + 0.5) * stride[1]
                                + 0.5 * tile_size[1]
                                      * scales[gs]
                                      * np.sqrt(1.0 / ratios[gr]))

        z = z.reshape(-1, 4)
        keep = ((z[:, 0] >= -allowed_border) &
                (z[:, 1] >= -allowed_border) &
                (z[:, 2] < image_shape[0] + allowed_border) &
                (z[:, 3] < image_shape[1] + allowed_border))
        return z[keep, :]

    @classmethod
    def iou(cls, x: 'Tensor(..., k, 4)', y: 'Tensor(..., l, 4)') -> 'Tensor(..., k, l)':
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

    @classmethod
    def nms(cls,
            bboxes: 'Tensor(l, 4)',
            scores: 'Tensor(l, )',
            threshold: float = 0.5) -> 'Tensor(k)':
        """ Non-maximum suppression implemented on pure pytorch. """
        areas = torch.prod(bboxes[:, 2:] - bboxes[:, :2], 1)
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break
            _bboxes = bboxes[order, :]
            xx1 = _bboxes[1:, 0].clamp(min=float(_bboxes[0, 0]))
            yy1 = _bboxes[1:, 1].clamp(min=float(_bboxes[0, 1]))
            xx2 = _bboxes[1:, 2].clamp(max=float(_bboxes[0, 2]))
            yy2 = _bboxes[1:, 3].clamp(max=float(_bboxes[0, 3]))

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            overlap = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (overlap <= threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        return torch.LongTensor(keep)

    def __init__(self,
                 input_shape,
                 image_shape,
                 filters,
                 stride,
                 tile_size,
                 ratios=(0.5, 1.0, 2.0),
                 scales=2**np.arange(3, 6),
                 nms_threshold=0.5,
                 pre_nms_top_n=100,
                 post_nms_top_n=100,
                 **kwargs):

        super().__init__(input_shape)

        self.pre_nms_top_n = int(pre_nms_top_n)
        self.post_nms_top_n = int(post_nms_top_n)
        self.nms_threshold = float(nms_threshold)
        self.scales = scales
        self.ratios = ratios

        self.anchors = self.generate_anchors(self.input_shape[1:],
                                             tile_size, stride,
                                             ratios, scales)

        self.rpn_conv = ConvBlock(
            input_shape=input_shape, layout='ca',
            c=dict(filters=filters, kernel_size=3, bias=True)
        )

        self.rpn_clf = ConvBlock(
            input_shape=self.rpn_conv.output_shape, layout='c',
            c=dict(filters=len(ratios) * len(scales) * 2,
                   kernel_size=1, bias=True)
        )

        self.rpn_reg = ConvBlock(
            input_shape=self.rpn_conv.output_shape, layout='c',
            c=dict(filters=len(ratios) * len(scales) * 4,
                   kernel_size=1, bias=True)
        )

    @classmethod
    def encode_bboxes(cls, ex_rois, gt_rois):
        ex_sizes = ex_rois[..., 2:] - ex_rois[..., :2]
        ex_centers = 0.5 * (ex_rois[..., 2:] + ex_rois[..., :2])

        gt_sizes = gt_rois[..., 2:] - gt_rois[..., :2]
        gt_centers = 0.5 * (gt_rois[..., 2:] + gt_rois[..., :2])

        return torch.cat([(gt_centers - ex_centers) / ex_sizes,
                          torch.log(gt_sizes / ex_sizes)], 1)

    @classmethod
    def decode_bboxes(cls, bboxes: 'Tensor(l, 4)', deltas: 'Tensor(n, l, 4)'):
        bboxes = bboxes.unsqueeze(1)

        bbox_sizes = bboxes[..., 2:] - bboxes[..., :2]
        bbox_centers = 0.5 * (bboxes[..., 2:] + bboxes[..., :2])

        pred_centers = deltas[..., :2] * bbox_sizes + bbox_centers
        pred_sizes = torch.exp(deltas[..., 2:]) * bbox_sizes

        pred_bboxes = deltas.clone()
        pred_bboxes[..., :2] = pred_centers - 0.5 * pred_sizes
        pred_bboxes[..., 2:] = pred_centers + 0.5 * pred_sizes
        return pred_bboxes

    def _extract_rois(self, proposals, scores) -> 'Tensor(l, 5)':
        batch_size = scores.size(0)
        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        rois = scores.new(batch_size, self.post_nms_top_n, 5).zero_()
        for i in range(batch_size):

            if self.pre_nms_top_n > 0 and self.pre_nms_top_n < scores_keep.numel():
                iorder = order[i, :self.pre_nms_top_n]
            else:
                iorder = order[i, :]

            iproposals = proposals_keep[i, iorder, :]
            iscores = scores_keep[i, iorder].view(-1, 1)
            ikeep_idx = self.nms(iproposals, iscores,
                                 self.nms_threshold).long().view(-1)

            if self.post_nms_top_n > 0:
                ikeep_idx = ikeep_idx[:self.post_nms_top_n]

            iproposals, iscores = iproposals[ikeep_idx,
                                             :], iscores[ikeep_idx, :]
            rois[i, :, 0], rois[i, :iproposals.size(0), 1:] = i, iproposals
        return rois

    def forward(self, inputs, bboxes):
        batch_size = inputs.size(0)
        inputs = self.rpn_conv(inputs)
        clf_output = self.rpn_clf(inputs)
        clf_output = F.softmax(clf_output.view(batch_size, 2,
                                               clf_output.size(1) // 2,
                                               clf_output.size(2),
                                               clf_output.size(3)), 1)

        clf_output = (
            clf_output
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(batch_size, 2, -1)
        )

        reg_output = (
            self.rpn_reg(inputs)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size, -1, 4)
        )

        scores = clf_output[:, 1, :]
        proposals = self.decode_bboxes(self.anchors, reg_output)
        # Clip proposals to be inside input images
        # May be image_shape 1, 2 must be changed over
        proposals[:, :, [0, 2]].clamp_(0, self.image_shape[1] - 1)
        proposals[:, :, [1, 3]].clamp_(0, self.image_shape[2] - 1)

        # Extract meaningful regions of interest bboxes
        rois = self._extract_rois(proposals, scores)

        if self.training:
            # Extract patches corresponding to rois
            labels = bboxes.new(batch_size, self.anchors.size(0)).fill_(-1)
            overlap = self.iou(self.anchors, bboxes)
