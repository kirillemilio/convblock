""" Contain different metrics functions. """


import torch as t
from torch.autograd import Variable
import numpy as np
try:
    from torch import device as torch_device
except ImportError:
    torch_device = int


def as_torch(data, dtype='float32',
             device=None, grad=False, async=False, **kwargs):
    """ Transform input data to torch Variable.

    Parameters
    ----------
    data : ndarray, torch Tensor or Variable
        input data.
    dtype : str
        data type.
    device : int or None
        gpu device. If None then cpu will be used.
    grad : bool
        whether data tensor require gradient computation.
    async : bool
        whether to enabl async mode.

    Returns
    -------
    Variable
    """
    if isinstance(data, np.ndarray):
        x = t.from_numpy(data)
    elif t.is_tensor(data):
        x = data
    elif isinstance(data, Variable):
        x = data.data

    if dtype == 'float16' or dtype == 'torch.float16':
        x = x.type(t.HalfTensor)
    elif dtype == 'float32' or dtype == 'torch.float32':
        x = x.type(t.FloatTensor)
    elif dtype == 'float64' or dtype == 'float' or dtype == 'torch.float64':
        x = x.type(t.DoubleTensor)

    elif dtype == 'int8' or dtype == 'torch.int8':
        x = x.type(t.ByteTensor)
    elif dtype == 'int16' or dtype == 'torch.int16':
        x = x.type(t.ShortTensor)
    elif dtype == 'int32' or dtype == 'torch.int32':
        x = x.type(t.IntTensor)
    elif dtype == 'int64' or dtype == 'int' or dtype == 'torch.int64':
        x = x.type(t.LongTensor)
    else:
        raise ValueError("Argument 'dtype' must be str.")

    if device is None:
        return Variable(x, requires_grad=grad, **kwargs)
    elif isinstance(device, (int, torch_device)):
        return Variable(x.cuda(device=device, async=async),
                        requires_grad=grad, **kwargs)


def as_numpy(data, dtype='float32'):
    """ Transform input data to numpy array with given dtype.

    Parameters
    ----------
    data : ndarray, torch Tensor or Variable
        input data.
    dtype : str
        dtype of output numpy array.

    Returns
    -------
    ndarray
    """
    if isinstance(data, np.ndarray):
        return data.astype(dtype)
    elif t.is_tensor(data):
        x = data
    elif isinstance(data, Variable):
        x = data.data
    return x.cpu().numpy()


def _tensor2float(y_true, y_pred):
    y_true = as_torch(y_true)
    y_pred = as_torch(y_pred)
    return y_true, y_pred


def mse(y_true, y_pred):
    """ Compute MSE metric.

    Parameters
    ----------
    y_pred : ndarray(batch_size, ...)
        numpy array containing predictions of model.
    y_true : np.ndarray(batch_size, ...)
        numpy array containing true target values.

    Returns
    -------
    float
        mean square error value.
    """
    y_true, y_pred = _tensor2float(y_true, y_pred)
    return t.mean((y_pred - y_true).float() ** 2)


def rmse(y_true, y_pred):
    """ Compute RMSE metric.

    Parameters
    ----------
    y_true : np.ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : np.ndarray(batch_size, ...)
        numpy array containing predictions of model.

    Returns
    -------
    float
        root mean square error value.
    """
    y_true, y_pred = _tensor2float(y_true, y_pred)
    return t.sqrt(t.mean((y_pred - y_true).float()) ** 2)


def mae(y_true, y_pred):
    """ Compute MAE metric.

    Parameters
    ----------
    y_true : np.ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : np.ndarray(batch_size, ...)
        numpy array containing predictions of model.

    Returns
    -------
    float
        mean average error value.
    """
    y_true, y_pred = _tensor2float(y_true, y_pred)
    return t.mean(t.abs((y_pred - y_true).float()))


def dice(y_true, y_pred, epsilon=10e-7):
    """ Compute Dice coefficient.

    Parameters
    ----------
    y_true : np.ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : np.ndarray(batch_size, ...)
        numpy array containing predictions of model.

    Returns
    -------
    float
        dice coefficient.
    """
    y_true, y_pred = _tensor2float(y_true, y_pred)
    #epsilon = t.tensor([epsilon], device=cuda_device)
    return 2 * t.sum(y_pred * y_true) / (t.sum(y_pred) + t.sum(y_true) + epsilon)


def sym_dice(y_true, y_pred, alpha, epsilon=10e-7):
    """ Symmetric dice coefficient.

    Parameters
    ----------
    y_true : np.ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : np.ndarray(batch_size, ...)
        numpy array containing predictions of model.
    alpha : float
        weight of dice coeffecient computed by '1' class labels.
    epsilon : float
        small real value for avoiding division by zero error.

    Returns
    -------
    float
        symetrized by '0-1' class labels dice coefficient.
    """
    y_true, y_pred = _tensor2float(y_true, y_pred)
    cuda_check = y_pred.is_cuda
    cuda_device = t.device('cuda', y_pred.get_device()) if cuda_check else t.device("cpu")
    alpha = t.tensor([alpha], device=cuda_device)
    return (1 - alpha) * dice(y_pred, y_true, epsilon) + alpha * dice(1 - y_pred, 1 - y_true, epsilon)


def tp(y_true, y_pred):
    """ Get number of True Positive values.

    Parameters
    ----------
    y_true : ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : ndarray(batch_size, ...)
        numpy array containing predictions of model.
    threshold : float
        threshold for mapping probabilities into class.

    Returns
    -------
    float
        number of true positive predictions.
    """
    y_pred = y_pred.argmax(dim=1)
    y_true, y_pred = _tensor2float(y_true, y_pred)
    return t.sum(y_pred * y_true)


def fp(y_true, y_pred):
    """ Get number of False Positive values.

    Parameters
    ----------
    y_true : ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : ndarray(batch_size, ...)
        numpy array containing predictions of model.
    threshold : float
        threshold for mapping probabilities into class.

    Returns
    -------
    float
        number of false positive predictions.
    """
    y_pred = y_pred.argmax(dim=1)
    y_true, y_pred = _tensor2float(y_true, y_pred)
    return t.sum(y_pred * (1. - y_true))


def tn(y_true, y_pred):
    """ Get number of True Negative values.

    Parameters
    ----------
    y_true : ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : ndarray(batch_size, ...)
        numpy array containing predictions of model.
    threshold : float
        threshold for mapping probabilities into class.

    Returns
    -------
    float
        number of true negative predictions.
    """
    y_pred = y_pred.argmin(dim=1)
    y_true, y_pred = _tensor2float(y_true, y_pred)
    return t.sum(y_pred * (1. - y_true))


def fn(y_true, y_pred):
    """ Get number of False Negative values.

    Parameters
    ----------
    y_true : ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : ndarray(batch_size, ...)
        numpy array containing predictions of model.
    threshold : float
        threshold for mapping probabilities into class.

    Returns
    -------
    float
        number of false negative predictions.
    """
    y_pred = y_pred.argmin(dim=1)
    y_true, y_pred = _tensor2float(y_true, y_pred)
    return t.sum(y_pred * y_true)


def tpr(y_true, y_pred, epsilon=10e-7):
    """ True positive rate.

    Parameters
    ----------
    y_true : ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : ndarray(batch_size, ...)
        numpy array containing predictions of model.
    threshold : float
        threshold for mapping probabilities into class.
    epsilon : float
        small real value for avoiding division by zero error.

    Returns
    -------
    float
        true positive rate value;
    """
    y_true, y_pred = _tensor2float(y_true, y_pred)
    tp_value = tp(y_true, y_pred)
    fn_value = fn(y_true, y_pred)
    return tp_value / (tp_value + fn_value + epsilon)


def tnr(y_true, y_pred, epsilon=10e-7):
    """ True negative rate.

    Parameters
    ----------
    y_true : ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : ndarray(batch_size, ...)
        numpy array containing predictions of model.
    threshold : float
        threshold for mapping probabilities into class.
    epsilon : float
        small real value for avoiding division by zero error.

    Returns
    -------
    float
        true negative rate value.
    """
    y_true, y_pred = _tensor2float(y_true, y_pred)
    tn_value = tn(y_true, y_pred)
    fp_value = fp(y_true, y_pred)
    return tn_value / (tn_value + fp_value + epsilon)


def fpr(y_true, y_pred, epsilon=10e-7):
    """ False positive rate.

    Parameters
    ----------
    y_true : ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : ndarray(batch_size, ...)
        numpy array containing predictions of model.
    threshold : float
        threshold for mapping probabilities into class.
    epsilon : float
        small real value for avoiding division by zero error.

    Returns
    -------
    float
        false positive rate value.
    """
    y_true, y_pred = _tensor2float(y_true, y_pred)
    return 1. - tpr(y_true, y_pred, epsilon)


def fnr(y_true, y_pred, epsilon=10e-7):
    """ False negative rate.

    Parameters
    ----------
    y_true : ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : ndarray(batch_size, ...)
        numpy array containing predictions of model.
    threshold : float
        threshold for mapping probabilities into class.
    epsilon : float
        small real valu for avoiding division by zero error.

    Returns
    -------
    float
        false negative rate value.
    """
    y_true, y_pred = _tensor2float(y_true, y_pred)
    return 1. - tnr(y_true, y_pred, epsilon)


def precision(y_true, y_pred, epsilon=10e-7):
    """ Compute precision metric.

    Parameters
    ----------
    y_true : ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred  : ndarray(batch_size, ...)
        numpy array containing predictions of model.
    threshold : float
        threshold for mapping probabilities into class.
    epsilon : float
        small real value for avoiding division by zero error.

    Returns
    -------
    float
        precision metric value.
    """
    tp_value = tp(y_true, y_pred)
    fp_value = fp(y_true, y_pred)
    return tp_value / (tp_value + fp_value + epsilon)


def recall(y_true, y_pred, epsilon=10e-7):
    """ Compute recall metric.

    Parameters
    ----------
    y_true : ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : ndarray(batch_size, ...)
        numpy array containing predictions of model.
    threshold : float
        threshold for mapping probabilities into class.
    epsilon : float
        small real value for avoiding division by zero error.

    Returns
    -------
    float
        recall metric value.
    """
    return tpr(y_true, y_pred, epsilon=epsilon)


def fscore(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec)


def accuracy(y_true, y_pred):
    """ Compute accuracy on input batched y_pred and y_true.

    Parameters
    ----------
    y_true : np.ndarray(batch_size, ...)
        numpy array containing true target values.
    y_pred : np.ndarray(batch_size, ...)
        numpy array containing predictions of model.
    threshold : float
        threshold for mapping probabilities into class.

    Returns
    -------
    float
        accuracy metric value.
    """
    y_pred = y_pred.argmax(dim=1)
    y_true, y_pred = _tensor2float(y_true, y_pred)
    result = t.mean(t.abs(y_pred - y_true))
    return 1. - result


ALL_METRICS = {'mse': mse,
               'rmse': rmse,
               'mae': mae,
               'dice': dice,
               'sym_dice': sym_dice,
               'tpr': tpr,
               'fpr': fpr,
               'tnr': tnr,
               'fnr': fnr,
               'precision': precision,
               'recall': recall,
               'accuracy': accuracy,
               'fscore': fscore}
