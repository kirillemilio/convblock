""" Contains Estimator class for (torch) model training/evaluation """


import sys, os, shutil, time, numpy as np, torch, pandas as pd
import dill
from torch.autograd import Variable
from collections import OrderedDict, namedtuple
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from .metrics import as_torch as _as_torch, as_numpy as _as_numpy


DefaultCallback = namedtuple('callback', ['func', 'how_many', 'kwargs'])
ReccurentCallback = namedtuple('reccurent_callback', ['func', 'every', 'how_many', 'kwargs'])


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Estimator():

    def __init__(self, model, save_folder, cuda_device=None, loss_fn=None, optimizer=None):
        if torch.__version__ == '0.3.1':
            if cuda_device is None:
                self._device = torch.cuda.current_device() if torch.cuda.is_available() else None
            elif isinstance(cuda_device, (int, list)):
                self._device = cuda_device
        else:
            if cuda_device is None:
                self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            elif isinstance(cuda_device, int):
                self._device = torch.device(cuda_device)
            elif isinstance(cuda_device, list):
                self._device = cuda_device

        # init
        self.save_folder = save_folder
        self._writer = SummaryWriter(save_folder)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self._labels_and_predictions_store = OrderedDict()
        if isinstance(model, str):
            self.model = None
            self.load(model)
        else:
            self.model = model
        # counters
        self._iteration_count = 0  # train iter counter (.fit)
        self._eval_iteration_count = 0  # eval iter count (.evaluate)
        self._epoch_count = 0  # train epoch count
        # metrics and loss
        # last batch metrics and loss
        self._last_train_batch_log = OrderedDict()  # last train batch loss and metrics
        self._last_eval_batch_log = OrderedDict()  # last val batch loss and metrics
        # container for current epoch train loss and mterics
        self._last_train_epoch_log = []
        # epoch metrics and loss
        self._eval_metrics = OrderedDict()  # dict for (.evaluate) metrics values, use iteration as index
        self._epoch_metrics = OrderedDict()  # dict for (.fit) train metrics values, use epoch as index
        # store for callbacks
        self._step_callbacks = []
        self._epoch_begin_callbacks = []
        self._epoch_end_callbacks = []
        self._recurrent_callbacks = []

    @staticmethod
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
        return _as_torch(data, dtype, device, grad, async, **kwargs)

    @staticmethod
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
        return _as_numpy(data, dtype)

    def step(self, inputs, targets):
        """ Make training step.

        Parameters
        ----------
        inputs : ndarray, torch Tensor or Variable
            inputs of model.
        targets : ndarray, torch Tensor or Variable
            targets of model.
        """
        inputs_ = self.as_torch(inputs, device=self._device, grad=False)
        targets_ = self.as_torch(targets, dtype=str(targets.dtype), device=self._device, grad=False)
        predictions = self.model(inputs_)
        loss = self.loss_fn(predictions, targets_)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, predictions

    def load(self, checkpoint):
        self.model = load_checkpoint(checkpoint, self.model, optimizer=self.optimizer)

    def save(self, checkpoint):
        self.checkpoint()

    @property
    def writer(self):
        return self._writer

    def log_scalar(self, name, value, step):
        """ Save scalar to Tensorboard log.

        Parameters
        ----------
        name : str
            how to name variable in tensorboard,
            use "/" for sections.
        value : int or float
            value to log
        step : int
            number of iteration or other index
        """
        self.writer.add_scalar(name, value, step)

    def log_images(self, name, images, step):
        """ Save list of images to Tensorboard log.

        Parameters
        ----------
        name : str
            how to name variable in tensorboard,
            use "/" for sections.
        images : list of 2D arrays
            images to be stored
        step : int
            number of iteration or other index
        """

        img_grid = vutils.make_grid(images, normalize=True, scale_each=True)
        self.writer.add_image(name, img_grid, step)

    def log_volume(self, name, volume, step, where='mid'):
        """ Save specific 2D slices from volumetric images
        to Tensorboard log.

        Parameters
        ----------
        name : str
            how to name variable in tensorboard,
            use "/" for sections.
        volume : np.array(batch_size, channels, z_dim, y_dim, x_dim)
            volumetric image
        step : int
            number of iteration or other index
        where : list, tuple, int, float or 'mid'
            what slices to log from volume.
            if 'mid', take middle slice from z-dim;
            if list or tuple of int/float,
            take multiple slices from z-dim, specified by list;
            if int or float, ise it as index of z-dim.
        """
        if where == 'mid':
            loc = [volume.shape[2] // 2]
        elif isinstance(where, (list, tuple)):
            loc = [int(k) for k in where]
        elif isinstance(where, (float, int)):
            loc = [int(where)]
        img = [img for z in loc for img in volume[:, :, z, :, :]]
        self.log_images(name, img, step)

    def _log_histogram(self, name, array, step):
        ''' Save distribution to Tensorboard log,
        e.g. network's weights

        Parameters
        ----------
        name : str
            how to name variable in tensorboard,
            use "/" for sections.
        array : list or array-alike
            values to be logged in histo
        step : int
            number of iteration or other index
        '''
        self.writer.add_histogram(name, array, step)

    def _log_graph(self, model):
        ''' Save model graph to Tensorboard log.

        Parameters
        ----------
        model : torch.nn.Module
            model with graph to be saved
        '''
        if hasattr(self.model, 'input_shape'):
            rnd = Variable(torch.randn((4,) + self.model.input_shape))
        else:
            rnd = Variable(torch.randn((4,) + self.model.module.input_shape))
        if torch.__version__ == '0.3.1':
            if self._device is not None:
                rnd = rnd.cuda(self._device)
        else:
            rnd = rnd.to(self._device)
        self.writer.add_graph(model, rnd)
        del rnd

    def _log_inputs_and_outputs(self, name, prediction, input, label, i):
        ''' Save input (image), label and prediction to Tensorboard.

        Parameters
        ----------
        name : str
            how to name section in tensorboard,
            this method create subsections `input`, `label`, `prediction`.
        prediction : torch.tensor
            predicted labels or masks for batch of images
        input : torch.tensor
            input batch of images or volumes
        label : torch.tensor
            ground truth classes for input batch
        i : int
            iteration number

        Note: this method use self.mode to distinguish `classification`
        and `segmentation`.
        '''
        log_func = self.log_images if input.dim() == 4 else self.log_volume

        log_func(name + '/input', input, i)
        if prediction.dim() == 2:
            prediction_class = prediction.argmax(dim=1)
            for nb, item in enumerate(prediction_class):
                self.log_scalar(name + 'prediction/' + str(nb), float(item.item()), i)
            for nb, item in enumerate(label):
                self.log_scalar(name + 'label/' + str(nb), float(item.item()), i)
        else:
            log_func(name + '/prediction', prediction, i)
            log_func(name + '/label', label, i)

    def log_pr_curve(self, name, labels, predictions, step):
        self.writer.add_pr_curve(name, labels, predictions, step)

    def compile(self, optimizer=None, loss=None, **kwargs):
        ''' Compiling model, i.e. fixing optimizer and loss function. '''
        if torch.__version__ == '0.3.1':
            if self._device is None:
                self.model.cpu()
            elif isinstance(self._device, int):
                self.model.cuda(self._device)
            elif isinstance(self._device, list):
                self.model = torch.nn.DataParallel(self.model,
                                                   device_ids=self._device)
        else:
            self.model.to(self._device)
        if loss:
            self.loss_fn = loss
        if optimizer:
            self.optimizer = optimizer(filter(lambda p: p.requires_grad,
                                              self.model.parameters()), **kwargs)

    def log_loss(self, name, value, step):
        """ Save loss value to Tensorboard and Estimator logs.

        Parameters
        ----------
        name : str
            how to name variable in tensorboard,
            use "/" for sections.
        value : int or float
            value to log
        step : int
            number of iteration or other index
        """
        log = self._last_train_batch_log if name.split("/")[0] == 'train' \
            else self._last_eval_batch_log

        log[name] = value
        self.log_scalar(name, value, step)

    def _log_model_params(self, model, i, include_grads=True):
        ''' Save network's weights distribution to Tensorboard log.

        Parameters
        ----------
        model : torch.nn.Module
            model with graph to be saved
        i : int
            iteration number
        '''
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), i)
        if include_grads:
            for name, param in self.model.named_parameters():
                self.writer.add_histogram('grad/' + name, param.grad.clone().cpu().data.numpy(), i)

    def log_metrics(self, log, metrics, y_true, y_pred, step, prefix=''):
        for metric in metrics:
            q = metric(y_true, y_pred)
            log[prefix + metric.__name__] = float(q.item())
            self.writer.add_scalar(prefix + metric.__name__, float(q.item()), step)
        return log

    def add_step_callback(self, callback, how_many_times=1, **kwargs):
        self._step_callbacks += [DefaultCallback(callback, how_many_times,
                                                 kwargs if kwargs else None)]
        return self

    def add_epoch_begin_callback(self, callback, how_many_times=1, **kwargs):
        self._epoch_begin_callbacks += [DefaultCallback(callback, how_many_times,
                                                        kwargs if kwargs else None)]
        return self

    def add_epoch_end_callback(self, callback, how_many_times=1, **kwargs):
        self._epoch_end_callbacks += [DefaultCallback(callback,
                                                      how_many_times,
                                                      kwargs if kwargs else None)]
        return self

    def add_recurrent_callback(self, callback, every, how_many_times=1, **kwargs):
        ''' Add callback performed `every` iterations for `how_many_times` in a row
        '''
        assert isinstance(every, int)
        assert every > 0
        self._recurrent_callbacks += [ReccurentCallback(callback, every,
                                                        how_many_times,
                                                        kwargs if kwargs else None)]
        return self

    def checkpoint(self):
        ''' Save Estimator's model checkpoint '''
        save_checkpoint(self.model, False, os.path.join(self.save_folder, str(self._iteration_count)))

    def log_model(self):
        ''' Log estimator's model to Tensorboard '''
        self._log_graph(self.model)

    def log_inputs_and_outputs(self, inputs, targets, it, prefix='val_check/'):
        self.model.eval()
        inputs = self.as_torch(inputs, device=self._device, grad=False)
        targets = self.as_torch(targets, device=self._device, grad=False)
        pred = self.model(inputs)
        self._log_inputs_and_outputs(prefix, pred, inputs, targets, it)

    def _do_callbacks(self, cbs):
        for c in cbs:
            for _ in range(c.how_many):
                if c.kwargs is None:
                    c.func()
                else:
                    c.func(**c.kwargs)

    def _do_reccurent_callbacks(self, it):
        for c in self._recurrent_callbacks:
            if not c.every % it and it != 0:
                for _ in range(c.how_many):
                    if c.kwargs is None:
                        c.func()
                    else:
                        c.func(**c.kwargs)

    def restart_iteration_count(self):
        self._iteration_count = 0

    def fit(self, inputs, targets, metrics=None, steps_per_epoch=10, epochs=1,
            initial_epoch=0, seed=9, log_model=False):
        # at the beginning, set seed, iter limit and validation mode
        if not self._iteration_count:
            if seed and seed >= 0:
                torch.manual_seed(seed)
                if log_model:
                    self.log_model()
            self.steps_per_epoch = steps_per_epoch

        # move training epoch counter to initial_epoch
        if initial_epoch:
            self._epoch_count = initial_epoch
            self._iteration_count = self.steps_per_epoch * self._epoch_count

        # epoch end condition:
        if steps_per_epoch:
            epoch_end = self._iteration_count and  \
                not self._iteration_count % self.steps_per_epoch
        else:
            epoch_end = None

        self.model.train()

        # ON EPOCH BEGIN
        if self._epoch_begin_callbacks:
            self._do_callbacks(self._epoch_begin_callbacks)

        # actual training step, iteration counter, metrics
        loss, predictions = self.step(inputs, targets)
        self.log_loss('train/loss', float(loss.item()), self._iteration_count)

        if metrics:
            self._last_train_batch_log = \
                self.log_metrics(self._last_train_batch_log, metrics, targets,
                                 predictions, self._iteration_count, 'train/')
        self._last_train_epoch_log += [self._last_train_batch_log.copy()]

        # STEP CALLBACKS (After step)
        if self._step_callbacks:
            self._do_callbacks(self._step_callbacks)

        if self._recurrent_callbacks:
            self._do_reccurent_callbacks(self._iteration_count)
        self._iteration_count += 1

        if epoch_end:
            # let's also compute average (train epoch) everythng
            epoch_means = dict((pd.DataFrame(self._last_train_epoch_log)).mean())
            self._epoch_metrics[self._epoch_count] = epoch_means

            if self._epoch_end_callbacks:
                self._do_callbacks(self._epoch_end_callbacks)

            # move counter and clear last_train container
            self._epoch_count += 1
            self._last_train_epoch_log = []

        if epochs:
            if self._iteration_count > epochs * self.steps_per_epoch:
                raise StopIteration

    def predict(self, inputs):
        inputs_ = self.as_torch(inputs, device=self._device, grad=False)
        return self.model(inputs_).detach().cpu().numpy()

    @property
    def eval_metrics(self):
        return pd.DataFrame(self._eval_metrics).transpose()

    def compute_metrics(self, inputs, targets, metrics, batch_size=None,
                        store_labels_and_predictions=False, log_inputs=False,
                        dst=None, it=None, ID=None, filename=None, **kwargs):
        ''' Validate model on `val_data` for `val_steps`, computing `metrics`

        Parameters
        ----------
        inputs : numpy.array
            input array for model
        targets : numpy.array
            array with ground truth labels
        metrics : list or tuple of callables
            metrics to be computed
        batch_size : int
            if item is too large and is splitted into many pieces
            in batch dimension (zero-dim), one may want to go through
            in mini-batches, taking batch_size at once, gather all
            predictions altogether and computing metrics for whole item.
        store_labels_and_predictions : bool
            if True, collects labels and predicitons.
            Necessary for ROC Curve drawing, Estimator.roc_auc_curve()
        dst : list
            list-object, where metrics would be inserted
        it : int
            change internal evaluation iteration counter to `it`,
            used in computed metrics output dict.
        ID : str
            _eval_metrics will use it as index, if passed.

        Computed metrics are stored in Estimator._eval_metrics.
        It is quick to see them as pd.DataFrame:
        Estimator.eval_metrics

        '''
        # manually state iteration counter, if any
        if it:
            self._eval_iteration_count = it
        self.model.eval()
        val_y = self.as_torch(targets, dtype=str(targets.dtype),
                              device=self._device, grad=False)
        # if `inputs` too large, use batch logic. i.e.
        # split-predict-gather and compute loss/metrics for entire `inputs`
        if batch_size:
            prediction_list = []
            for b_iter in tqdm(range(len(inputs) // batch_size + 1),
                               desc='Processing item: {}'.format(
                                     self._eval_iteration_count)):
                # slicing inputs by batch_size:
                if b_iter < len(inputs) // batch_size:
                    val_x_batch = inputs[b_iter * batch_size: (b_iter + 1) * batch_size, :]
                # check if tail is smaller than batch_size, and take it
                elif b_iter == len(inputs) // batch_size and len(inputs) % batch_size:
                    val_x_batch = inputs[b_iter * batch_size: len(inputs), :]

                # handle CUDA memory allocation problems:
                if torch.__version__ == '0.3.1':
                    val_x_batch = self.as_torch(val_x_batch, grad=False,
                                                volatile=True, device=self._device)
                    prediction_list += [self.model(val_x_batch)]
                else:
                    val_x_batch = self.as_torch(val_x_batch, grad=False,
                                                device=self._device)
                    with torch.no_grad():
                        prediction_list += [self.model(val_x_batch)]
                del val_x_batch
                torch.cuda.empty_cache()

            # concating predictions in one tensor
            val_pred = torch.cat(prediction_list)
        else:
            val_x = self.as_torch(inputs, grad=False, device=self._device)
            val_pred = self.model(val_x)

        if log_inputs:
            self._log_inputs_and_outputs('val_check', val_pred, val_x, val_y,
                                         self._eval_iteration_count)

        val_loss = self.loss_fn(val_pred, val_y)
        self.log_loss('val/loss', float(val_loss.item()),
                      self._eval_iteration_count)
        if metrics:
            self._last_eval_batch_log = \
                self.log_metrics(self._last_eval_batch_log, metrics, val_y,
                                 val_pred, self._eval_iteration_count, 'val/')

        # Saving metrics to _eval_metrics and DataFrame
        index = ID if ID else self._eval_iteration_count

        if store_labels_and_predictions:
            labels_predictions = OrderedDict()
            labels_predictions['labels'] = val_y.cpu().detach().numpy()
            labels_predictions['predictions'] = val_pred.cpu().detach().numpy()
            self._labels_and_predictions_store[index] = labels_predictions

        if not dst:
            self._eval_metrics[index] = self._last_eval_batch_log.copy()
            self._eval_iteration_count += 1
        else:
            dst += [OrderedDict([(index, self._last_eval_batch_log.copy())])]

        df = pd.DataFrame(self._eval_metrics[index], index=list([index]))

        if not filename:
            filename = os.path.join(self.save_folder,
                                    self.model.__class__.__name__ + '.csv')
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                header = not bool(len(f.readline()))
            f.close()
            df.to_csv(filename, mode='a', header=header, index=False)
        else:
            df.to_csv(filename, index=False)

    def roc_auc_curve(self, transform_predictions=softmax):

        y_true = np.concatenate(
            pd.DataFrame(
                self._labels_and_predictions_store)
            .transpose()['labels'].values)
        func = transform_predictions if transform_predictions else lambda x: x
        y_pred = np.max(
            func(
                np.concatenate(
                    pd.DataFrame(
                        self._labels_and_predictions_store)
                    .transpose()['predictions'].values)
                ),
            1)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
# Huge thanks to github.com/henryre/pytorch-fitmodule/


class ProgressBar(object):
    """Cheers @ajratner and keras-team"""

    def __init__(self, n, length=40):
        # Protect against division by zero
        self.n = max(1, n)
        self.nf = float(n)
        self.length = length
        # Precalculate the i values that should trigger a write operation
        self.ticks = set([round(i / 100.0 * n) for i in range(101)])
        self.ticks.add(n-1)
        self._start = time.time()
        self.interval = 0.05
        self._last_update = 0
        self.bar(0)

    def bar(self, i, message=" "):
        """Assumes i ranges through [0, n-1]"""
        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if i:
            time_per_unit = (now - self._start) / i
        else:
            time_per_unit = 0
        if self.n is not None and i < self.n:
            eta = time_per_unit * (self.n - i)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta

            info = ' - ETA: %s' % eta_format
        if (now - self._last_update < self.interval and i < self.n):
                return
        if i in self.ticks:
            if i == self.n-1:
                message = ''
                info = ''
            b = int(np.ceil(((i+1) / self.nf) * self.length))
            sys.stdout.write("\r[{0}{1}] {2}%\t{3}\t{4}".format(
                "="*b, " "*(self.length-b), int(100*((i+1) / self.nf)),
                message, info
            ))
            sys.stdout.flush()

    def close(self, message=""):
        # Move the bar to 100% before closing
        self.bar(self.n-1)
        sys.stdout.write("\n{0}\n\n".format(message))
        sys.stdout.flush()


def log_to_message(log, precision=4):
    fmt = "{0}: {1:." + str(precision) + "f}"
    return "    ".join(fmt.format(k, v) for k, v in log.items())


# thanks to @cs230-stanford
def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path.

    If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    model = torch.load(checkpoint, pickle_module=dill)
    return model


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.

    If is_best==True, also saves checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict,
            may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath, pickle_module=dill)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
