# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2019/12/17
#   description:
#
#================================================================

import time
import tensorflow as tf
keras = tf.keras
K = keras.backend

from .callbacks.ckpt_callbacks import CheckpointCallback
from .utils import get_bs_from_inp


class BasicTrainer(keras.callbacks.Callback):
    """basic trainer. 

    Itself is a callback that can be executed before,after train/val

    Attributes:
        params: EasyDict. The overall parameters.
        callbacks: list. The callbacks for the trainer.
        models: list. All the models will be used and checkpointed.

    """

    def __init__(self, params):
        """
            Args:
                params: EasyDict. The parameters.
        """
        super(BasicTrainer, self).__init__()
        self.params = params
        self.initiate()

        self.logger = params.logger

    def initiate(self):
        """initiate the trainer"""
        #initiate metrics
        self._metrics = EasyDict()
        self._train_metrics = EasyDict()
        self._val_metrics = EasyDict()

        # create loss for both train and val
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.metric("loss", train_metric=self.train_loss, val_metric=self.val_loss)

        #initiate logs for callbacks
        self.callbacks = []
        self.logs = EasyDict()
        self._update_logs()

        # register models
        self.register_models()
        # log step
        self.cur_step = tf.Variable(0, dtype=tf.int64)
        tf.summary.experimental.set_step(self.cur_step)
        #define ckpt
        self.define_ckpt()

        # print interval
        self.log_steps = self.params.get("log_steps", 100)

    def train_batch(self, inputs, batch):
        pass

    def val_batch(self, inputs, batch):
        pass

    def train(self,
              train_dataset,
              val_dataset,
              epochs=200,
              validation_steps=None,
              validation_freq=1):
        """model train on `train_dataset`.

        Args:
            train_dataset: `tf.data` dataset. The dataset used in train. Iterator of the dataset should
                             return a tuple (inp, tar). `inp` and `tar` can be a nested structure.
            val_dataset: `tf.data` dataset or None. The same as `train_dataset` except used during 
                            validation. If None, no validation will be performed. 

        Kwargs:
            epochs: Integer. The maximum train epochs.Default is 200.
            validation_steps: Integer or `None`. The validation steps. If None, the validation will 
                                iterate until val_dataset ends.
            validation_freq: Integer or `None`. It's the frequency in epoch to do validation. Default is 
                                1, that is validation after each epoch.

        """
        begin_time = time.time()
        fmt = "Epoch {}, step {}, cost {:.4f}s. metrics: {}"

        trainer_callbacks = self.callbacks

        # before train begin
        self._reset_metric_state()
        self._call_callbacks("on_train_begin")

        # train begin
        for epoch in range(epochs):

            with self.train_summary_writer.as_default():

                #before epoch begin
                self._call_callbacks("on_epoch_begin", epoch, self.logs)
                self.logger.info("Epoch {}. Learning rate: {}".format(epoch,
                                                                      self._print_model_lr()))

                epoch_t1 = time.time()

                # epoch begin
                for step, (inp, tar) in enumerate(datasets):

                    t1 = time.time()
                    batch_size = get_bs_from_inp(inp)

                    # before train_batch
                    logs = self.logs.copy()
                    logs.update({"size": batch_size, "batch": step})
                    self._call_callbacks("on_train_batch_begin", step, logs)

                    # train batch
                    train_ret = self.train_batch(inputs, cur_step)
                    self._update_logs()

                    # after train_batch
                    logs = self.logs.copy()
                    logs.update({"size": batch_size, "batch": step, "result": train_ret})
                    self._call_callbacks("on_train_batch_end", step, logs)

                    t2 = time.time()
                    if step % self.log_steps == 0:
                        self.logger.info(
                            fmt.format(epoch, step, t2 - t1,
                                       self.print_metrics(self._train_metrics)))
                # log at the end of the epoch.
                self.logger.info("Epoch {} finished.\n".format(epoch) +
                                 fmt.format(epoch, step, t2 -
                                            t1, self.print_metrics(self._train_metrics)))

            if not val_dataset or (epoch + 1) % validation_freq != 0:
                continue    #skip validation

            with self.val_summary_writer.as_default():

                # validation begin
                self.logger.info("Begin evaluation. epoch {}".format(epoch))
                self._call_callbacks("on_test_begin", self.logs)

                for step, (inp, tar) in enumerate(val_dataset):
                    if validation_steps and step == validation_steps:
                        break
                    t1 = time.time()

                    batch_size = get_bs_from_inp(inp)

                    # on val_batch begin
                    logs = self.logs.copy()
                    logs.update({"size": batch_size, "batch": step})
                    self._call_callbacks("on_test_batch_begin", step, logs)

                    val_ret = self.val_batch(inputs, step)
                    self._update_logs()

                    # on val_batch end
                    logs = self.logs.copy()
                    logs.update({"size": batch_size, "batch": step, "result": val_ret})
                    self._call_callbacks("on_test_batch_end", step, logs)

                    t2 = time.time()
                    self.logger.info(
                        fmt.format(epoch, step, t2 - t1, self.print_metrics(self._val_metrics)))

                self._call_callbacks("on_test_end", self.logs)

            self._call_callbacks("on_epoch_end", self.logs)

            self._reset_metric_state()

            epoch_t2 = time.time()
            self.logger.info("Epoch {}. Total Time: {:.4f}s".format(epoch, epoch_t2 - epoch_t1))
        end_time = time.time()
        self.logger.info("Training cost total:{:.2f}s".format(end_time - begin_time))
        pass

    def evaluate(self, dataset, steps=None):
        """TODO: Docstring for evaluate.

        Args:
            dataset (TODO): TODO

        Kwargs:
            steps (TODO): TODO

        Returns: TODO

        """
        pass

    def predict(self, x, batch_size=None, steps=None):
        """TODO: Docstring for predict.

        Args:
            x (TODO): TODO

        Kwargs:
            batch_size (TODO): TODO
            steps (TODO): TODO

        Returns: TODO

        """
        pass

    def register_models(self):
        self.models = []
        for attr in dir(self):
            if isinstance(attr, (keras.Model, keras.Sequential)):
                self.models.append(attr)

    def define_ckpt(self):
        ckpt_dict = {"epoch": tf.Variable(0), 'cur_step': self.cur_step}

        for i, model in enumerate(self.models):

            if hasattr(model, "name"):
                model_name = model.name
            else:
                model_name = "model_" + i
                model.name = model_name
            optimizer_name = model_name + "_optimizer"

            ckpt_dict[model_name] = model
            ckpt_dict[optimizer_name] = model.optimizer

        self.ckpt = tf.train.Checkpoint(**ckpt_dict)

        #ckpt callback
        ckpt_path = os.path.join(self.params.workspace, "ckpt")
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt_callbacks = CheckpointCallback(ckpt_path,
                                            ckpt=self.ckpt,
                                            save_freq=params.get("save_freq", 1),
                                            max_to_keep=params.get("max_to_keep", 100))
        self.callbacks.append(ckpt_callbacks)

    def create_summary_writer(self):
        log_dir = os.path.join(self.params.workspace, "tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
        self.val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    def train_metric(self, key, metric=None):
        """add or return train metric.

        Args:
            key: String. The actual key will be `"train_"+key`.
        Kwargs:
            metric: Metric. If the `key` is a new metric, value must be specified.

        Returns: Metric.

        """
        actual_key = "train_" + key
        return self.metric(key, train_metric=metric, mode="train")

    def val_metric(self, key, metric=None):
        """add or return val metric.

        Args:
            key: String. The actual key will be `"val_"+key`.
        Kwargs:
            metric: Metric. If the `key` is a new metric, value must be specified.

        Returns: Metric.

        """
        actual_key = "val_" + key
        return self.metric(key, val_metric=metric, mode="val")

    def metric(self, key, train_metric=None, val_metric=None, mode="both"):
        """add or return metric.
        Args:
            key: String. The metric name.

        Kwargs:
            train_metric: Metric. Should be specified if the metric has not been created.
            val_metric: Metric. Should be specified if mode is "both".
            mode: String. Either "train" or "val" or "both". If mode is "both", 
                the train metric will be returned.

        Returns: the metric.

        """
        train_key = "train_" + key
        val_key = "val_" + key

        if mode == "both":
            assert train_metric is not None and val_metric is not None, \
                    "Both train_metric and val_metric should not be None when mode is {}".format(mode)
            self._metrics[train_key] = train_metric
            self._train_metrics[train_key] = train_metric
            self._metrics[val_key] = val_metric
            self._val_metrics[val_key] = val_metric
            return train_metric
        elif mode == "train":
            if train_metric is not None:
                self._metrics[train_key] = train_metric
                self._train_metrics[train_key] = train_metric
            else:
                assert train_key in self._metrics, "Metric: {} is not created".format(train_key)

            return self._metrics[train_key]
        else:
            if val_metric is not None:
                self._metrics[val_key] = val_metric
                self._val_metrics[val_key] = val_metric
            else:
                assert val_key in self._metrics, "Metric: {} is not created".format(val_key)
            return self._metrics[val_key]

    def _update_logs(self):
        for k, v in self._metrics.items():
            self.logs[k] = v.result()

    def _print_model_lr(self):
        res = ""
        for model in models:
            model_lr = model.optimizer.decayed_lr(tf.float32)
            tf.summary.scalar(model.name + "/learning_rate",
                              model_lr,
                              step=tf.summary.experimental.get_step())
            res += "{}:{:.7f}".format(model.name,
                                      K.get_value(model.optimizer.decayed_lr(tf.float32)))
        return res

    def _print_dict(self, dic):
        res = ""
        for k, v in dic.items():
            res += "{}:{:.7f},".format(k, v.result())
        return res

    def _call_callbacks(self, phase, *args):
        getattr(self, phase)(*args)    # The trainer itself is a callback.

        [getattr(callback, phase)(*args) for callback in trainer_callbacks]    # trainer callbacks

        [getattr(callback, phase)(*args) for callback in model.callbacks for model in self.models
        ]    # each model's callbacks

    def _reset_metric_state(self):
        for k, metric in self._metrics.items():
            metric.reset_states()
