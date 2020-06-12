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

import tensorflow as tf
keras = tf.keras


def build_model(model, optimizer, callbacks=None):
    """TODO: Docstring for build_model.

    Args:
        model: keras.Model. 
        input_shape: List. Input shape including batch_size.
        optimizer: keras.optimizers.

    Kwargs:
        callbacks: keras.

    Returns: TODO

    """
    model.optimizer = optimizer
    model.callbacks = callbacks
    if model.callbacks is not None:
        for callback in model.callbacks:
            callback.set_model(model)
    return model


def get_optimizer(params, custom_scheduler=None):
    """get optimizer from params

    Args:
        params (TODO): TODO

    Returns: TODO

    """
    other_args = {}
    if "clipnorm" in params:
        other_args["clipnorm"] = params.clipnorm
    if "clipvalue" in params:
        other_args["clipvalue"] = params.clipvalue

    if params.lr_decay_policy == "exp":
        lr = tf.keras.optimizers.schedules.ExponentialDecay(params.lr,
                                                            params.decay_steps,
                                                            params.decay_rate,
                                                            staircase=params.staircase)
    else:
        lr = params.init_lr
    if params.optimizer == "adam":
        beta_1 = params.get("beta_1", 0.9)
        beta_2 = params.get("beta_2", 0.999)
        optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1, beta_2=beta_2, **other_args)
    elif params.optimizer == "sgd":
        momentum = params.get("momentum", 0.9)
        optimizer = tf.keras.optimizers.SGD(lr, momentum=momentum, **other_args)
    else:
        raise ValueError("unsupported optimizer: {}".format(params.optimizer))
    return optimizer
