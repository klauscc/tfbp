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


def build_model(model, input_shape, optimizer, callbacks=None):
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
    for callback in model.callbacks:
        callback.set_model(model)
    model.build(input_shape)
    return model


def get_optimizer(params, custom_scheduler=None):
    """get optimizer from params

    Args:
        params (TODO): TODO

    Returns: TODO

    """
    pass
