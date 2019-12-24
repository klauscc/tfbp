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


def get_bs_from_inp(inp):
    """get batch_size from inp

    Args:
        inp: Nest structure of Tensor or Tensor.

    Returns: TODO

    """
    if isinstance(inp, (dict,)):
        x = inp[inp.keys()[0]]
    elif isinstance(inp, (list)):
        x = inp[0]
    elif isinstance(inp, tf.Tensor):
        x = inp
    return keras.backend.get_value(tf.shape(x)[0])
