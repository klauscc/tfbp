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
import subprocess
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


def get_vacant_gpu():
    com = "nvidia-smi|sed -n '/%/p'|sed 's/|/\\n/g'|sed -n '/MiB/p'|sed 's/ //g'|sed 's/MiB/\\n/'|sed '/\\//d'"
    gpum = subprocess.check_output(com, shell=True)
    gpum = gpum.decode('utf-8').split('\n')
    gpum = gpum[:-1]
    if len(gpum) == 0:
        return -1
    for i, d in enumerate(gpum):
        gpum[i] = int(gpum[i])
    gpu_id = gpum.index(min(gpum))
    if len(gpum) == 4:
        gpu_id = 3 - gpu_id
    return gpu_id
