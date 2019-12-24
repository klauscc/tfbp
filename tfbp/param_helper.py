# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2019/09/24
#   description:
#
#================================================================

import os
import ast
import argparse
import datetime
from functools import partial
from ruamel.yaml import YAML
from easydict import EasyDict

from .utils import print_log, get_vacant_gpu


def _literal_eval(value):
    try:
        v = ast.literal_eval(value)
    except:
        v = value
    return v


def get_params(default_params=None):
    """get the params from yaml file and args. The args will override arguemnts in the yaml file.
    Returns: EasyDict instance.

    """
    parser = _default_arg_parser()
    return _update_arg_params(parser, default_params)


def _default_arg_parser():
    """Define a default arg_parser.

    Returns: 
        A argparse.ArgumentParser. More arguments can be added.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", help="Model param file path. Required!", default="")
    parser.add_argument("--workspace",
                        help="Path to save the ckpts and results. Required!",
                        default="")
    parser.add_argument(
        "--gpu_id",
        help="GPU Id to run the model. If not specified, an empty card will be seletected",
        type=int,
        default=-2)
    return parser


def _update_arg_params(arg_parser, default_params=None):
    """ update parameters from arg_parser.

    Args:
        arg_parser: argparse.ArgumentParser.
    """

    parsed, unknown = arg_parser.parse_known_args()
    if default_params and parsed.params == "" and "params" in default_params:
        parsed.params = default_params["params"]

    if os.path.isfile(parsed.params):
        yaml = YAML()
        params = yaml.load(open(parsed.params, "r"))
    else:
        params = {}

    if default_params:
        for k, v in default_params.items():
            if k not in params:
                params[k] = v

    for arg in unknown:
        if arg.startswith(("-", "--")):
            arg_parser.add_argument(arg)

    args = arg_parser.parse_args()
    dict_args = vars(args)

    for key, value in dict_args.items():    # override params from the arg_parser
        if value != None and value != -1 and value != "":
            params[key] = value

    for k, v in params.items():
        params[k] = _literal_eval(v)
        if isinstance(v, str) and "," in v:
            params[k] = [_literal_eval(s) for s in v.split(",")]

    if params["gpu_id"] < 0:
        gpu_id = get_vacant_gpu()
        params["gpu_id"] = gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params["gpu_id"])

    # make ckpt paths
    params = EasyDict(params)
    if params.get("phase", "train") == "train":
        if os.path.isdir(params.workspace):
            dirname, basename = os.path.split(params.workspace)
            i = 1
            new_path = os.path.join(dirname, basename + "-{}".format(i))
            while os.path.isdir(new_path):
                i += 1
                new_path = os.path.join(dirname, basename + "-{}".format(i))
            message = "{} already exist. results will be saved to: {}".format(
                params.workspace, new_path)
            print(message)
            params.workspace = new_path
        os.makedirs(os.path.join(params.workspace, "ckpts"), exist_ok=True)
        os.makedirs(os.path.join(params.workspace, "tensorboard"), exist_ok=True)

    os.makedirs(params.workspace, exist_ok=True)
    log_file = os.path.join(params.workspace, "log.txt")
    with open(log_file, "at") as f:
        f.write("experiment time: {}\n".format(datetime.datetime.today()))
    params.print_fn = partial(print_log, log_file=log_file)

    return params
