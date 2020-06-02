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

from .utils import get_vacant_gpu
from .logger import get_logger, set_logger

logger = get_logger()


def get_params(default_params=None, do_print=False):
    """get the params from yaml file and args. The args will override arguemnts in the yaml file.
    Returns: EasyDict instance.

    """
    parser = _default_arg_parser()
    params = _update_arg_params(parser, default_params)
    if do_print:
        print_params(params)
    return params


def print_params(params):
    """print params with formatter.

    Args:
        params (EasyDict): the params. 

    Returns: The printed string.

    """
    msg = "Params:\n\n"
    for k, v in params.items():
        msg += "{:20}:{}".format(k, v)
        msg += "\n"
    logger.info(msg)
    return msg


def get_log_file(params):
    """get log file.
    """
    return os.path.join(params.workspace, "log.txt")


def _literal_eval(value):
    try:
        v = ast.literal_eval(value)
    except:
        v = value
    return v


def _default_arg_parser():
    """Define a default arg_parser.

    Returns: 
        A argparse.ArgumentParser. More arguments can be added.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Model config file path.", default="", type=str)
    parser.add_argument("--workspace",
                        help="Path to save the ckpts and results.",
                        default="",
                        type=str)
    parser.add_argument("--new_workspace_if_exist",
                        help="Whether to create a new workspace if the specified exist.",
                        default=False)
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
    if default_params and parsed.config_file == "" and "config_file" in default_params:
        parsed.config_file = default_params["config_file"]

    # update params from config_file
    if os.path.isfile(parsed.config_file):
        yaml = YAML()
        params = yaml.load(open(parsed.config_file, "r"))
    else:
        params = {}

    # update default_params.
    if default_params:
        for k, v in default_params.items():
            if k not in params:
                params[k] = v

    # update params from arg_parser.
    for arg in unknown:
        if arg.startswith(("-", "--")):
            arg_parser.add_argument(arg)
    args = arg_parser.parse_args()
    dict_args = vars(args)
    for key, value in dict_args.items():    # override params from the arg_parser
        if key not in params:
            params[key] = value
        elif value != None and value != -1 and value != "":
            params[key] = value

    # eval str or number.
    for k, v in params.items():
        params[k] = _literal_eval(v)
        if isinstance(v, str) and "," in v:
            params[k] = [_literal_eval(s) for s in v.split(",")]

    if params["gpu_id"] < 0:
        gpu_id = get_vacant_gpu()
        params["gpu_id"] = gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params["gpu_id"])

    params = EasyDict(params)

    params = process_workspace(params)

    return params


def update_workspace(params):
    """update workspace

    Args:
        params (EasyDict): The params.

    Returns: 
        str. new workspace path.

    """
    if os.path.isdir(
            params.workspace
    ) and params.new_workspace_if_exist:    # if workspace already exist and new_workspace_if_exist
        dirname, basename = os.path.split(params.workspace)
        i = 1
        new_path = os.path.join(dirname, basename + "-{}".format(i))
        while os.path.isdir(new_path):
            i += 1
            new_path = os.path.join(dirname, basename + "-{}".format(i))
    else:
        new_path = params.workspace
    return new_path


def process_workspace(params):
    """
    Args:
        params (EasyDict): The params.

    Returns:
        EasyDict. update the workspace.

    """
    # if workspace is not specified, do not create workspace.
    new_workspace = update_workspace(params)

    if new_workspace != "":
        os.makedirs(new_workspace, exist_ok=True)

    # update logger
    log_file = os.path.join(new_workspace, "log.txt")
    params.logger = set_logger(log_to_file=log_file)

    if new_workspace == "":
        return params

    if new_workspace != params.workspace:
        message = "{} already exist. results will be saved to: {}".format(
            params.workspace, new_workspace)
        logger.info(message)

    params.workspace = new_workspace

    return params
