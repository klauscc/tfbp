# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2020/06/01
#   description:
#
#================================================================

import os
import sys
import logging

LOGGER = logging.getLogger("tensorflow")
LOGGER.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s-%(levelname)s: %(message)s',
                              datefmt='%m-%d %H:%M')


def set_logger(log_to_file=''):

    global LOGGER

    # add file handler
    if log_to_file != '':
        fh = logging.FileHandler(log_to_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        LOGGER.addHandler(fh)
    return LOGGER


def get_logger():
    return LOGGER
