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
from tfbp import logger
import tensorflow

log_to_file = '/tmp/tensorflow.log'
logger.set_logger(log_to_file)
log = logger.get_logger()

log.info("Save log to file: {}".format(log_to_file))
