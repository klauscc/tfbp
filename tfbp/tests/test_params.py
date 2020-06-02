# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2020/06/02
#   description:
#
#================================================================

from tfbp.param_helper import get_params, get_log_file

default_params = {"workspace": "/tmp/tfbp/test", "new_workspace_if_exist": False}
params = get_params(do_print=True)

params = get_params(default_params=default_params, do_print=True)
params.logger.info("Log to {}".format(get_log_file(params)))
