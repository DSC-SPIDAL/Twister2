# Copyright 2016 Twitter. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
''' cli_helper.py '''
import logging
import twister2.tools.cli.src.python.config as config
import twister2.tools.cli.src.python.args as args
import twister2.tools.cli.src.python.execute as execute
import twister2.tools.cli.src.python.jars as jars

from twister2.tools.cli.src.python.log import Log

################################################################################

def create_parser(subparsers, action, help_arg):
    '''
    :param subparsers:
    :param action:
    :param help_arg:
    :return:
    '''
    parser = subparsers.add_parser(
        action,
        help=help_arg,
        usage="%(prog)s [options] cluster <job-id>",
        add_help=True)

    args.add_titles(parser)
    args.add_cluster_role_env(parser)
    args.add_job(parser)

    args.add_config(parser)
    args.add_verbose(parser)

    parser.set_defaults(subcommand=action)
    return parser


################################################################################

def create_parser_without_jobid(subparsers, action, help_arg):
    '''
    :param subparsers:
    :param action:
    :param help_arg:
    :return:
    '''
    parser = subparsers.add_parser(
        action,
        help=help_arg,
        usage="%(prog)s [options] cluster",
        add_help=True)

    args.add_titles(parser)
    args.add_cluster_role_env(parser)

    args.add_config(parser)
    args.add_verbose(parser)

    parser.set_defaults(subcommand=action)
    return parser


################################################################################
# pylint: disable=dangerous-default-value
def run(command, cl_args, action, extra_args=[], extra_lib_jars=[]):
    '''
    helper function to take action on topologies
    :param command:
    :param cl_args:
    :param action:        description of action taken
    :return:
    '''
    job_id = cl_args['job-id']

    new_args = [
        "--cluster", cl_args['cluster'],
        "--twister2_home", config.get_twister2_dir(),
        "--config_path", config.get_twister2_conf_dir(),
        "--job_id", job_id,
        "--command", command,
    ]
    new_args += extra_args

    lib_jars = config.get_twister2_libs(jars.resource_scheduler_jars() + jars.statemgr_jars())
    lib_jars += extra_lib_jars

    if Log.getEffectiveLevel() == logging.DEBUG:
        new_args.append("--verbose")

    java_defines = []
    conf_dir_common = config.get_twister2_cluster_conf_dir("common", config.get_twister2_conf_dir())
    java_defines.append("java.util.logging.config.file=" + conf_dir_common + "/logger.properties")

    # invoke the runtime manager to kill the job
    result = execute.twister2_class(
        'edu.iu.dsc.tws.rsched.core.RuntimeManagerMain',
        lib_jars,
        extra_jars=[],
        args=new_args,
        java_defines=java_defines
    )

    err_msg = "Failed to %s %s" % (action, job_id)
    succ_msg = "Successfully %s %s" % (action, job_id)
    result.add_context(err_msg, succ_msg)
    return result

################################################################################
# pylint: disable=dangerous-default-value
def run_without_jobid(command, cl_args, action, extra_args=[], extra_lib_jars=[]):
    '''
    helper function to take action on topologies
    :param command:
    :param cl_args:
    :param action:        description of action taken
    :return:
    '''

    new_args = [
        "--cluster", cl_args['cluster'],
        "--twister2_home", config.get_twister2_dir(),
        "--config_path", config.get_twister2_conf_dir(),
        "--command", command,
    ]
    new_args += extra_args

    lib_jars = config.get_twister2_libs(jars.resource_scheduler_jars() + jars.statemgr_jars())
    lib_jars += extra_lib_jars

    if Log.getEffectiveLevel() == logging.DEBUG:
        new_args.append("--verbose")

    java_defines = []
    conf_dir_common = config.get_twister2_cluster_conf_dir("common", config.get_twister2_conf_dir())
    java_defines.append("java.util.logging.config.file=" + conf_dir_common + "/logger.properties")

    # invoke the runtime manager to kill the job
    result = execute.twister2_class(
        'edu.iu.dsc.tws.rsched.core.RuntimeManagerMain',
        lib_jars,
        extra_jars=[],
        args=new_args,
        java_defines=java_defines
    )

    err_msg = "Failed to %s" % (action)
    succ_msg = "Successfully %s" % (action)
    result.add_context(err_msg, succ_msg)
    return result
