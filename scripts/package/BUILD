package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_deb", "pkg_tar")
load("//tools/rules:twister2_client.bzl", "twister2_client_bin_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_conf_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_slurm_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_standalone_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_aurora_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_kubernetes_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_kubernetes_deployment_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_mesos_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_common_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_nomad_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_dashboard_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_lib_third_party_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_lib_task_scheduler_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_lib_resource_scheduler_files")
load("//tools/rules:twister2_client.bzl", "twister2_curator_zookeeper_lib_files")
load("//tools/rules:twister2_client.bzl", "twister2_kubernetes_lib_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_lib_api_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_lib_task_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_lib_data_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_lib_data_lmdb_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_lib_communication_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_example_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_python_example_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_lib_common_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_lib_executor_files")
load("//tools/rules:twister2_client.bzl", "twister2_client_lib_connector_files")
load("//tools/rules:twister2_client.bzl", "twister2_ucx_native_libs")
load("//tools/rules:twister2_core.bzl", "twister2_core_files")
load("//tools/rules:twister2_core.bzl", "twister2_core_conf_files")
load("//tools/rules:twister2_core.bzl", "twister2_core_lib_communication_files")
load("//tools/rules:twister2_core.bzl", "twister2_core_lib_resource_scheduler_files")
load("//tools/rules:twister2_core.bzl", "twister2_core_lib_task_scheduler_files")
load("//tools/rules:twister2_core.bzl", "twister2_core_lib_connector_files")
load("//tools/rules:twister2_core.bzl", "twister2_core_lib_data_files")
load("//tools/rules:twister2_core.bzl", "twister2_core_lib_executor_files")
load("//tools/rules:twister2_core.bzl", "twister2_core_lib_data_lmdb_files")
load("//tools/rules:twister2_core.bzl", "twister2_harp_integration_files")
load("//tools/rules:twister2_core.bzl", "twister2_dashboard_files")
load("//tools/rules:twister2_core.bzl", "twister2_deeplearning_files")
load("//tools/rules:twister2_core.bzl", "twister2_storm_files")
load("//tools/rules:twister2_core.bzl", "twister2_beam_files")
load("//tools/rules:twister2_core.bzl", "twister2_core_checkpointing_files")
load("//tools/rules:twister2_core.bzl", "twister2_core_tset_files")
load("//tools/rules:twister2_core.bzl", "twister2_core_dl_files")
load("//tools/rules:twister2_core.bzl", "twister2_python_support_files")
load("//tools/rules:twister2_master.bzl", "twister2_master_lib_files")
load("//tools/rules:twister2_master.bzl", "twister2_master_jersey_files")
load("//tools/rules:twister2_client.bzl", "twister2_arrow_lib_files")
load(":pack_functions.bzl", "pack_tw2")
load("//:t2_meta.bzl", "T2_VERSION")

generated_release_files = [
    "release.yaml",
]

filegroup(
    name = "release_files",
    srcs = generated_release_files,
)

filegroup(
    name = "tarpkgs",
    srcs = [
        ":twister2-%s" % T2_VERSION,
        ":twister2-core-%s" % T2_VERSION,
    ],
)

filegroup(
    name = "tarpkgs_no_mpi",
    srcs = [
        ":twister2-no-mpi-%s" % T2_VERSION,
        ":twister2-core-no-mpi-%s" % T2_VERSION,
    ],
)

######################################################
# twister2 core packaging
######################################################

pack_tw2(
    name = "twister2-core-no-mpi",
    extension = "tar.gz",
    # srcs = generated_release_files,
    package_dir = "twister2-core",
    version = T2_VERSION,
    deps = [
        ":twister2-arrow-lib",
        ":twister2-checkpointing",
        ":twister2-core-lib-api",
        ":twister2-core-lib-common",
        ":twister2-core-lib-communication",
        ":twister2-core-lib-connector",
        ":twister2-core-lib-data",
        ":twister2-core-lib-executor",
        ":twister2-core-lib-lmdb",
        ":twister2-core-lib-resource-scheduler",
        ":twister2-core-lib-task",
        ":twister2-core-lib-task-scheduler",
        ":twister2-core-lib-third_party",
        ":twister2-curator-zookeeper-libs",
        ":twister2-kubernetes-libs",
        ":twister2-master-jersey",
        ":twister2-master-lib",
        ":twister2-python-support",
        ":twister2-tset",
        ":twister2-dl",
    ],
)

pack_tw2(
    name = "twister2-core",
    extension = "tar.gz",
    # srcs = generated_release_files,
    package_dir = "twister2-core",
    version = T2_VERSION,
    deps = [
        ":twister2-arrow-lib",
        ":twister2-beam",
        ":twister2-checkpointing",
        ":twister2-core-lib-api",
        ":twister2-core-lib-common",
        ":twister2-core-lib-communication",
        ":twister2-core-lib-connector",
        ":twister2-core-lib-data",
        ":twister2-core-lib-executor",
        ":twister2-core-lib-lmdb",
        ":twister2-core-lib-resource-scheduler",
        ":twister2-core-lib-task",
        ":twister2-core-lib-task-scheduler",
        ":twister2-core-lib-third_party",
        ":twister2-curator-zookeeper-libs",
        ":twister2-harp-libs",
        ":twister2-kubernetes-libs",
        ":twister2-master-jersey",
        ":twister2-master-lib",
        ":twister2-ompi-bin",
        ":twister2-ompi-etc",
        ":twister2-ompi-include",
        ":twister2-ompi-jars",
        ":twister2-ompi-lib",
        ":twister2-python-support",
        ":twister2-tset",
        ":twister2-dl",
    ],
)

pack_tw2(
    name = "twister2-ompi-lib",
    srcs = select({
        "//tools/platform:darwin": ["@ompi3darwin//:ompi-lib-files"],
        "//conditions:default": ["@ompi3//:ompi-lib-files"],
    }),
    package_dir = "ompi/lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-ompi-bin",
    srcs = select({
        "//tools/platform:darwin": ["@ompi3darwin//:ompi-bin-files"],
        "//conditions:default": ["@ompi3//:ompi-bin-files"],
    }),
    package_dir = "ompi/bin",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-ompi-include",
    srcs = select({
        "//tools/platform:darwin": ["@ompi3darwin//:ompi-include-files"],
        "//conditions:default": ["@ompi3//:ompi-include-files"],
    }),
    package_dir = "ompi/include",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-ompi-etc",
    srcs = select({
        "//tools/platform:darwin": ["@ompi3darwin//:ompi-etc-files"],
        "//conditions:default": ["@ompi3//:ompi-etc-files"],
    }),
    package_dir = "ompi/etc",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-ompi-jars",
    srcs = select({
        "//tools/platform:darwin": ["@ompi3darwin//:ompi-jar-files"],
        "//conditions:default": ["@ompi3//:ompi-jar-files"],
    }),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-core-lib-api",
    srcs = twister2_client_lib_api_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-core-lib-task",
    srcs = twister2_client_lib_task_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-core-lib-resource-scheduler",
    srcs = twister2_client_lib_resource_scheduler_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-kubernetes-libs",
    srcs = twister2_kubernetes_lib_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-curator-zookeeper-libs",
    srcs = twister2_curator_zookeeper_lib_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-core-lib-task-scheduler",
    srcs = twister2_client_lib_task_scheduler_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-core-lib-third_party",
    srcs = twister2_client_lib_third_party_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-core-lib-communication",
    srcs = twister2_client_lib_communication_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-core-lib-common",
    srcs = twister2_client_lib_common_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-arrow-lib",
    srcs = twister2_arrow_lib_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-core-lib-executor",
    srcs = twister2_client_lib_executor_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-core-lib-connector",
    srcs = twister2_client_lib_connector_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-core-lib-data",
    srcs = twister2_client_lib_data_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-core-lib-lmdb",
    srcs = twister2_core_lib_data_lmdb_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

######################################################
# twister2 client packaging
######################################################

pack_tw2(
    name = "twister2-no-mpi",
    extension = "tar.gz",
    # srcs = generated_release_files,
    package_dir = "twister2-%s" % T2_VERSION,
    version = T2_VERSION,
    deps = [
        ":twister2-arrow-lib",
        ":twister2-bin",
        ":twister2-checkpointing",
        ":twister2-communication",
        ":twister2-conf-aurora",
        ":twister2-conf-dashboard",
        ":twister2-conf-kubernetes",
        ":twister2-conf-kubernetes-deployment",
        ":twister2-conf-mesos",
        ":twister2-conf-nomad",
        ":twister2-conf-slurm",
        ":twister2-conf-standalone",
        ":twister2-core-lib-executor",
        ":twister2-curator-zookeeper-libs",
        ":twister2-dashboard",
        ":twister2-deeplearning",
        ":twister2-dist",
        ":twister2-examples",
        ":twister2-fault-tolerance",
        ":twister2-harp-libs",
        ":twister2-kubernetes-libs",
        ":twister2-lib-api",
        ":twister2-lib-common",
        ":twister2-lib-connector",
        ":twister2-lib-data",
        ":twister2-lib-data-lmdb",
        ":twister2-lib-resource-scheduler",
        ":twister2-lib-task",
        ":twister2-lib-task-scheduler",
        ":twister2-lib-third_party",
        ":twister2-master-jersey",
        ":twister2-master-lib",
        ":twister2-python-examples",
        ":twister2-python-support",
        ":twister2-tset",
        ":twister2-dl",
        ":twister2-ucx-native-libs",
        ":twister2-ucx-native-libs-dir"
    ],
)

pack_tw2(
    name = "twister2",
    extension = "tar.gz",
    # srcs = generated_release_files,
    package_dir = "twister2-%s" % T2_VERSION,
    version = T2_VERSION,
    deps = [
        ":twister2-arrow-lib",
        ":twister2-beam",
        ":twister2-bin",
        ":twister2-checkpointing",
        ":twister2-communication",
        ":twister2-conf-aurora",
        ":twister2-conf-common",
        ":twister2-conf-dashboard",
        ":twister2-conf-kubernetes",
        ":twister2-conf-kubernetes-deployment",
        ":twister2-conf-mesos",
        ":twister2-conf-nomad",
        ":twister2-conf-slurm",
        ":twister2-conf-standalone",
        ":twister2-core-lib-executor",
        ":twister2-curator-zookeeper-libs",
        ":twister2-dashboard",
        ":twister2-deeplearning",
        ":twister2-dist",
        ":twister2-examples",
        ":twister2-harp-libs",
        ":twister2-kubernetes-libs",
        ":twister2-lib-api",
        ":twister2-lib-common",
        ":twister2-lib-connector",
        ":twister2-lib-data",
        ":twister2-lib-data-lmdb",
        ":twister2-lib-resource-scheduler",
        ":twister2-lib-task",
        ":twister2-lib-task-scheduler",
        ":twister2-lib-third_party",
        ":twister2-master-jersey",
        ":twister2-master-lib",
        ":twister2-ompi-bin",
        ":twister2-ompi-etc",
        ":twister2-ompi-include",
        ":twister2-ompi-lib",
        ":twister2-python-examples",
        ":twister2-python-support",
        ":twister2-storm",
        ":twister2-test",
        ":twister2-test-tests",
        ":twister2-test-tests-apps-tera-sort",
        ":twister2-test-tests-comms",
        ":twister2-test-tests-integration",
        ":twister2-test-tests-integration-comms",
        ":twister2-test-tests-integration-task",
        ":twister2-test-tree",
        ":twister2-test-util",
        ":twister2-test-values",
        ":twister2-tset",
        ":twister2-dl",
        ":twister2-ucx-native-libs",
        ":twister2-ucx-native-libs-dir"
    ],
)

pack_tw2(
    name = "twister2-conf-slurm",
    srcs = twister2_client_slurm_files(),
    package_dir = "conf/slurm",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-conf-standalone",
    srcs = twister2_client_standalone_files(),
    package_dir = "conf/standalone",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-conf-aurora",
    srcs = twister2_client_aurora_files(),
    package_dir = "conf/aurora",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-conf-kubernetes",
    srcs = twister2_client_kubernetes_files(),
    package_dir = "conf/kubernetes",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-conf-kubernetes-deployment",
    srcs = twister2_client_kubernetes_deployment_files(),
    package_dir = "conf/kubernetes/deployment",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-conf-common",
    srcs = twister2_client_common_files(),
    package_dir = "conf/common",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-conf-mesos",
    srcs = twister2_client_mesos_files(),
    package_dir = "conf/mesos",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-conf-nomad",
    srcs = twister2_client_nomad_files(),
    package_dir = "conf/nomad",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-conf-dashboard",
    srcs = twister2_client_dashboard_files(),
    package_dir = "conf/dashboard",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-dist",
    srcs = [":twister2-core-%s" % T2_VERSION],
    package_dir = "dist",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-bin",
    srcs = twister2_client_bin_files(),
    package_dir = "bin",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-lib-api",
    srcs = twister2_client_lib_api_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-lib-task",
    srcs = twister2_client_lib_task_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-lib-data",
    srcs = twister2_client_lib_data_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-lib-data-lmdb",
    srcs = twister2_client_lib_data_lmdb_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-lib-resource-scheduler",
    srcs = twister2_client_lib_resource_scheduler_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-lib-task-scheduler",
    srcs = twister2_client_lib_task_scheduler_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-lib-third_party",
    srcs = twister2_client_lib_third_party_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-communication",
    srcs = twister2_client_lib_communication_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-lib-common",
    srcs = twister2_client_lib_common_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-lib-connector",
    srcs = twister2_client_lib_connector_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-examples",
    srcs = twister2_client_example_files(),
    package_dir = "examples",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-python-examples",
    srcs = twister2_client_python_example_files(),
    package_dir = "examples/python",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-master-lib",
    srcs = twister2_master_lib_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-master-jersey",
    srcs = twister2_master_jersey_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-harp-libs",
    srcs = twister2_harp_integration_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-dashboard",
    srcs = twister2_dashboard_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-deeplearning",
    srcs = twister2_deeplearning_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-storm",
    srcs = twister2_storm_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-beam",
    srcs = twister2_beam_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-python-support",
    srcs = twister2_python_support_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-checkpointing",
    srcs = twister2_core_checkpointing_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-tset",
    srcs = twister2_core_tset_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-dl",
    srcs = twister2_core_dl_files(),
    package_dir = "lib",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-test",
    srcs = ["//util/test"],
    package_dir = "util/test",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-test-values",
    srcs = ["//util/test:test_values"],
    package_dir = "util/test/values",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-test-util",
    srcs = ["//util/test:test_util"],
    package_dir = "util/test/util",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-test-tree",
    srcs = ["//util/test:test_tree"],
    package_dir = "util/test/tree",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-test-tests",
    srcs = ["//util/test:test_tests"],
    package_dir = "util/test/tests",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-test-tests-apps-tera-sort",
    srcs = ["//util/test:test_tests_apps_terasort"],
    package_dir = "util/test/tests/apps/terasort",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-test-tests-comms",
    srcs = ["//util/test:test_tests_comms"],
    package_dir = "util/test/tests/comms",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-test-tests-integration",
    srcs = ["//util/test:test_tests_integration"],
    package_dir = "util/test/tests/integration",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-test-tests-integration-comms",
    srcs = ["//util/test:test_tests_integration_comms"],
    package_dir = "util/test/tests/integration/comms",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-test-tests-integration-task",
    srcs = ["//util/test:test_tests_integration_task"],
    package_dir = "util/test/tests/integration/task",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-ucx-native-libs",
    srcs = twister2_ucx_native_libs(),
    package_dir = "lib/ucx",
    version = T2_VERSION,
)

pack_tw2(
    name = "twister2-ucx-native-libs-dir",
    srcs = ["EMPTY"],
    package_dir = "lib/ucx/ucx",
    version = T2_VERSION,
)
