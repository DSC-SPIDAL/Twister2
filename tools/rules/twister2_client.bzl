# Twister2 client files

def twister2_client_bin_files():
    return [
        "//twister2/tools/cli/src/python:twister2",
        "//third_party/nomad:nomad",
    ]

def twister2_client_conf_files():
    return [
        "//twister2/config/src/yaml:conf-common-yaml",
        "//twister2/config/src/yaml:conf-yaml",
        "//twister2/config/src/yaml:conf-local-yaml",
        "//twister2/config/src/yaml:conf-slurm-yaml",
        "//twister2/config/src/yaml:conf-standalone-yaml",
        "//twister2/config/src/yaml:conf-aurora-yaml",
        "//twister2/config/src/yaml:conf-kubernetes-yaml",
        "//twister2/config/src/yaml:conf-mesos-yaml",
        "//twister2/config/src/yaml:conf-dashboard",
    ]

def twister2_client_dashboard_files():
    return [
        "//twister2/config/src/yaml:conf-dashboard",
    ]

def twister2_client_common_files():
    return [
        "//twister2/config/src/yaml:conf-common-yaml",
    ]

def twister2_client_local_files():
    return [
        "//twister2/config/src/yaml:conf-local-yaml",
    ]

def twister2_client_standalone_files():
    return [
        "//twister2/config/src/yaml:conf-standalone-yaml",
    ]

def twister2_client_slurm_files():
    return [
        "//twister2/config/src/yaml:conf-slurm-yaml",
    ]

def twister2_client_aurora_files():
    return [
        "//twister2/config/src/yaml:conf-aurora-yaml",
    ]

def twister2_client_kubernetes_files():
    return [
        "//twister2/config/src/yaml:conf-kubernetes-yaml",
    ]

def twister2_client_kubernetes_deployment_files():
    return [
        "//twister2/config/src/yaml:conf-kubernetes-deployment-yaml",
    ]

def twister2_client_mesos_files():
    return [
        "//twister2/config/src/yaml:conf-mesos-yaml",
    ]

def twister2_client_nomad_files():
    return [
        "//twister2/config/src/yaml:conf-nomad-yaml",
    ]

def twister2_client_lib_task_scheduler_files():
    return [
        "//twister2/taskscheduler/src/java:taskscheduler-java",
    ]

def twister2_client_lib_resource_scheduler_files():
    return [
        "//twister2/resource-scheduler/src/java:resource-scheduler-java",
        "@commons_cli_commons_cli//jar",
        "//twister2/proto:proto-java",
        "//third_party:ompi_javabinding_java",
        "@com_google_guava_guava//jar",
        "@commons_io_commons_io//jar",
        "@org_apache_commons_commons_compress//jar",
        "@io_swagger_swagger_annotations//jar",
        "@com_google_code_gson_gson//jar",
        "@commons_codec_commons_codec//jar",
        "@com_hashicorp_nomad_nomad_sdk//jar",
        "@com_fasterxml_jackson_core_jackson_annotations//jar",
        "@com_fasterxml_jackson_core_jackson_core//jar",
        "@com_fasterxml_jackson_core_jackson_databind//jar",
        "@com_google_code_findbugs_jsr305//jar",
        "@commons_logging_commons_logging//jar",
        "@org_apache_httpcomponents_httpclient//jar",
        "@org_apache_httpcomponents_httpcore//jar",
        "@org_bouncycastle_bcpkix_jdk15on//jar",
        "@org_bouncycastle_bcprov_jdk15on//jar",
        "@com_google_protobuf//:protobuf_java",
    ]

def twister2_kubernetes_lib_files():
    return [
        "@maven//:io_kubernetes_client_java",
        "@maven//:io_kubernetes_client_java_api",
        "@maven//:io_kubernetes_client_java_proto",
        "@com_squareup_okhttp_okhttp//jar",
        "@com_squareup_okhttp_logging_interceptor//jar",
        "@com_squareup_okhttp_okhttp_ws//jar",
        "@com_squareup_okio_okio//jar",
        "@joda_time_joda_time//jar",
    ]

def twister2_curator_zookeeper_lib_files():
    return [
        "@maven//:org_apache_curator_curator_client",
        "@maven//:org_apache_curator_curator_framework",
        "@maven//:org_apache_curator_curator_recipes",
        "@maven//:org_apache_zookeeper_zookeeper",
        "@maven//:org_apache_zookeeper_zookeeper_jute",
    ]

def twister2_client_lib_api_files():
    return [
        "//twister2/api/src/java:api-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/checkpointing:checkpointing-api-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/comms:comms-api-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/config:config-api-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/data:data-api-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/dataset:dataset-api-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/exceptions:exceptions-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/faulttolerance:fault-tolerance-api-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/net:network-api-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/resource:resource-api-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/scheduler:scheduler-api-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/compute:task-api-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/tset:tset-api-java",
        "//twister2/api/src/java/edu/iu/dsc/tws/api/util:api-utils-java",
        "//twister2/proto/utils:proto-utils-java",
    ]

def twister2_client_lib_task_files():
    return [
        "//twister2/task/src/main/java:task-java",
    ]

def twister2_client_lib_data_files():
    return [
        "//twister2/data/src/main/java:data-java",
        "@org_apache_hadoop_hadoop_hdfs//jar",
        "@org_apache_hadoop_hadoop_common//jar",
        "@org_apache_hadoop_hadoop_annotations//jar",
        "@org_apache_hadoop_hadoop_auth//jar",
        "@org_apache_hadoop_hadoop_mapreduce_client_core//jar",
        "@com_google_code_findbugs_jsr305//jar",
        "@com_fasterxml_woodstox_woodstox_core//jar",
        "@org_codehaus_woodstox_stax2_api//jar",
        "@commons_io_commons_io//jar",
        "@commons_collections_commons_collections//jar",
        "@org_apache_commons_commons_lang3//jar",
        "@commons_lang_commons_lang//jar",
        "@commons_configuration_commons_configuration//jar",
        "@log4j_log4j//jar",
        "@org_apache_htrace_htrace_core4//jar",
        "@org_apache_hadoop_hadoop_hdfs_client//jar",
    ]

def twister2_client_lib_connector_files():
    return [
        "//twister2/connectors/src/java:connector-java",
        "@org_xerial_snappy_snappy_java//jar",
        "@org_lz4_lz4_java//jar",
        "@org_slf4j_slf4j_api//jar",
        "@org_apache_kafka_kafka_clients//jar",
    ]

def twister2_client_lib_executor_files():
    return [
        "//twister2/executor/src/java:executor-java",
    ]

def twister2_client_lib_data_lmdb_files():
    return [
        "//twister2/data/src/main/java:data-java",
        "@org_lmdbjava_lmdbjava//jar",
        "@org_lmdbjava_lmdbjava_native_linux_x86_64//jar",
        "@org_lmdbjava_lmdbjava_native_windows_x86_64//jar",
        "@org_lmdbjava_lmdbjava_native_osx_x86_64//jar",
        "@com_github_jnr_jnr_ffi//jar",
        "@com_github_jnr_jnr_constants//jar",
        "@com_github_jnr_jffi//jar",
        "//third_party:com_github_jnr_jffi_native",
    ]

def twister2_client_lib_communication_files():
    return [
        "//twister2/comms/src/java:comms-java",
        "@org_yaml_snakeyaml//jar",
        "@com_esotericsoftware_kryo//jar",
        "@com_google_guava_guava//jar",
        "@org_apache_commons_commons_lang3//jar",
        "@org_apache_commons_commons_collections4//jar",
        "@org_objenesis_objenesis//jar",
        "@com_esotericsoftware_minlog//jar",
        "@com_esotericsoftware_reflectasm//jar",
        "@org_ow2_asm_asm//jar",
        "//third_party:ompi_javabinding_java",
    ]

def twister2_client_lib_common_files():
    return [
        "//twister2/common/src/java:common-java",
    ]

def twister2_client_example_files():
    return [
        "//twister2/examples/src/java:examples-java",
    ]

def twister2_client_python_example_files():
    return [
        "//twister2/examples/src/python:examples-python",
    ]

def twister2_client_lib_third_party_files():
    return [
        "@org_slf4j_slf4j_api//jar",
        "@org_slf4j_slf4j_jdk14//jar",
    ]

def twister2_client_lib_master_files():
    return [
        "//twister2/connectors/src/java:master-java",
    ]

def twister2_client_lib_arrow_files():
    return [
            "//twister2/arrow/src/main/java:arrow-java",
            "@org_apache_arrow_arrow_vector",
            "@org_apache_arrow_arrow_memory",
            "@io_netty_netty_buffer",
            "@io_netty_netty_common",
            "@com_google_flatbuffers_flatbuffers_java",
        ]

#def twister2_client_lib_connector_files():
#    return [
#        "//twister2/connectors/src/java:connector-java",
#        "@org_xerial_snappy_snappy_java//jar",
#        "@org_lz4_lz4_java//jar",
#        "@org_slf4j_slf4j_api//jar",
#        "@org_apache_kafka_kafka_clients//jar",
#        "@org_apache_kafka_kafka_clients//jar",
#    ]
