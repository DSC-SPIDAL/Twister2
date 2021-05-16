#reference : https://github.com/google/bazel-common/blob/master/workspace_defs.bzl

load("@bazel_tools//tools/build_defs/repo:java.bzl", "java_import_external")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@rules_jvm_external//:defs.bzl", "maven_install")

_MAVEN_MIRRORS = [
    "https://repo1.maven.org/maven2/",
    "https://repo.maven.apache.org/maven2/",
    "https://maven.google.com",
]

deps = []

def _maven_import(artifact, sha256, licenses, **kwargs):
    parts = artifact.split(":")
    group_id = parts[0]
    artifact_id = parts[1]
    version = parts[2]
    name = ("%s_%s" % (group_id, artifact_id)).replace(".", "_").replace("-", "_")
    url_suffix = "{0}/{1}/{2}/{1}-{2}.jar".format(group_id.replace(".", "/"), artifact_id, version)

    #    if (name in deps):
    #        print("%s has defined multiple times" % name)
    #
    #    deps.append(name)

    java_import_external(
        name = name,
        jar_urls = [base + url_suffix for base in _MAVEN_MIRRORS],
        jar_sha256 = sha256,
        licenses = licenses,
        tags = ["maven_coordinates=" + artifact],
        **kwargs
    )

def load_modules():
    #Protocol Buffers
    _maven_import(
        artifact = "com.google.protobuf:protobuf-java:3.9.0",
        licenses = ["notice"],
        sha256 = "6c96d85eac237fea84d9d5e7413c85b62f2df0b9f7b17b0168bd1e28b09ff0e8",
    )

    #Guava
    _maven_import(
        artifact = "com.google.guava:guava:25.0-jre",
        licenses = ["notice"],
        sha256 = "3fd4341776428c7e0e5c18a7c10de129475b69ab9d30aeafbb5c277bb6074fa9",
    )

    #Skylib
    skylib_version = "9430df29e4c648b95bf39a57e4336b44a0a0582a"

    http_archive(
        name = "bazel_skylib",
        strip_prefix = "bazel-skylib-{}".format(skylib_version),
        urls = ["https://github.com/bazelbuild/bazel-skylib/archive/{}.zip".format(skylib_version)],
    )

    maven_install(
        name = "maven",
        artifacts = [
            "com.opencsv:opencsv:5.0",
            "org.powermock:powermock-module-junit4-common:1.6.2",
            "org.apache.curator:curator-client:4.2.0",
            "org.apache.curator:curator-framework:4.2.0",
            "org.apache.curator:curator-recipes:4.2.0",
            "org.apache.zookeeper:zookeeper:3.5.6",
            "org.apache.zookeeper:zookeeper-jute:3.5.6",
            "io.kubernetes:client-java:7.0.0",
            "io.kubernetes:client-java-proto:7.0.0",
            "io.kubernetes:client-java-api:7.0.0",
            "com.squareup.okhttp3:okhttp:3.14.3",
            "com.squareup.okhttp3:logging-interceptor:3.14.3",
            "com.squareup.okio:okio:1.17.2",
            "com.google.code.gson:gson:2.8.0",
            "io.gsonfire:gson-fire:1.8.3",
            "com.google.re2j:re2j:1.3",
            "org.apache.arrow:arrow-vector:0.16.0",
            "org.apache.arrow:arrow-memory:0.16.0",
            "org.apache.arrow:arrow-format:0.16.0",
            "com.google.flatbuffers:flatbuffers-java:1.9.0",
            "javax.annotation:javax.annotation-api:1.3.2",
            "io.netty:netty-buffer:4.1.27.Final",
            "io.netty:netty-common:4.1.27.Final",
            "io.netty:netty-all:4.1.27.Final",
            "io.netty:netty-transport:4.1.27.Final",
            "io.netty:netty-transport-native-epoll:4.1.27.Final",
            "org.apache.commons:commons-math3:3.6.1",
        ],
        repositories = [
            "https://repo1.maven.org/maven2",
            "https://maven.google.com",
        ],
        fetch_sources = False,  # Fetch source jars. Defaults to False.
        fail_on_missing_checksum = False,
    )

    #Generated MVN artifacts
    #    _maven_import(artifact = "org.powermock:powermock-module-junit4-common:1.6.2", licenses = ["notice"], sha256 = "d3911d010a954ddd912d6d4f5dde5eed0bd6535936654c69a9b63789a0b08723")
    _maven_import(artifact = "org.powermock:powermock-api-support:1.6.2", licenses = ["notice"], sha256 = "89e32d0c53dac114ea5e6506b140cf441a7964bde7abba6caacaa3cffa09f0ea")
    _maven_import(artifact = "org.slf4j:slf4j-jdk14:1.7.7", licenses = ["notice"], sha256 = "9909915e991269e5f3f4d8eb51fb0a1da7ff8bc6174a4cf25970f373abafcf9b")
    _maven_import(artifact = "com.esotericsoftware:minlog:1.3.0", licenses = ["notice"], sha256 = "f7b399d3a5478a4f3e0d98bd1c9f47766119c66414bc33aa0f6cde0066f24cc2")
    _maven_import(artifact = "io.swagger:swagger-annotations:1.5.12", licenses = ["notice"], sha256 = "3607ffca7ceaca1f5257a454c322237ea4559f1a0c906da2634864b52215d9c0")
    _maven_import(artifact = "org.slf4j:slf4j-api:1.7.25", licenses = ["notice"], sha256 = "18c4a0095d5c1da6b817592e767bb23d29dd2f560ad74df75ff3961dbde25b79")
    _maven_import(artifact = "org.powermock:powermock-reflect:1.6.2", licenses = ["notice"], sha256 = "94c0ea545990f1e439de77e4b6dafe32090d6276eb43a99df9e50c6c8845d57d")
    _maven_import(artifact = "org.powermock:powermock-module-junit4:1.6.2", licenses = ["notice"], sha256 = "c0cbdaa81a19b93095909de41afedeb7d499b828984a4511a6f20d937a70a67c")
    _maven_import(artifact = "org.powermock:powermock-api-mockito:1.6.2", licenses = ["notice"], sha256 = "a5e0be1d52982c81b9c0169622a9ef66d9398eaefd858b43029d16b7a773b7df")
    _maven_import(artifact = "commons-logging:commons-logging:1.1.1", licenses = ["notice"], sha256 = "ce6f913cad1f0db3aad70186d65c5bc7ffcc9a99e3fe8e0b137312819f7c362f")
    _maven_import(artifact = "org.lmdbjava:lmdbjava-native-osx-x86_64:0.9.21-1", licenses = ["notice"], sha256 = "5ebe0302edd76cb63b25fdfea86f75af1e917c0c0355e12f939fccb3c83409bc")
    _maven_import(artifact = "org.ow2.asm:asm:4.2", licenses = ["notice"], sha256 = "3c7e45fe303bd02193d951df134255033b9d8147e77508d09703bac245e6cd9b")
    _maven_import(artifact = "junit:junit:4.11", licenses = ["notice"], sha256 = "90a8e1603eeca48e7e879f3afbc9560715322985f39a274f6f6070b43f9d06fe")
    _maven_import(artifact = "com.google.code.findbugs:jsr305:3.0.0", licenses = ["notice"], sha256 = "bec0b24dcb23f9670172724826584802b80ae6cbdaba03bdebdef9327b962f6a")
    _maven_import(artifact = "com.esotericsoftware:reflectasm:1.10.0", licenses = ["notice"], sha256 = "0eea1d90c566eba2a536b4e7ede0138d3ccde59a001d83e157cbf5649d540975")
    _maven_import(artifact = "com.fasterxml.jackson.core:jackson-annotations:2.8.8", licenses = ["notice"], sha256 = "1ff7b1c91658506f1050b39d1564eb4d5dc63586dd709bad58428a63775d75a8")
    _maven_import(artifact = "commons-cli:commons-cli:1.3.1", licenses = ["notice"], sha256 = "3a2f057041aa6a8813f5b59b695f726c5e85014a703d208d7e1689098e92d8c0")
    _maven_import(artifact = "org.apache.httpcomponents:httpmime:4.4", licenses = ["notice"], sha256 = "91d9abfebb3a404090106e10bc6bb0fd33072375c9df27f0f91a4c613289bd4b")
    _maven_import(artifact = "com.github.jnr:jffi:1.2.16", licenses = ["notice"], sha256 = "7a616bb7dc6e10531a28a098078f8184df9b008d5231bdc5f1c131839385335f")
    _maven_import(artifact = "org.lz4:lz4-java:1.6.0", licenses = ["notice"], sha256 = "d229545aa2b1d5203c876614bdbcffcacc303697f4f8f26f764e1d6c1ed2e416")
    _maven_import(artifact = "org.apache.hadoop:hadoop-auth:3.2.1", licenses = ["notice"], sha256 = "81645fff08e0b8bd464e033b1b2e10f70b4f7f81a82e057fcb2be88be29a94d5")

    #    _maven_import(artifact = "org.lz4:lz4-java:1.4", licenses = ["notice"], sha256 = "9ed51eb236340cab58780ed7d20741ff812bcb3875beb974fa7cf9ddea272358")
    _maven_import(artifact = "org.apache.hadoop:hadoop-auth:3.2.1", licenses = ["notice"], sha256 = "81645fff08e0b8bd464e033b1b2e10f70b4f7f81a82e057fcb2be88be29a94d5")
    _maven_import(artifact = "org.lmdbjava:lmdbjava-native-linux-x86_64:0.9.21-1", licenses = ["notice"], sha256 = "d0905caca075b5b63e598398514253dabc1ee7604fd5e12050513aeb6435ac0d")
    _maven_import(artifact = "net.openhft:chronicle-queue:4.6.55", licenses = ["notice"], sha256 = "cc37d54a902b2e125389de06d7e273c6ed366e92d520e5782eb109d933070e6c")
    _maven_import(artifact = "org.yaml:snakeyaml:1.15", licenses = ["notice"], sha256 = "79ea8aac6590f49ee8390c2f17ed9343079e85b44158a097b301dfee42af86ec")
    _maven_import(artifact = "commons-io:commons-io:2.5", licenses = ["notice"], sha256 = "a10418348d234968600ccb1d988efcbbd08716e1d96936ccc1880e7d22513474")
    _maven_import(artifact = "org.lmdbjava:lmdbjava-native-windows-x86_64:0.9.21-1", licenses = ["notice"], sha256 = "f1fce7ddc18cbd9d3f5637d04ef2891ecb0717b6ad86b2372988a039e06ed1d6")
    _maven_import(artifact = "org.apache.httpcomponents:httpcore:4.4.5", licenses = ["notice"], sha256 = "64d5453874cab7e40a7065cb01a9a9ca1053845a9786b478878b679e0580cec3")
    _maven_import(artifact = "commons-io:commons-io:2.6", licenses = ["notice"], sha256 = "f877d304660ac2a142f3865badfc971dec7ed73c747c7f8d5d2f5139ca736513")
    _maven_import(artifact = "com.fasterxml.jackson.core:jackson-core:2.8.8", licenses = ["notice"], sha256 = "d9bde8c72c22202bf17b05c7811db4964ff8e843d97c00a9bfb048c0fe7a726b")
    _maven_import(artifact = "com.esotericsoftware:kryo:4.0.2", licenses = ["notice"], sha256 = "8cd8bcb6920b165ff21b82f7a44f2c7e87dcb24560e52b9dc7d085fba9cfcce5")
    _maven_import(artifact = "com.hashicorp.nomad:nomad-sdk:0.7.0", licenses = ["notice"], sha256 = "d04dda58d0242f87e66b333f5143d0f3aabb7e8ce1653c0ed1293a648cb18541")
    _maven_import(artifact = "commons-codec:commons-codec:1.11", licenses = ["notice"], sha256 = "e599d5318e97aa48f42136a2927e6dfa4e8881dff0e6c8e3109ddbbff51d7b7d")
    _maven_import(artifact = "commons-configuration:commons-configuration:1.6", licenses = ["notice"], sha256 = "46b71b9656154f6a16ea4b1dc84026b52a9305f8eff046a2b4655fa1738e5eee")
    _maven_import(artifact = "com.github.jnr:jnr-constants:0.9.9", licenses = ["notice"], sha256 = "6862e69646fb726684d8610bc5a65740feab5f235d8d1dc7596113bd1ad54181")
    _maven_import(artifact = "commons-beanutils:commons-beanutils:1.9.2", licenses = ["notice"], sha256 = "23729e3a2677ed5fb164ec999ba3fcdde3f8460e5ed086b6a43d8b5d46998d42")
    _maven_import(artifact = "org.lmdbjava:lmdbjava:0.6.0", licenses = ["notice"], sha256 = "1b02194c292c767fe3b2af8af85c43101cb54d45fbe16c148f78cb89abeb11de")
    _maven_import(artifact = "org.powermock:powermock-core:1.6.2", licenses = ["notice"], sha256 = "48cc45502caa34c017911c6f153b0269dfa731ec706fb196072c8b0d938c4433")
    _maven_import(artifact = "com.google.code.findbugs:jsr305:3.0.2", licenses = ["notice"], sha256 = "766ad2a0783f2687962c8ad74ceecc38a28b9f72a2d085ee438b7813e928d0c7")
    _maven_import(artifact = "org.slf4j:slf4j-api:1.7.7", licenses = ["notice"], sha256 = "69980c038ca1b131926561591617d9c25fabfc7b29828af91597ca8570cf35fe")
    _maven_import(artifact = "org.mockito:mockito-all:1.10.19", licenses = ["notice"], sha256 = "d1a7a7ef14b3db5c0fc3e0a63a81b374b510afe85add9f7984b97911f4c70605")
    _maven_import(artifact = "org.apache.hadoop:hadoop-annotations:3.2.1", licenses = ["notice"], sha256 = "4b92c11ae697b51c8e66950ab3a3572bc4e09e378ff4f5ba2e6b82167c38da7f")
    _maven_import(artifact = "org.objenesis:objenesis:2.1", licenses = ["notice"], sha256 = "c74330cc6b806c804fd37e74487b4fe5d7c2750c5e15fbc6efa13bdee1bdef80")
    _maven_import(artifact = "org.bouncycastle:bcpkix-jdk15on:1.56", licenses = ["notice"], sha256 = "7043dee4e9e7175e93e0b36f45b1ec1ecb893c5f755667e8b916eb8dd201c6ca")
    _maven_import(artifact = "joda-time:joda-time:2.9.3", licenses = ["notice"], sha256 = "a05f5b8b021802a71919b18702aebdf286148188b3ee9d26e6ec40e8d0071487")
    _maven_import(artifact = "org.apache.commons:commons-compress:1.15", licenses = ["notice"], sha256 = "a778bbd659722889245fc52a0ec2873fbbb89ec661bc1ad3dc043c0757c784c4")
    _maven_import(artifact = "antlr:antlr:2.7.7", licenses = ["notice"], sha256 = "88fbda4b912596b9f56e8e12e580cc954bacfb51776ecfddd3e18fc1cf56dc4c")
    _maven_import(artifact = "org.apache.commons:commons-lang3:3.6", licenses = ["notice"], sha256 = "89c27f03fff18d0b06e7afd7ef25e209766df95b6c1269d6c3ebbdea48d5f284")
    _maven_import(artifact = "commons-lang:commons-lang:2.6", licenses = ["notice"], sha256 = "50f11b09f877c294d56f24463f47d28f929cf5044f648661c0f0cfbae9a2f49c")
    _maven_import(artifact = "org.apache.httpcomponents:httpclient:4.5.2", licenses = ["notice"], sha256 = "0dffc621400d6c632f55787d996b8aeca36b30746a716e079a985f24d8074057")
    _maven_import(artifact = "commons-collections:commons-collections:3.2.2", licenses = ["notice"], sha256 = "eeeae917917144a68a741d4c0dff66aa5c5c5fd85593ff217bced3fc8ca783b8")
    _maven_import(artifact = "com.github.jnr:jnr-ffi:2.1.7", licenses = ["notice"], sha256 = "2ed1bedf59935cd3cc0964bac5cd91638b2e966a82041fe0a6c85f52279c9b34")
    _maven_import(artifact = "org.xerial.snappy:snappy-java:1.1.4", licenses = ["notice"], sha256 = "f75ec0fa9c843e236c6e1512c17c095cfffd175f32e21ea0e3eccb540d77f002")
    _maven_import(artifact = "com.fasterxml.woodstox:woodstox-core:5.0.3", licenses = ["notice"], sha256 = "a1c04b64fbfe20ae9f2c60a3bf1633fed6688ae31935b6bd4a457a1bbb2e82d4")
    _maven_import(artifact = "org.bouncycastle:bcpkix-jdk15on:1.56", licenses = ["notice"], sha256 = "7043dee4e9e7175e93e0b36f45b1ec1ecb893c5f755667e8b916eb8dd201c6ca")
    _maven_import(artifact = "org.apache.kafka:kafka-clients:1.0.0", licenses = ["notice"], sha256 = "8644fba65a277c41831c8704c6cb49b146c8278f836abd8c2d04dbffdc1a7c4a")
    _maven_import(artifact = "org.apache.zookeeper:zookeeper:3.4.11", licenses = ["notice"], sha256 = "72d402ed238019b638aefb3b592ddde9c52cfbb7956aadcbd419b8c76febc1b1")

    #Kafka
    _maven_import(artifact = "org.apache.kafka:kafka-clients:2.3.1", licenses = ["notice"], sha256 = "1426faeb8385f3b2174428fae092260862bfd9b78504d3839e1f84891cad7396")
    _maven_import(artifact = "com.github.luben:zstd-jni:1.4.0-1", licenses = ["notice"], sha256 = "0d45847c7a1fc59c24ee71d942cc1faea6a78ce7a88bf65838358bda2a316567")

    _maven_import(artifact = "com.google.guava:guava:20.0", licenses = ["notice"], sha256 = "36a666e3b71ae7f0f0dca23654b67e086e6c93d192f60ba5dfd5519db6c288c8")
    _maven_import(artifact = "com.puppycrawl.tools:checkstyle:6.17", licenses = ["notice"], sha256 = "61a8b52d03a5b163d0983cdc4b03396a92ea7f8dc8c007dda30f4db673e9e60c")
    _maven_import(artifact = "org.apache.htrace:htrace-core4:4.2.0-incubating", licenses = ["notice"], sha256 = "fcab21b4ae0829e99142d77240fa2963a85f6ff1ca3fc0f386f8b4ff3cae3b82")
    _maven_import(artifact = "org.apache.hadoop:hadoop-mapreduce-client-core:3.2.1", licenses = ["notice"], sha256 = "987331ccf68f839b6162517c352eff13fc2792d02c008603513d26fe8a9d4be3")
    _maven_import(artifact = "com.fasterxml.jackson.core:jackson-databind:2.9.7", licenses = ["notice"], sha256 = "675376decfc070b039d2be773a97002f1ee1e1346d95bd99feee0d56683a92bf")
    _maven_import(artifact = "org.bouncycastle:bcprov-jdk15on:1.56", licenses = ["notice"], sha256 = "963e1ee14f808ffb99897d848ddcdb28fa91ddda867eb18d303e82728f878349")
    _maven_import(artifact = "org.bouncycastle:bcprov-jdk15on:1.56", licenses = ["notice"], sha256 = "963e1ee14f808ffb99897d848ddcdb28fa91ddda867eb18d303e82728f878349")
    _maven_import(artifact = "org.apache.hadoop:hadoop-common:3.2.1", licenses = ["notice"], sha256 = "48d55329372a38a4c5395b33e19c53d914c19e626d81e6a1148496d2d2532373")
    _maven_import(artifact = "org.apache.hadoop:hadoop-hdfs-client:3.2.1", licenses = ["notice"], sha256 = "284dc541ff2f62e0fc549a88e4d57ba393fc1e93fc4df665b0b041489497d130")
    _maven_import(artifact = "org.apache.mesos:mesos:1.5.0", licenses = ["notice"], sha256 = "66cb1222778c0fd665d99bbd3b57e05e769fd5fe2683374fecc82dab50cf9376")
    _maven_import(artifact = "org.apache.hadoop:hadoop-hdfs:3.2.1", licenses = ["notice"], sha256 = "c49ad967a7f5bb69f26f640a76c28a84fece127735fda2f081824ded76dc0aa6")
    _maven_import(artifact = "org.codehaus.woodstox:stax2-api:3.0.1", licenses = ["notice"], sha256 = "68ed11b72b138356063b8baf8551b9d67f46717b3c0001890949195c5b153199")
    _maven_import(artifact = "log4j:log4j:1.2.17", licenses = ["notice"], sha256 = "1d31696445697720527091754369082a6651bd49781b6005deb94e56753406f9")
    _maven_import(artifact = "it.unimi.dsi:fastutil:7.0.13", licenses = ["notice"], sha256 = "2ec909c77642b9c0220ab3e1c69bfcad3072789e2bcae5acdb5fb1df1ca14f04")
    _maven_import(artifact = "org.glassfish.jersey.core:jersey-client:2.27", licenses = ["notice"], sha256 = "aba407bda94df54f590041b4cde5f2fa31db45bd8b4cf7575af48c1f8f81bb04")
    _maven_import(artifact = "javax.ws.rs:javax.ws.rs-api:2.1.1", licenses = ["notice"], sha256 = "2c309eb2c9455ffee9da8518c70a3b6d46be2a269b2e2a101c806a537efe79a4")
    _maven_import(artifact = "org.glassfish.hk2.external:javax.inject:2.5.0-b62", licenses = ["notice"], sha256 = "2877904ff2888842b52df5d0748c511df3e1a63b7292e2e326ec6ae329e769e6")
    _maven_import(artifact = "org.glassfish.jersey.core:jersey-common:2.27", licenses = ["notice"], sha256 = "9a9578c6dac52b96195a614150f696d455db6b6d267a645c3120a4d0ee495789")
    _maven_import(artifact = "org.glassfish.hk2:hk2-locator:2.5.0-b63", licenses = ["notice"], sha256 = "4a162237e8cbe4ae96b8b8e230c364bd9ab6cb898513be9a6a0725e63097e975")
    _maven_import(artifact = "org.glassfish.hk2:hk2-api:2.5.0-b63", licenses = ["notice"], sha256 = "468fdb35e70b5babd72ba7eca9a8ba409665b63c1681048700206253757abf55")
    _maven_import(artifact = "org.javassist:javassist:3.24.0-GA", licenses = ["notice"], sha256 = "aba81efa678b621203fb89aeff81d6f126f7a9dd709401e5609c42976684ae23")
    _maven_import(artifact = "org.glassfish.hk2:osgi-resource-locator:2.5.0-b42", licenses = ["notice"], sha256 = "803552cd82a5741fa6273194ad27275a3a8ed18aa7d6be5a04ad6ad5972853c9")
    _maven_import(artifact = "org.glassfish.hk2:hk2-utils:2.5.0-b63", licenses = ["notice"], sha256 = "f5275b5b4956a61e1ca6ef3b7723315fc96b167f9991870d0264ef4fe1d2d603")
    _maven_import(artifact = "org.glassfish.hk2.external:aopalliance-repackaged:2.5.0-b63", licenses = ["notice"], sha256 = "70bbc43cbdcdb51310085ed7780f845251b7b23a009d08e571fe1cb9f59e7511")
    _maven_import(artifact = "org.glassfish.jersey.bundles.repackaged:jersey-guava:2.26-b03", licenses = ["notice"], sha256 = "95e6d574880062d255606e3ff1b8d41a548f85ff8cd41ca5e169acd595606da9")
    _maven_import(artifact = "org.glassfish.jersey.inject:jersey-hk2:2.26", licenses = ["notice"], sha256 = "4e9ab17a051eaacae0829a45cbdb5c603876006fdc86b5d9c5005be10266dcf7")
    _maven_import(artifact = "org.glassfish.jersey.media:jersey-media-json-jackson:2.27", licenses = ["notice"], sha256 = "815a783428d87e3f74591c6a9e4fd9c4bf37f5492e4c574b0a3e26a731dabc86")
    _maven_import(artifact = "org.glassfish.jersey.ext:jersey-entity-filtering:2.27", licenses = ["notice"], sha256 = "529b7ee7830441cffe98851b1e6edc0edd30e8b066052999daa5de63c56302b2")
    _maven_import(artifact = "com.fasterxml.jackson.jaxrs:jackson-jaxrs-base:2.9.7", licenses = ["notice"], sha256 = "aa392e17f53f8a4bfa30a7b871d4c8847dadf8a51bd2345c778c8c24ad1546bf")
    _maven_import(artifact = "com.fasterxml.jackson.jaxrs:jackson-jaxrs-json-provider:2.9.7", licenses = ["notice"], sha256 = "e2822cb192420154ad648b6e73ce91426b2b8a829698819505b1a6bfc02b08ea")
    _maven_import(artifact = "com.fasterxml.jackson.module:jackson-module-jaxb-annotations:2.9.7", licenses = ["notice"], sha256 = "027a1ddfc6e2372166b8825346476f8f62e5f6fac0546c5db1ee3bec0e8d104b")
    _maven_import(artifact = "com.fasterxml.jackson.core:jackson-annotations:2.9.7", licenses = ["notice"], sha256 = "8bf8c224e9205f77a0e239e96e473bdb263772db4ab85ecd1810e14c04132c5e")

    #dashboard
    _maven_import(artifact = "org.springframework.boot:spring-boot-starter:2.1.0.RELEASE", licenses = ["notice"], sha256 = "02450499d623deb3d5f5635258e615ed141642d7944d91fbd7c1614887ae8a29")
    _maven_import(artifact = "org.slf4j:jul-to-slf4j:1.7.25", licenses = ["notice"], sha256 = "416c5a0c145ad19526e108d44b6bf77b75412d47982cce6ce8d43abdbdbb0fac")
    _maven_import(artifact = "org.springframework.boot:spring-boot-starter-logging:2.1.0.RELEASE", licenses = ["notice"], sha256 = "ecff924a9132c9419fddb60b6e5037cb1e77728098dcb9908dd943715c588291")
    _maven_import(artifact = "org.springframework.plugin:spring-plugin-metadata:1.2.0.RELEASE", licenses = ["notice"], sha256 = "aa58a6e6d038553b6bfae03bd18cd985e4bfb37cb2fb6406551b87f57283b00a")
    _maven_import(artifact = "org.springframework.boot:spring-boot-starter-data-jpa:2.1.0.RELEASE", licenses = ["notice"], sha256 = "16131b28aea396cf6fa7d71ed24c631a9f2c7afe1ac7df274c4c732a930db27c")
    _maven_import(artifact = "org.springframework.boot:spring-boot-starter-jdbc:2.1.0.RELEASE", licenses = ["notice"], sha256 = "8ba8730b3cefbc3d78cab5927fbdae2c27d567998d3db4de78ea7da629ad5581")
    _maven_import(artifact = "org.springframework.boot:spring-boot-starter-aop:2.1.0.RELEASE", licenses = ["notice"], sha256 = "97a3368fd49084325c3dc53e1291398045ecdfd45c6454e1e4715131e7cab681")
    _maven_import(artifact = "org.springframework.boot:spring-boot-starter-test:2.1.0.RELEASE", licenses = ["notice"], sha256 = "37085b69844c57f1664673d7fe013a33a8d7db5532b75974cb43adc55876a8ec")
    _maven_import(artifact = "org.springframework.boot:spring-boot-starter-tomcat:2.1.0.RELEASE", licenses = ["notice"], sha256 = "3de89d57ef091683a020ee19ca2c0e30a5b67f6335d1aa4c20a06954ccfe0720")
    _maven_import(artifact = "org.springframework.boot:spring-boot-starter-web:2.1.0.RELEASE", licenses = ["notice"], sha256 = "3acba18295ead0e03168bf91a6a6d7916e37294b84a5f5dd9ef393e8e9c1bdb2")
    _maven_import(artifact = "com.fasterxml.jackson.module:jackson-module-parameter-names:2.9.7", licenses = ["notice"], sha256 = "3e2e224238d923b5396599a6ce753a74cebf69b17c0af1f0448a59f8a03c4bc2")
    _maven_import(artifact = "org.springframework.boot:spring-boot-starter-json:2.1.0.RELEASE", licenses = ["notice"], sha256 = "bbdf99a453a464f696a2ce16b2350850c1b36ea187495c6ba2f1c0a0757ade59")
    _maven_import(artifact = "javax.annotation:javax.annotation-api:1.3.2", licenses = ["notice"], sha256 = "e04ba5195bcd555dc95650f7cc614d151e4bcd52d29a10b8aa2197f3ab89ab9b")
    _maven_import(artifact = "io.swagger:swagger-annotations:1.5.20", licenses = ["notice"], sha256 = "69dee1ef78137a3ac5f9716193224049eab41b83fc6b845c2522efceb0af0273")
    _maven_import(artifact = "org.springframework:spring-jcl:5.1.2.RELEASE", licenses = ["notice"], sha256 = "857503050793c73c02ae77041e31d66936a4fd59beb97ee6c72fae2bab1993c2")
    _maven_import(artifact = "org.mapstruct:mapstruct:1.2.0.Final", licenses = ["notice"], sha256 = "a3d2414cb7adbd5ae9b29bff5197a42d6e48bdf68d9798d437f48e798abd2309")
    _maven_import(artifact = "org.springframework.plugin:spring-plugin-core:1.2.0.RELEASE", licenses = ["notice"], sha256 = "de8d411556cccbb9a68a4b40f847e473593336412de86fb3f6f7f61f3923c09e")
    _maven_import(artifact = "org.apache.logging.log4j:log4j-to-slf4j:2.11.1", licenses = ["notice"], sha256 = "ade27136788da38fe2b0f2b331c8f2e1a07c4e64dd45bf3d09efc49ecddfecc4")
    _maven_import(artifact = "javax.transaction:javax.transaction-api:1.3", licenses = ["notice"], sha256 = "603df5e4fc1eeae8f5e5d363a8be6c1fa47d0df1df8739a05cbcb9fafd6df2da")
    _maven_import(artifact = "org.slf4j:slf4j-api:1.7.25", licenses = ["notice"], sha256 = "18c4a0095d5c1da6b817592e767bb23d29dd2f560ad74df75ff3961dbde25b79")
    _maven_import(artifact = "net.minidev:accessors-smart:1.2", licenses = ["notice"], sha256 = "0c7c265d62fc007124dc32b91336e9c4272651d629bc5fa1a4e4e3bc758eb2e4")
    _maven_import(artifact = "com.vaadin.external.google:android-json:0.0.20131108.vaadin1", licenses = ["notice"], sha256 = "dfb7bae2f404cfe0b72b4d23944698cb716b7665171812a0a4d0f5926c0fac79")
    _maven_import(artifact = "com.fasterxml.jackson.datatype:jackson-datatype-jdk8:2.9.7", licenses = ["notice"], sha256 = "ec67a3d5e6abc7c7c611dd02ad270bbac0ca9a98b32c6cc821fb011a5863b99e")
    _maven_import(artifact = "org.apache.logging.log4j:log4j-slf4j-impl:2.11.1", licenses = ["notice"], sha256 = "21a53ca21dfdce610036da8428e6142d2c8c0c7af210d97a6ec5c97c55ce9ae5")
    _maven_import(artifact = "org.skyscreamer:jsonassert:1.5.0", licenses = ["notice"], sha256 = "a310bc79c3f4744e2b2e993702fcebaf3696fec0063643ffdc6b49a8fb03ef39")
    _maven_import(artifact = "io.springfox:springfox-swagger2:2.9.2", licenses = ["notice"], sha256 = "5341bf351c3e14e5a8436f81eeb2dc8f9f07ef83c8cd046b4e0edea33d0f8c52")
    _maven_import(artifact = "com.fasterxml.jackson.core:jackson-annotations:2.9.0", licenses = ["notice"], sha256 = "45d32ac61ef8a744b464c54c2b3414be571016dd46bfc2bec226761cf7ae457a")
    _maven_import(artifact = "com.fasterxml:classmate:1.4.0", licenses = ["notice"], sha256 = "2829acc59abf4aa6b72579697a0391c0fc69df7772ae59c58e0237f909cd6803")
    _maven_import(artifact = "io.springfox:springfox-spi:2.9.2", licenses = ["notice"], sha256 = "8e0d6a9ef7b75060f2fd1797759880d259b292c159043bd624d68f1b57734d79")
    _maven_import(artifact = "org.jboss.logging:jboss-logging:3.3.2.Final", licenses = ["notice"], sha256 = "cb914bfe888da7d9162e965ac8b0d6f28f2f32eca944a00fbbf6dd3cf1aacc13")
    _maven_import(artifact = "net.bytebuddy:byte-buddy-agent:1.9.3", licenses = ["notice"], sha256 = "547288e013a9d1f4a4ce2ab84c24e3edda6e433c7fa6b2c3c3613932671b05b1")
    _maven_import(artifact = "javax.activation:javax.activation-api:1.2.0", licenses = ["notice"], sha256 = "43fdef0b5b6ceb31b0424b208b930c74ab58fac2ceeb7b3f6fd3aeb8b5ca4393")
    _maven_import(artifact = "org.ow2.asm:asm:5.0.4", licenses = ["notice"], sha256 = "896618ed8ae62702521a78bc7be42b7c491a08e6920a15f89a3ecdec31e9a220")
    _maven_import(artifact = "org.springframework:spring-aspects:5.1.2.RELEASE", licenses = ["notice"], sha256 = "b8fa73bfd3af2ea083ae915415a066e102249f3b11e5327e2f0efe45ce33a506")
    _maven_import(artifact = "org.hamcrest:hamcrest-library:1.3", licenses = ["notice"], sha256 = "711d64522f9ec410983bd310934296da134be4254a125080a0416ec178dfad1c")
    _maven_import(artifact = "org.hamcrest:hamcrest-core:1.3", licenses = ["notice"], sha256 = "66fdef91e9739348df7a096aa384a5685f4e875584cce89386a7a47251c4d8e9")
    _maven_import(artifact = "org.objenesis:objenesis:2.6", licenses = ["notice"], sha256 = "5e168368fbc250af3c79aa5fef0c3467a2d64e5a7bd74005f25d8399aeb0708d")
    _maven_import(artifact = "org.hibernate.common:hibernate-commons-annotations:5.0.4.Final", licenses = ["notice"], sha256 = "b509d514d33265c0e8d872a3bf93df9da1c4d8760bdeec274b73c3310976c4f8")
    _maven_import(artifact = "javax.validation:validation-api:2.0.1.Final", licenses = ["notice"], sha256 = "9873b46df1833c9ee8f5bc1ff6853375115dadd8897bcb5a0dffb5848835ee6c")
    _maven_import(artifact = "io.springfox:springfox-core:2.9.2", licenses = ["notice"], sha256 = "70ed452095f0cf4d916d4f5120e79f9ea7ba609f4fdfb1f6e863227c20dd0a0b")
    _maven_import(artifact = "io.springfox:springfox-swagger-common:2.9.2", licenses = ["notice"], sha256 = "1d8534e2d38f989a84900166264cde966e9368ce0af74ce5ddda48ab6cd744fb")
    _maven_import(artifact = "io.springfox:springfox-schema:2.9.2", licenses = ["notice"], sha256 = "f289487967890dbb3698aaa9eaaac656c9bb9e30ee8cd399980ae8d8f888783f")
    _maven_import(artifact = "com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.9.7", licenses = ["notice"], sha256 = "231ca383e0f71d5e372ad0aa5165e1a9767a0ce28ec3e6f5992b2b394aa3abd4")
    _maven_import(artifact = "io.swagger:swagger-models:1.5.20", licenses = ["notice"], sha256 = "0adbb590fc665f17594f8bc7acce6871ed5602c8a50d0ad5419e3b72efaef639")
    _maven_import(artifact = "javax.xml.bind:jaxb-api:2.3.1", licenses = ["notice"], sha256 = "88b955a0df57880a26a74708bc34f74dcaf8ebf4e78843a28b50eae945732b06")
    _maven_import(artifact = "net.minidev:json-smart:2.3", licenses = ["notice"], sha256 = "903f48c8aa4c3f6426440b8d32de89fa1dc23b1169abde25e4e1d068aa67708b")
    _maven_import(artifact = "org.springframework.boot:spring-boot-test-autoconfigure:2.1.0.RELEASE", licenses = ["notice"], sha256 = "aef4c559d2fe57fe1134d7758762de0e765b208f01842910ebcdced40d8ee3ea")
    _maven_import(artifact = "javax.persistence:javax.persistence-api:2.2", licenses = ["notice"], sha256 = "5578b71b37999a5eaed3fea0d14aa61c60c6ec6328256f2b63472f336318baf4")
    _maven_import(artifact = "org.xmlunit:xmlunit-core:2.6.2", licenses = ["notice"], sha256 = "4f0e407dc9eb19582d74b9bcbeeef5117ccae42ebc4dd589db2da506a7c2e17d")
    _maven_import(artifact = "io.springfox:springfox-spring-web:2.9.2", licenses = ["notice"], sha256 = "df925e7a2435de246afd68b83800e1f2a4b6d8031692298740e261a4a9b30b3d")
    _maven_import(artifact = "org.jboss:jandex:2.0.5.Final", licenses = ["notice"], sha256 = "9112a9c33175b8c64b999ecf47b649fdf1cd6fa8262d0677895e976ed2891f0b")
    _maven_import(artifact = "org.springframework.boot:spring-boot-test:2.1.0.RELEASE", licenses = ["notice"], sha256 = "c9d4987af795410be39262ee831c6b3398eaab528e8eb8d00b67e857ab1d1f22")
    _maven_import(artifact = "org.springframework:spring-orm:5.1.2.RELEASE", licenses = ["notice"], sha256 = "c03ec6a49d798de82869b1da78f05b90745ba02a17b3afd024d58a1226b7d41c")
    _maven_import(artifact = "ch.qos.logback:logback-classic:1.2.3", licenses = ["notice"], sha256 = "fb53f8539e7fcb8f093a56e138112056ec1dc809ebb020b59d8a36a5ebac37e0")
    _maven_import(artifact = "com.jayway.jsonpath:json-path:2.4.0", licenses = ["notice"], sha256 = "60441c74fb64e5a480070f86a604941927aaf684e2b513d780fb7a38fb4c5639")
    _maven_import(artifact = "org.springframework:spring-tx:5.1.2.RELEASE", licenses = ["notice"], sha256 = "e9feb01ae0c6b1fbb7f9f5a60631f5913a47a96e5febdc6ceb9703a287158fbe")
    _maven_import(artifact = "org.apache.tomcat.embed:tomcat-embed-el:9.0.12", licenses = ["notice"], sha256 = "1f613885f5475db3a02fe20e167337c5fed858228773984fe96b8b99beccca03")
    _maven_import(artifact = "org.apache.tomcat.embed:tomcat-embed-websocket:9.0.12", licenses = ["notice"], sha256 = "57e0df646bd7e56f28e6d8f8c559552e3a1c62beca6d4d8468deda6633ac5744")
    _maven_import(artifact = "org.yaml:snakeyaml:1.23", licenses = ["notice"], sha256 = "13009fb5ede3cf2be5a8d0f1602155aeaa0ce5ef5f9366892bd258d8d3d4d2b1")
    _maven_import(artifact = "org.apache.logging.log4j:log4j-api:2.11.1", licenses = ["notice"], sha256 = "493b37b5a6c49c4f5fb609b966375e4dc1783df436587584ca1dc7e861d0742b")
    _maven_import(artifact = "org.springframework:spring-expression:5.1.2.RELEASE", licenses = ["notice"], sha256 = "24c958b6172da35e6755d62769fa3a8b4ddc430cda31da976f008a7a226f15de")
    _maven_import(artifact = "org.dom4j:dom4j:2.1.1", licenses = ["notice"], sha256 = "a2ef5fb4990b914a31176c51f6137f6f04253dd165420985051f9fd4fb032128")
    _maven_import(artifact = "com.fasterxml.jackson.core:jackson-core:2.9.7", licenses = ["notice"], sha256 = "9e5bc0efabd9f0cac5c1fdd9ae35b16332ed22a0ee19a356de370a18a8cb6c84")
    _maven_import(artifact = "junit:junit:4.12", licenses = ["notice"], sha256 = "59721f0805e223d84b90677887d9ff567dc534d7c502ca903c0c2b17f05c116a")
    _maven_import(artifact = "org.springframework:spring-aop:5.1.2.RELEASE", licenses = ["notice"], sha256 = "46ab5d470272b8c3b06714ca27709f9fabdc65ada5ffa4b0744042dbc85c4466")
    _maven_import(artifact = "org.springframework.data:spring-data-jpa:2.1.2.RELEASE", licenses = ["notice"], sha256 = "cd41c13ae834704e6008f1f682d3320093ef659a7713b3d65c89bcbf9d30a155")
    _maven_import(artifact = "org.springframework:spring-jdbc:5.1.2.RELEASE", licenses = ["notice"], sha256 = "fd0ac6f59f3bcbc21410778f1902dfc61348f32dbdc7a48b0349fea4cf99cf97")
    _maven_import(artifact = "ch.qos.logback:logback-core:1.2.3", licenses = ["notice"], sha256 = "5946d837fe6f960c02a53eda7a6926ecc3c758bbdd69aa453ee429f858217f22")
    _maven_import(artifact = "antlr:antlr:2.7.7", licenses = ["notice"], sha256 = "88fbda4b912596b9f56e8e12e580cc954bacfb51776ecfddd3e18fc1cf56dc4c")
    _maven_import(artifact = "org.mockito:mockito-core:2.23.0", licenses = ["notice"], sha256 = "637991bfc37fdd2a7adfe610f2ee5290acf9b15a7e4347bc3c96c61ed9dfe043")
    _maven_import(artifact = "org.springframework:spring-test:5.1.2.RELEASE", licenses = ["notice"], sha256 = "f1b6660b13a60c268d91a59a8b81ded3ef9febd72ace6db1660298654def0a55")
    _maven_import(artifact = "org.springframework:spring-beans:5.1.2.RELEASE", licenses = ["notice"], sha256 = "131737816184d8e772a6e3bbf46025de432a0cc7a2646fef8d63450ec9444033")
    _maven_import(artifact = "org.springframework:spring-webmvc:5.1.2.RELEASE", licenses = ["notice"], sha256 = "8da6ebe54db7ebe65e35ad455a56c462afc808664157736d449a384614e08d46")
    _maven_import(artifact = "org.springframework.boot:spring-boot:2.1.0.RELEASE", licenses = ["notice"], sha256 = "ac4666d28d2b5081de70a33fd6d0085bbc08c2801e7405188ed93f804ab85035")
    _maven_import(artifact = "org.springframework:spring-context:5.1.2.RELEASE", licenses = ["notice"], sha256 = "d280d2622cd01dad237a1d4de335b6869167a737e2b29df3d02552da3ecfeafb")
    _maven_import(artifact = "org.springframework.data:spring-data-commons:2.1.2.RELEASE", licenses = ["notice"], sha256 = "a11eadd0c5fa5e5f4f9e28bff97e6bf2912a8fc5cae5a4cfac800b9c43d4a397")
    _maven_import(artifact = "org.hibernate.validator:hibernate-validator:6.0.13.Final", licenses = ["notice"], sha256 = "62e11d55188d97ea7e044fc6bf24da261e5c6b13ab971b758f8578afbb3de965")
    _maven_import(artifact = "org.springframework.boot:spring-boot-autoconfigure:2.1.0.RELEASE", licenses = ["notice"], sha256 = "45d66895db8fabc34b58170717a5f3abde33a7894b887d7b42a29118405a9b50")
    _maven_import(artifact = "org.springframework:spring-core:5.1.2.RELEASE", licenses = ["notice"], sha256 = "3f646f7a51bd3a32c89241b899f6cc73dc40ea8275cd3233f4699668bfb839c5")
    _maven_import(artifact = "com.fasterxml.jackson.core:jackson-databind:2.9.7", licenses = ["notice"], sha256 = "675376decfc070b039d2be773a97002f1ee1e1346d95bd99feee0d56683a92bf")
    _maven_import(artifact = "org.springframework:spring-web:5.1.2.RELEASE", licenses = ["notice"], sha256 = "fc135b74d9aa285c4aa8d9b19e2367817a7cd2d2af53f726b3d1466de428f4b0")
    _maven_import(artifact = "org.apache.logging.log4j:log4j-core:2.11.1", licenses = ["notice"], sha256 = "a20c34cdac4978b76efcc9d0db66e95600bd807c6a0bd3f5793bcb45d07162ec")
    _maven_import(artifact = "com.h2database:h2:1.4.197", licenses = ["notice"], sha256 = "37f5216e14af2772930dff9b8734353f0a80e89ba3f33e065441de6537c5e842")
    _maven_import(artifact = "org.aspectj:aspectjweaver:1.9.2", licenses = ["notice"], sha256 = "b98ad94989052b195150edf1f85db2ee10f33e140d416f19f03c9746da16b691")
    _maven_import(artifact = "org.assertj:assertj-core:3.11.1", licenses = ["notice"], sha256 = "2ee2bd3e81fc818d423d442b658f28acf938d9078d6ba016a64b362fdd7779e8")
    _maven_import(artifact = "com.google.guava:guava:20.0", licenses = ["notice"], sha256 = "36a666e3b71ae7f0f0dca23654b67e086e6c93d192f60ba5dfd5519db6c288c8")
    _maven_import(artifact = "io.springfox:springfox-swagger-ui:2.9.2", licenses = ["notice"], sha256 = "44ee72b046428a694c44095c60f8156bcc505faff2d5b142b0f8175a6570b307")
    _maven_import(artifact = "org.apache.tomcat.embed:tomcat-embed-core:9.0.12", licenses = ["notice"], sha256 = "28e42ed45029912444cb995fffe28b3d387a1f04ecbf03d27b51af7a57aa1acf")
    _maven_import(artifact = "net.bytebuddy:byte-buddy:1.9.3", licenses = ["notice"], sha256 = "a27350be602caea67a33d31281496c84c69b5ab34ddc228e9ff2253fc8f9cd31")
    _maven_import(artifact = "org.hibernate:hibernate-core:5.3.7.Final", licenses = ["notice"], sha256 = "862822a3ebf43aa38ff7d36346bb4cef1fc5a5c400b0a8f35d4a33df816202e9")
    _maven_import(artifact = "org.springframework.boot:spring-boot-loader:2.1.0.RELEASE", licenses = ["notice"], sha256 = "776f84cf9c67e3c5c719e92bd9369c9b10b5e35f3814d8eb3de1317539c96eb1")
    _maven_import(artifact = "it.unimi.dsi:fastutil:8.2.2", licenses = ["notice"], sha256 = "a6492bd60e4a93c3e302c00291497696bf6d2f927eacaff11cb4fa336dfd5097")
    _maven_import(artifact = "org.slf4j:slf4j-api:1.7.25", licenses = ["notice"], sha256 = "18c4a0095d5c1da6b817592e767bb23d29dd2f560ad74df75ff3961dbde25b79")
    _maven_import(artifact = "com.thoughtworks.paranamer:paranamer:2.7", licenses = ["notice"], sha256 = "63e3f53f8f70784b65c25b2ee475813979d6d0e7f7b2510b364c4e1f4a803ccc")
    _maven_import(artifact = "com.fasterxml.jackson.core:jackson-annotations:2.9.9", licenses = ["notice"], sha256 = "1100a5884ddc4439a77165e1b9668c6063c07447cd2f6c9f69e3688ee76080c1")
    _maven_import(artifact = "org.codehaus.jackson:jackson-core-asl:1.9.13", licenses = ["notice"], sha256 = "440a9cb5ca95b215f953d3a20a6b1a10da1f09b529a9ddea5f8a4905ddab4f5a")
    _maven_import(artifact = "com.fasterxml.jackson.core:jackson-core:2.9.9", licenses = ["notice"], sha256 = "3083079be6088db2ed0a0c6ff92204e0aa48fa1de9db5b59c468f35acf882c2c")
    _maven_import(artifact = "org.apache.commons:commons-compress:1.8.1", licenses = ["notice"], sha256 = "5fca136503f86ecc6cb61fbd17b137d59e56b45c7a5494e6b8fd3cabd4697fbd")
    _maven_import(artifact = "org.apache.beam:beam-model-job-management:2.14.0", licenses = ["notice"], sha256 = "abe89830f2319a3dc15c5334ddaadb2074845cb2d00381985feec7e1d14017b6")
    _maven_import(artifact = "com.fasterxml.jackson.core:jackson-databind:2.9.9", licenses = ["notice"], sha256 = "5cbbf429d9e32e3881f0a1438a1f666912219327e9e68b5dcaef6d8e5c5f6b28")
    _maven_import(artifact = "org.codehaus.jackson:jackson-mapper-asl:1.9.13", licenses = ["notice"], sha256 = "74e7a07a76f2edbade29312a5a2ebccfa019128bc021ece3856d76197e9be0c2")
    _maven_import(artifact = "org.xerial.snappy:snappy-java:1.1.7.3", licenses = ["notice"], sha256 = "7eea31c0a25d35cd092d8aec08bed04f22152409b58d63d43839074a9ab7ab97")

    #    _maven_import(artifact = "org.xerial.snappy:snappy-java:1.1.4", licenses = ["notice"], sha256 = "f75ec0fa9c843e236c6e1512c17c095cfffd175f32e21ea0e3eccb540d77f002")
    _maven_import(artifact = "org.apache.avro:avro:1.8.2", licenses = ["notice"], sha256 = "f754a0830ce67a5a9fa67a54ec15d103ef15e1c850d7b26faf7b647eeddc82d3")
    _maven_import(artifact = "org.apache.beam:beam-model-pipeline:2.14.0", licenses = ["notice"], sha256 = "683a66f3ad5deab6be513e2d16870e1569d2a9a75dba833536358b278b9612cb")
    _maven_import(artifact = "org.apache.beam:beam-sdks-java-core:2.14.0", licenses = ["notice"], sha256 = "45abd6f498080cbf1df73d386ae20aca35d7284605504214d3cbf03d9f459cf8")
    _maven_import(artifact = "org.apache.beam:beam-sdks-java-fn-execution:2.14.0", licenses = ["notice"], sha256 = "03cc6924b1f0f98faa7424fae4cddaa75d4b8f7111b31ddbe8586113f41d1061")
    _maven_import(artifact = "org.apache.beam:beam-runners-core-java:2.14.0", licenses = ["notice"], sha256 = "904a66fbf35b7919b9e40a75e2f013a5e4be1876c157a5768f89086e271e4276")
    _maven_import(artifact = "org.apache.beam:beam-runners-core-construction-java:2.14.0", licenses = ["notice"], sha256 = "c5c2db509a9d45e4ce48661be85c4ef216330ed5697efbc2af623f0f6958661e")
    _maven_import(artifact = "org.apache.beam:beam-model-fn-execution:2.14.0", licenses = ["notice"], sha256 = "c134f05d0947484826849663261d5847c94bc9c1b6e9b718019991f92df11bff")
    _maven_import(artifact = "args4j:args4j:2.33", licenses = ["notice"], sha256 = "91ddeaba0b24adce72291c618c00bbdce1c884755f6c4dba9c5c46e871c69ed6")
    _maven_import(artifact = "org.apache.beam:beam-runners-java-fn-execution:2.14.0", licenses = ["notice"], sha256 = "54c33c8dc09c9a145664563a899af06bb2b66d71a7f30e72444673dde476cb20")
    _maven_import(artifact = "org.apache.beam:beam-vendor-sdks-java-extensions-protobuf:2.14.0", licenses = ["notice"], sha256 = "7d7f2e082fa22d3fb8cff2ff5923a10a548ca04a6aaa9e9315db26ee2e6632b2")
    _maven_import(artifact = "joda-time:joda-time:2.10.1", licenses = ["notice"], sha256 = "d269671656767e05a58dd634cbafc36ed70d417220b058d11c0d88dfd281616d")
    _maven_import(artifact = "org.apache.beam:beam-model-pipeline:2.14.0", licenses = ["notice"], sha256 = "683a66f3ad5deab6be513e2d16870e1569d2a9a75dba833536358b278b9612cb")
    _maven_import(artifact = "org.apache.beam:beam-vendor-guava-20_0:0.1", licenses = ["notice"], sha256 = "ea30aaecc425d9630ae8b0f285add31bbbc1600acf8be2a582ed3a7c1891b3f6")
    _maven_import(artifact = "javax.xml.bind:jaxb-api:2.3.1", licenses = ["notice"], sha256 = "88b955a0df57880a26a74708bc34f74dcaf8ebf4e78843a28b50eae945732b06")
    _maven_import(artifact = "org.apache.beam:beam-vendor-grpc-1_13_1:0.2", licenses = ["notice"], sha256 = "3cdb4a043692be8a51e58ca5a6de55073c55a6500557852a3ad0b5d0fee33f49")

    # For Python Support
    _maven_import(artifact = "net.sf.py4j:py4j:0.10.8.1", licenses = ["notice"], sha256 = "4c484e75a3d8695ccbb7d4327298c48fc9bb8fe979bb90fa092d1b67459f3835")
    _maven_import(artifact = "black.ninia:jep:3.9.0", licenses = ["notice"], sha256 = "de8a69bc028d131f23a41bb964c0707a51dc8defcf7418bb891d7b4ab168b16d")

    #for apache commons collections 4.4
    _maven_import(artifact = "org.hamcrest:hamcrest-core:1.3", licenses = ["notice"], sha256 = "66fdef91e9739348df7a096aa384a5685f4e875584cce89386a7a47251c4d8e9")
    _maven_import(artifact = "org.objenesis:objenesis:3.0.1", licenses = ["notice"], sha256 = "7a8ff780b9ff48415d7c705f60030b0acaa616e7f823c98eede3b63508d4e984")
    _maven_import(artifact = "junit:junit:4.12", licenses = ["notice"], sha256 = "59721f0805e223d84b90677887d9ff567dc534d7c502ca903c0c2b17f05c116a")
    _maven_import(artifact = "org.apache.commons:commons-lang3:3.9", licenses = ["notice"], sha256 = "de2e1dcdcf3ef917a8ce858661a06726a9a944f28e33ad7f9e08bea44dc3c230")
    _maven_import(artifact = "org.easymock:easymock:4.0.2", licenses = ["notice"], sha256 = "104370107ef64d115e642cbdc14cea438a8d076f5aee9a9cb7882dc9a8ed4123")
    _maven_import(artifact = "org.apache.commons:commons-collections4:4.4", licenses = ["notice"], sha256 = "1df8b9430b5c8ed143d7815e403e33ef5371b2400aadbe9bda0883762e0846d1")
    _maven_import(artifact = "org.apache.commons:commons-configuration2:2.6", licenses = ["notice"], sha256 = "225788911e53af0b29a31a18e0d03b05d86aa9c9e0b3c6686982c30c10f931fb")


    #for DL
    _maven_import(artifact = "org.apache.maven:maven-plugin-api:2.0",licenses = ["notice"],sha256 = "5b62626069d85bb463314572734988d47bc98aab9f0ed48d2f1f9554960f5a35",)
    _maven_import(artifact = "org.qunix:structure-maven-plugin:0.0.2",licenses = ["notice"],sha256 = "5f07ad70ba250c2fd49937ea6cbc1cbb51d8facd43872fbf65dff47e6d8858fc",)
    _maven_import(artifact = "org.apache.maven:maven-profile:2.0.8",licenses = ["notice"],sha256 = "bd566c2fdb896e3dd157dc3e49b3c10f93250daaa6462af3cd42ad5b4aeda0a3",)
    _maven_import(artifact = "org.hamcrest:hamcrest-core:1.3",licenses = ["notice"],sha256 = "66fdef91e9739348df7a096aa384a5685f4e875584cce89386a7a47251c4d8e9",)
    _maven_import(artifact = "org.apache.maven:maven-plugin-registry:2.0.8",licenses = ["notice"],sha256 = "a7e77626ec4c8382a4c11ba808684402582693df00f0c39f7c0fc02cd1ece9ab",)
    _maven_import(artifact = "org.apache.maven:maven-repository-metadata:2.0.8",licenses = ["notice"],sha256 = "aff8473e802e4e1c226a777a198f72fbdf7ef36f6f972df6de763b767f652ee1",)
    _maven_import(artifact = "org.apache.maven:maven-settings:2.0.8",licenses = ["notice"],sha256 = "e1873a36ea2debc0bb6210c3064faae91cf12108e9f8b9845e283022f58ecb46",)
    _maven_import(artifact = "classworlds:classworlds:1.1",licenses = ["notice"],sha256 = "4e3e0ad158ec60917e0de544c550f31cd65d5a97c3af1c1968bf427e4a9df2e4",)
    _maven_import(artifact = "org.apache.maven.wagon:wagon-provider-api:1.0",licenses = ["notice"],sha256 = "b28dd1302ac34433d8d1b45fb254e093cd7b47277441af2018c8a3a4d8c1a60d",)
    _maven_import(artifact = "org.apache.maven:maven-artifact-manager:2.0.8",licenses = ["notice"],sha256 = "c257564b252dc69ff3f3603971164fcb387adbd000818e72c959e6be7b6319cd",)
    _maven_import(artifact = "org.apache.maven:maven-artifact:2.0.8",licenses = ["notice"],sha256 = "5cf23417cdee6a8e1eb6b9c015c8feea62cedb7dceb3ad098e6869569fcfd1c0",)
    _maven_import(artifact = "org.apache.maven:maven-model:2.0.8",licenses = ["notice"],sha256 = "51cde4f45d74720eaf567444d88c9ffdfc6896dd58dcc459403f6613d8439255",)
    _maven_import(artifact = "org.apache.maven:maven-project:2.0.8",licenses = ["notice"],sha256 = "46799ed8812c96f1e651958cbd864155cc8a9ae8d4ebe2392b845d232a910d31",)
    _maven_import(artifact = "junit:junit:4.11",licenses = ["notice"],sha256 = "90a8e1603eeca48e7e879f3afbc9560715322985f39a274f6f6070b43f9d06fe",)
    _maven_import(artifact = "org.codehaus.plexus:plexus-utils:1.4.6",licenses = ["notice"],sha256 = "b46fe183dc28526d083287817e6374ad460cef2d780f6096d3ea46caa0a25106",)
    _maven_import(artifact = "org.apache.commons:commons-lang3:3.3.2",licenses = ["notice"],sha256 = "6b81d10754dadf184d386011486e6509c2cc0c3d33565ced4fb4402b9413d47d",)
    _maven_import(artifact = "commons-collections:commons-collections:3.2.1",licenses = ["notice"],sha256 = "87363a4c94eaabeefd8b930cb059f66b64c9f7d632862f23de3012da7660047b",)
    _maven_import(artifact = "com.intel.analytics.bigdl.core.dist:all:0.11.1",licenses = ["notice"],sha256 = "6d5ff2212f362d289cbb3db0b401fe5ad5005a63aafce251252130ce0d0c6a7a",)
