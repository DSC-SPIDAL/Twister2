//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.examples.comms;

import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.iu.dsc.tws.api.JobConfig;
import edu.iu.dsc.tws.api.Twister2Job;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.examples.Utils;
import edu.iu.dsc.tws.examples.comms.batch.BAllGatherExample;
import edu.iu.dsc.tws.examples.comms.batch.BAllReduceExample;
import edu.iu.dsc.tws.examples.comms.batch.BBroadcastExample;
import edu.iu.dsc.tws.examples.comms.batch.BDJoinExample;
import edu.iu.dsc.tws.examples.comms.batch.BDKeyedGatherExample;
import edu.iu.dsc.tws.examples.comms.batch.BDirectExample;
import edu.iu.dsc.tws.examples.comms.batch.BGatherExample;
import edu.iu.dsc.tws.examples.comms.batch.BJoinExample;
import edu.iu.dsc.tws.examples.comms.batch.BJoinStudentExample;
import edu.iu.dsc.tws.examples.comms.batch.BKeyedGatherExample;
import edu.iu.dsc.tws.examples.comms.batch.BKeyedPartitionExample;
import edu.iu.dsc.tws.examples.comms.batch.BKeyedReduceExample;
import edu.iu.dsc.tws.examples.comms.batch.BPartitionExample;
import edu.iu.dsc.tws.examples.comms.batch.BReduceExample;
import edu.iu.dsc.tws.examples.comms.batch.BTAllToAll;
import edu.iu.dsc.tws.examples.comms.stream.SAllGatherExample;
import edu.iu.dsc.tws.examples.comms.stream.SAllReduceExample;
import edu.iu.dsc.tws.examples.comms.stream.SBroadcastExample;
import edu.iu.dsc.tws.examples.comms.stream.SDirectExample;
import edu.iu.dsc.tws.examples.comms.stream.SGatherExample;
import edu.iu.dsc.tws.examples.comms.stream.SKeyedGatherExample;
import edu.iu.dsc.tws.examples.comms.stream.SKeyedPartitionExample;
import edu.iu.dsc.tws.examples.comms.stream.SKeyedReduceExample;
import edu.iu.dsc.tws.examples.comms.stream.SPartitionExample;
import edu.iu.dsc.tws.examples.comms.stream.SReduceExample;
import edu.iu.dsc.tws.examples.utils.bench.BenchmarkMetadata;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.rsched.job.Twister2Submitter;

public class ExampleMain {
  private static final Logger LOG = Logger.getLogger(ExampleMain.class.getName());

  public static void main(String[] args) throws ParseException {
    // first load the configurations from command line and config files
    Config config = ResourceAllocator.loadConfig(new HashMap<>());

    Options options = new Options();
    options.addOption(Constants.ARGS_WORKERS, true, "Workers");
    options.addOption(Constants.ARGS_SIZE, true, "Size");
    options.addOption(Constants.ARGS_ITR, true, "Iteration");
    options.addOption(Constants.ARGS_WARMPU_ITR, true, "Warmup Iterations");
    options.addOption(Utils.createOption(Constants.ARGS_OPERATION, true, "Operation", true));
    options.addOption(Constants.ARGS_STREAM, false, "Stream");
    options.addOption(Utils.createOption(Constants.ARGS_TASK_STAGES, true, "Throughput mode", true));
    options.addOption(Utils.createOption(Constants.ARGS_GAP, true, "Gap", false));
    options.addOption(Utils.createOption(Constants.ARGS_FNAME, true, "File name", false));
    options.addOption(Utils.createOption(Constants.ARGS_OUTSTANDING, true, "Throughput no of messages", false));
    options.addOption(Utils.createOption(Constants.ARGS_THREADS, true, "Threads", false));
    options.addOption(Utils.createOption(Constants.ARGS_PRINT_INTERVAL, true, "Threads", false));
    options.addOption(Utils.createOption(Constants.ARGS_DATA_TYPE, true, "Data", false));
    options.addOption(Utils.createOption(Constants.ARGS_INIT_ITERATIONS, true, "Data", false));
    options.addOption(Constants.ARGS_VERIFY, false, "verify");
    options.addOption(Utils.createOption(BenchmarkMetadata.ARG_BENCHMARK_METADATA, true, "Benchmark Metadata", false));
    options.addOption(Utils.createOption(Constants.ARGS_WINDOW, false, "Weather windowing is used", false));
    options.addOption(Utils.createOption(Constants.ARG_RESOURCE_MEMORY, true, "Instance memory", false));

    CommandLineParser commandLineParser = new DefaultParser();
    CommandLine cmd = commandLineParser.parse(options, args);
    int workers = Integer.parseInt(cmd.getOptionValue(Constants.ARGS_WORKERS));

    String operation = cmd.getOptionValue(Constants.ARGS_OPERATION);
    boolean stream = cmd.hasOption(Constants.ARGS_STREAM);
    boolean verify = cmd.hasOption(Constants.ARGS_VERIFY);
    int size = 1;
    int itr = 1;
    int warmUpItr = 0;
    if (cmd.hasOption(Constants.ARGS_SIZE)) {
      size = Integer.parseInt(cmd.getOptionValue(Constants.ARGS_SIZE));
    }
    if (cmd.hasOption(Constants.ARGS_ITR)) {
      itr = Integer.parseInt(cmd.getOptionValue(Constants.ARGS_ITR));
    }
    if (cmd.hasOption(Constants.ARGS_WARMPU_ITR)) {
      warmUpItr = Integer.valueOf(cmd.getOptionValue(Constants.ARGS_WARMPU_ITR));
    }
    String threads = "true";
    if (cmd.hasOption(Constants.ARGS_THREADS)) {
      threads = cmd.getOptionValue(Constants.ARGS_THREADS);
    }

    String taskStages = cmd.getOptionValue(Constants.ARGS_TASK_STAGES);
    String gap = "0";
    if (cmd.hasOption(Constants.ARGS_GAP)) {
      gap = cmd.getOptionValue(Constants.ARGS_GAP);
    }

    String fName = "";
    if (cmd.hasOption(Constants.ARGS_FNAME)) {
      fName = cmd.getOptionValue(Constants.ARGS_FNAME);
    }

    String outstanding = "0";
    if (cmd.hasOption(Constants.ARGS_OUTSTANDING)) {
      outstanding = cmd.getOptionValue(Constants.ARGS_OUTSTANDING);
    }

    String printInt = "1";
    if (cmd.hasOption(Constants.ARGS_PRINT_INTERVAL)) {
      printInt = cmd.getOptionValue(Constants.ARGS_PRINT_INTERVAL);
    }

    String dataType = "default";
    if (cmd.hasOption(Constants.ARGS_DATA_TYPE)) {
      dataType = cmd.getOptionValue(Constants.ARGS_DATA_TYPE);
    }
    String intItr = "0";
    if (cmd.hasOption(Constants.ARGS_INIT_ITERATIONS)) {
      intItr = cmd.getOptionValue(Constants.ARGS_INIT_ITERATIONS);
    }

    boolean runBenchmark = cmd.hasOption(BenchmarkMetadata.ARG_BENCHMARK_METADATA);
    String benchmarkMetadata = null;
    if (runBenchmark) {
      benchmarkMetadata = cmd.getOptionValue(BenchmarkMetadata.ARG_BENCHMARK_METADATA);
    }

    int memory = 1024;
    if (cmd.hasOption(Constants.ARG_RESOURCE_MEMORY)) {
      memory = Integer.parseInt(cmd.getOptionValue(Constants.ARG_RESOURCE_MEMORY));
    }

    boolean window = cmd.hasOption(Constants.ARGS_WINDOW);

    // build JobConfig
    JobConfig jobConfig = new JobConfig();
    jobConfig.put(Constants.ARGS_ITR, Integer.toString(itr));
    jobConfig.put(Constants.ARGS_WARMPU_ITR, Integer.toString(warmUpItr));
    jobConfig.put(Constants.ARGS_OPERATION, operation);
    jobConfig.put(Constants.ARGS_SIZE, Integer.toString(size));
    jobConfig.put(Constants.ARGS_WORKERS, Integer.toString(workers));
    jobConfig.put(Constants.ARGS_TASK_STAGES, taskStages);
    jobConfig.put(Constants.ARGS_GAP, gap);
    jobConfig.put(Constants.ARGS_FNAME, fName);
    jobConfig.put(Constants.ARGS_OUTSTANDING, outstanding);
    jobConfig.put(Constants.ARGS_THREADS, threads);
    jobConfig.put(Constants.ARGS_PRINT_INTERVAL, printInt);
    jobConfig.put(Constants.ARGS_DATA_TYPE, dataType);
    jobConfig.put(Constants.ARGS_INIT_ITERATIONS, intItr);
    jobConfig.put(Constants.ARGS_VERIFY, verify);
    jobConfig.put(Constants.ARGS_STREAM, stream);
    jobConfig.put(Constants.ARG_RESOURCE_MEMORY, memory);
    jobConfig.put(BenchmarkMetadata.ARG_RUN_BENCHMARK, runBenchmark);
    if (runBenchmark) {
      jobConfig.put(BenchmarkMetadata.ARG_BENCHMARK_METADATA, benchmarkMetadata);
    }
    jobConfig.put(Constants.ARGS_WINDOW, window);

    // build the job
    if (!stream) {
      switch (operation) {
        case "reduce":
          submitJob(config, workers, jobConfig, BReduceExample.class.getName(), memory);
          break;
        case "allreduce":
          submitJob(config, workers, jobConfig, BAllReduceExample.class.getName(), memory);
          break;
        case "keyedreduce":
          submitJob(config, workers, jobConfig, BKeyedReduceExample.class.getName(), memory);
          break;
        case "partition":
          submitJob(config, workers, jobConfig, BPartitionExample.class.getName(), memory);
          break;
        case "keyedpartition":
          submitJob(config, workers, jobConfig, BKeyedPartitionExample.class.getName(), memory);
          break;
        case "gather":
          submitJob(config, workers, jobConfig, BGatherExample.class.getName(), memory);
          break;
        case "allgather":
          submitJob(config, workers, jobConfig, BAllGatherExample.class.getName(), memory);
          break;
        case "keyedgather":
          submitJob(config, workers, jobConfig, BKeyedGatherExample.class.getName(), memory);
          break;
        case "dkeyedgather":
          submitJob(config, workers, jobConfig, BDKeyedGatherExample.class.getName(), memory);
          break;
        case "join":
          submitJob(config, workers, jobConfig, BJoinExample.class.getName(), memory);
          break;
        case "joinstudent":
          submitJob(config, workers, jobConfig, BJoinStudentExample.class.getName(), memory);
          break;
        case "djoin":
          submitJob(config, workers, jobConfig, BDJoinExample.class.getName(), memory);
          break;
        case "direct":
          submitJob(config, workers, jobConfig, BDirectExample.class.getName(), memory);
          break;
        case "bcast":
          submitJob(config, workers, jobConfig, BBroadcastExample.class.getName(), memory);
          break;
        case "alltoall":
          submitJob(config, workers, jobConfig, BTAllToAll.class.getName(), memory);
      }
    } else {
      switch (operation) {
        case "reduce":
          submitJob(config, workers, jobConfig, SReduceExample.class.getName(), memory);
          break;
        case "keyedreduce":
          submitJob(config, workers, jobConfig, SKeyedReduceExample.class.getName(), memory);
          break;
        case "bcast":
          submitJob(config, workers, jobConfig, SBroadcastExample.class.getName(), memory);
          break;
        case "partition":
          submitJob(config, workers, jobConfig, SPartitionExample.class.getName(), memory);
          break;
        case "keyedpartition":
          submitJob(config, workers, jobConfig, SKeyedPartitionExample.class.getName(), memory);
          break;
        case "gather":
          submitJob(config, workers, jobConfig, SGatherExample.class.getName(), memory);
          break;
        case "keyedgather":
          submitJob(config, workers, jobConfig, SKeyedGatherExample.class.getName(), memory);
          break;
        case "allreduce":
          submitJob(config, workers, jobConfig, SAllReduceExample.class.getName(), memory);
          break;
        case "allgather":
          submitJob(config, workers, jobConfig, SAllGatherExample.class.getName(), memory);
          break;
        case "direct":
          submitJob(config, workers, jobConfig, SDirectExample.class.getName(), memory);
          break;
        default:
          LOG.log(Level.SEVERE, "Un-supported operation: " + operation);
      }
    }
  }

  private static void submitJob(Config config, int containers,
                                JobConfig jobConfig, String clazz, int memory) {
    Twister2Job twister2Job;
    twister2Job = Twister2Job.newBuilder()
        .setJobName(clazz)
        .setWorkerClass(clazz)
        .addComputeResource(1, memory, containers)
        .setConfig(jobConfig)
        .build();
    // now submit the job
    Twister2Submitter.submitJob(twister2Job, config);
  }
}
