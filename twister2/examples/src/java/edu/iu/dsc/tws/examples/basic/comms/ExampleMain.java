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
package edu.iu.dsc.tws.examples.basic.comms;

import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.iu.dsc.tws.api.JobConfig;
import edu.iu.dsc.tws.api.Twister2Submitter;
import edu.iu.dsc.tws.api.job.Twister2Job;
import edu.iu.dsc.tws.common.config.Config;
import edu.iu.dsc.tws.common.resource.WorkerComputeResource;
import edu.iu.dsc.tws.examples.Utils;
import edu.iu.dsc.tws.examples.basic.comms.batch.BGatherExample;
import edu.iu.dsc.tws.examples.basic.comms.batch.BKeyedPartitionExample;
import edu.iu.dsc.tws.examples.basic.comms.batch.BKeyedReduceExample;
import edu.iu.dsc.tws.examples.basic.comms.batch.BPartitionExample;
import edu.iu.dsc.tws.examples.basic.comms.batch.BReduceExample;
import edu.iu.dsc.tws.examples.basic.comms.stream.SAllGatherExample;
import edu.iu.dsc.tws.examples.basic.comms.stream.SAllReduceExample;
import edu.iu.dsc.tws.examples.basic.comms.stream.SBroadcastExample;
import edu.iu.dsc.tws.examples.basic.comms.stream.SGatherExample;
import edu.iu.dsc.tws.examples.basic.comms.stream.SKeyedPartitionExample;
import edu.iu.dsc.tws.examples.basic.comms.stream.SPartitionExample;
import edu.iu.dsc.tws.examples.basic.comms.stream.SReduceExample;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;

public class ExampleMain {
  private static final Logger LOG = Logger.getLogger(ExampleMain.class.getName());

  public static void main(String[] args) throws ParseException {
    // first load the configurations from command line and config files
    Config config = ResourceAllocator.loadConfig(new HashMap<>());

    Options options = new Options();
    options.addOption(Constants.ARGS_CONTAINERS, true, "Containers");
    options.addOption(Constants.ARGS_SIZE, true, "Size");
    options.addOption(Constants.ARGS_ITR, true, "Iteration");
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

    CommandLineParser commandLineParser = new DefaultParser();
    CommandLine cmd = commandLineParser.parse(options, args);
    int containers = Integer.parseInt(cmd.getOptionValue(Constants.ARGS_CONTAINERS));
    int size = Integer.parseInt(cmd.getOptionValue(Constants.ARGS_SIZE));
    int itr = Integer.parseInt(cmd.getOptionValue(Constants.ARGS_ITR));
    String operation = cmd.getOptionValue(Constants.ARGS_OPERATION);
    boolean stream = cmd.hasOption(Constants.ARGS_STREAM);

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

    String printInt = "0";
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

    // build JobConfig
    JobConfig jobConfig = new JobConfig();
    jobConfig.put(Constants.ARGS_ITR, Integer.toString(itr));
    jobConfig.put(Constants.ARGS_OPERATION, operation);
    jobConfig.put(Constants.ARGS_SIZE, Integer.toString(size));
    jobConfig.put(Constants.ARGS_CONTAINERS, Integer.toString(containers));
    jobConfig.put(Constants.ARGS_TASK_STAGES, taskStages);
    jobConfig.put(Constants.ARGS_GAP, gap);
    jobConfig.put(Constants.ARGS_FNAME, fName);
    jobConfig.put(Constants.ARGS_OUTSTANDING, outstanding);
    jobConfig.put(Constants.ARGS_THREADS, threads);
    jobConfig.put(Constants.ARGS_PRINT_INTERVAL, printInt);
    jobConfig.put(Constants.ARGS_DATA_TYPE, dataType);
    jobConfig.put(Constants.ARGS_INIT_ITERATIONS, intItr);

    // build the job
    Twister2Job twister2Job;
    if (!stream) {
      if (operation.equals("reduce")) {
        twister2Job = Twister2Job.newBuilder()
            .setName("reduce-batch-bench")
            .setWorkerClass(BReduceExample.class.getName())
            .setRequestResource(new WorkerComputeResource(2, 1024), containers)
            .setConfig(jobConfig)
            .build();
        // now submit the job
        Twister2Submitter.submitJob(twister2Job, config);
      } else if (operation.equals("keyedreduce")) {
        twister2Job = Twister2Job.newBuilder()
            .setName("keyed-reduce-batch-bench")
            .setWorkerClass(BKeyedReduceExample.class.getName())
            .setRequestResource(new WorkerComputeResource(2, 1024), containers)
            .setConfig(jobConfig)
            .build();
        // now submit the job
        Twister2Submitter.submitJob(twister2Job, config);
      } else if (operation.equals("partition")) {
        twister2Job = Twister2Job.newBuilder()
            .setName("partition-batch-bench")
            .setWorkerClass(BPartitionExample.class.getName())
            .setRequestResource(new WorkerComputeResource(2, 1024), containers)
            .setConfig(jobConfig)
            .build();
        // now submit the job
        Twister2Submitter.submitJob(twister2Job, config);
      } else if (operation.equals("keyedpartition")) {
        twister2Job = Twister2Job.newBuilder()
            .setName("keyed-partition-batch-bench")
            .setWorkerClass(BKeyedPartitionExample.class.getName())
            .setRequestResource(new WorkerComputeResource(2, 1024), containers)
            .setConfig(jobConfig)
            .build();
        // now submit the job
        Twister2Submitter.submitJob(twister2Job, config);
      } else if (operation.equals("gather")) {
        twister2Job = Twister2Job.newBuilder()
            .setName("partition-batch-bench")
            .setWorkerClass(BGatherExample.class.getName())
            .setRequestResource(new WorkerComputeResource(2, 1024), containers)
            .setConfig(jobConfig)
            .build();
        // now submit the job
        Twister2Submitter.submitJob(twister2Job, config);
      }
    } else {
      switch (operation) {
        case "reduce":
          twister2Job = Twister2Job.newBuilder()
              .setName("reduce-stream-bench")
              .setWorkerClass(SReduceExample.class.getName())
              .setRequestResource(new WorkerComputeResource(2, 1024), containers)
              .setConfig(jobConfig)
              .build();
          // now submit the job
          Twister2Submitter.submitJob(twister2Job, config);
          break;
        case "bcast":
          twister2Job = Twister2Job.newBuilder()
              .setName("bcast-stream-bench")
              .setWorkerClass(SBroadcastExample.class.getName())
              .setRequestResource(new WorkerComputeResource(2, 1024), containers)
              .setConfig(jobConfig)
              .build();
          // now submit the job
          Twister2Submitter.submitJob(twister2Job, config);
          break;
        case "partition":
          twister2Job = Twister2Job.newBuilder()
              .setName("partition-stream-bench")
              .setWorkerClass(SPartitionExample.class.getName())
              .setRequestResource(new WorkerComputeResource(2, 1024), containers)
              .setConfig(jobConfig)
              .build();
          // now submit the job
          Twister2Submitter.submitJob(twister2Job, config);
          break;
        case "keyedpartition":
          twister2Job = Twister2Job.newBuilder()
              .setName("keyed-partition-stream-bench")
              .setWorkerClass(SKeyedPartitionExample.class.getName())
              .setRequestResource(new WorkerComputeResource(2, 1024), containers)
              .setConfig(jobConfig)
              .build();
          // now submit the job
          Twister2Submitter.submitJob(twister2Job, config);
          break;
        case "gather":
          twister2Job = Twister2Job.newBuilder()
              .setName("gather-stream-bench")
              .setWorkerClass(SGatherExample.class.getName())
              .setRequestResource(new WorkerComputeResource(2, 1024), containers)
              .setConfig(jobConfig)
              .build();
          // now submit the job
          Twister2Submitter.submitJob(twister2Job, config);
          break;
        case "allreduce":
          twister2Job = Twister2Job.newBuilder()
              .setName("allreduce-stream-bench")
              .setWorkerClass(SAllReduceExample.class.getName())
              .setRequestResource(new WorkerComputeResource(2, 1024), containers)
              .setConfig(jobConfig)
              .build();
          // now submit the job
          Twister2Submitter.submitJob(twister2Job, config);
          break;
        case "allgather":
          twister2Job = Twister2Job.newBuilder()
              .setName("allgather-stream-bench")
              .setWorkerClass(SAllGatherExample.class.getName())
              .setRequestResource(new WorkerComputeResource(2, 1024), containers)
              .setConfig(jobConfig)
              .build();
          // now submit the job
          Twister2Submitter.submitJob(twister2Job, config);
          break;
        default:
          LOG.log(Level.SEVERE, "Un-supported operation: " + operation);
      }
    }
  }
}
