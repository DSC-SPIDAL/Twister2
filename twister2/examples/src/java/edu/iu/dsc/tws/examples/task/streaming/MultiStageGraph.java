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
package edu.iu.dsc.tws.examples.task.streaming;

import java.util.HashMap;
import java.util.List;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.JobConfig;
import edu.iu.dsc.tws.api.Twister2Job;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageTypes;
import edu.iu.dsc.tws.api.compute.IMessage;
import edu.iu.dsc.tws.api.compute.graph.ComputeGraph;
import edu.iu.dsc.tws.api.compute.graph.OperationMode;
import edu.iu.dsc.tws.api.compute.nodes.BaseCompute;
import edu.iu.dsc.tws.api.compute.nodes.BaseSource;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.SchedulerContext;
import edu.iu.dsc.tws.api.resource.Twister2Worker;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.rsched.job.Twister2Submitter;
import edu.iu.dsc.tws.task.ComputeEnvironment;
import edu.iu.dsc.tws.task.impl.ComputeConnection;
import edu.iu.dsc.tws.task.impl.ComputeGraphBuilder;

public class MultiStageGraph implements Twister2Worker {
  private static final Logger LOG = Logger.getLogger(MultiStageGraph.class.getName());

  @Override
  public void execute(WorkerEnvironment workerEnv) {

    ComputeEnvironment cEnv = ComputeEnvironment.init(workerEnv);

    GeneratorTask g = new GeneratorTask();
    ReduceTask rt = new ReduceTask();
    PartitionTask r = new PartitionTask();

    ComputeGraphBuilder builder = ComputeGraphBuilder.newBuilder(workerEnv.getConfig());
    builder.addSource("source", g, 4);
    ComputeConnection pc = builder.addCompute("compute", r, 4);
    pc.partition("source").viaEdge("partition-edge").withDataType(MessageTypes.OBJECT);
    ComputeConnection rc = builder.addCompute("sink", rt, 1);
    rc.reduce("compute")
        .viaEdge("compute-edge")
        .withReductionFunction((object1, object2) -> object1);
    builder.setMode(OperationMode.STREAMING);

    ComputeGraph graph = builder.build();
    graph.setGraphName("MultiTaskGraph");
    cEnv.getTaskExecutor().execute(graph, cEnv.getTaskExecutor().plan(graph));
  }

  private static class GeneratorTask extends BaseSource {
    private static final long serialVersionUID = -254264903510284748L;

    private int count = 0;

    @Override
    public void execute() {
      boolean wrote = context.write("partition-edge", "Hello");
      if (wrote) {
        count++;
        if (count % 100 == 0) {
          LOG.info(String.format("%d %d Source sent count : %d", context.getWorkerId(),
              context.globalTaskId(), count));
        }
      }
    }
  }

  private static class ReduceTask extends BaseCompute {
    private static final long serialVersionUID = -254264903510284791L;
    private int count = 0;

    @Override
    public boolean execute(IMessage message) {
      count++;
      LOG.info(String.format("%d %d Reduce received count: %d", context.getWorkerId(),
          context.globalTaskId(), count));
      return true;
    }
  }

  private static class PartitionTask extends BaseCompute {
    private static final long serialVersionUID = -254264903510284798L;

    private int count = 0;

    @Override
    public boolean execute(IMessage message) {
      if (message.getContent() instanceof List) {
        count += ((List) message.getContent()).size();
        for (Object o : (List) message.getContent()) {
          context.write("compute-edge", o);
        }
      }
      LOG.info(String.format("%d %d Partition Received count: %d", context.getWorkerId(),
          context.globalTaskId(), count));
      return true;
    }
  }

  public static void main(String[] args) {
    // first load the configurations from command line and config files
    Config config = ResourceAllocator.loadConfig(new HashMap<>());

    // build JobConfig
    HashMap<String, Object> configurations = new HashMap<>();
    configurations.put(SchedulerContext.THREADS_PER_WORKER, 8);

    // build JobConfig
    JobConfig jobConfig = new JobConfig();
    jobConfig.putAll(configurations);

    Twister2Job.Twister2JobBuilder jobBuilder = Twister2Job.newBuilder();
    jobBuilder.setJobName(MultiStageGraph.class.getName());
    jobBuilder.setWorkerClass(MultiStageGraph.class.getName());
    jobBuilder.addComputeResource(1, 512, 4);
    jobBuilder.setConfig(jobConfig);

    // now submit the job
    Twister2Submitter.submitJob(jobBuilder.build(), config);
  }
}
